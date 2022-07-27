import time
import torch

import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F

import dgl
from dgl.nn import GraphConv

import numpy as np

from SE3_network import SE3Transformer,TFN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SE3_param = {
        "num_layers"    : 2,
        "num_channels"  : 16,
        "num_degrees"   : 2,
        "l0_in_features": 32,
        "l0_out_features": 8,
        "l1_in_features": 1,
        "l1_out_features": 1,
        "num_edge_features": 32,
        "div": 2,
        "n_heads": 4
        }

tfn_param = {
        "num_layers"    : 2,
        "num_channels"  : 16,
        "num_degrees"   : 2,
        "l0_in_features": 32,
        "l0_out_features": 8,
        "l1_in_features": 1,
        "l1_out_features": 1,
        "num_edge_features": 32,
        }

def make_graph(xyz, pair, idx, top_k=18, kmin=12):
    B, L = xyz.shape[:2]
    device = xyz.device
    
    # distance map from current CA coordinates
    D = torch.cdist(xyz[:,:,0,:], xyz[:,:,0,:]) + torch.eye(L, device=device).unsqueeze(0)*999.9  # (B, L, L)
    
    # sequence separation
    sep = idx[:,None,:] - idx[:,:,None]
    sep = sep.abs() + torch.eye(L, device=device).unsqueeze(0)*999.9
    
    # get top_k neighbors
    D_neigh, E_idx = torch.topk(D, min(top_k, L), largest=False) # shape of E_idx: (B, L, top_k)
    topk_matrix = torch.zeros((B, L, L), device=device)
    topk_matrix.scatter_(2, E_idx, 1.0)
    cond = torch.logical_or(topk_matrix > 0.0, sep < kmin)
    b,i,j = torch.where(cond)

    src = b*L+i
    tgt = b*L+j
    G = dgl.graph((src, tgt), num_nodes=B*L).to(device)
    G.edata['d'] = (xyz[b,j,0,:] - xyz[b,i,0,:]).detach()
    G.edata['w'] = pair[b,i,j]
    return G

def rbf(D):
    # Distance radial basis function
    D_min, D_max, D_count = 0., 20., 36
    D_mu = torch.linspace(D_min, D_max, D_count).to(D.device)
    D_mu = D_mu[None,:]
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)
    RBF = torch.exp(-((D_expand - D_mu) / D_sigma)**2)
    return RBF

class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(d_model))
        self.b_2 = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = torch.sqrt(x.var(dim=-1, keepdim=True, unbiased=False) + self.eps)
        x = self.a_2*(x-mean)
        x /= std
        x += self.b_2
        return x

class SE3Refine(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super(SE3Refine, self).__init__()
        self.learning_rate = 0.001

        #self.se3 = TFN(**tfn_param)
        self.se3 = SE3Transformer(**SE3_param)
        self.lossfn = nn.MSELoss(reduction='sum')
        self.norm_node1 = nn.InstanceNorm1d(58)
        self.embed_node = nn.Linear(58, SE3_param['l0_in_features'])
        self.norm_node2 = LayerNorm(SE3_param['l0_in_features'])
        self.norm_edge1 = nn.InstanceNorm2d(12)
        self.embed_e1 =  nn.Linear(12, SE3_param['num_edge_features'])
        self.embed_e2 = nn.Linear(SE3_param['num_edge_features']+37, SE3_param['num_edge_features'])
        self.norm_edge2 = LayerNorm(SE3_param['num_edge_features'])
        self.norm_edge3 = LayerNorm(SE3_param['num_edge_features'])


    def training_step(self, batch, batch_idx):
        nodes, pair, bond, init_pos, init_xyz, init_atom, init_CA, label_xyz, seq_l, pdbs = batch
        bsz = 1
        L = seq_l[0]

        init_xyz = init_xyz.reshape(L, 1, 3)
        init_xyz = init_xyz.reshape(bsz,L,1,3)
        
        init_pos = init_pos.reshape(L, 1, 3)
        init_pos = init_pos.reshape(bsz,L,1,3)
        
        pair = pair.reshape(bsz,L,L,12)
        
        idx = torch.arange(L).long().view(1, L)
        idx = idx.to(device)
        idx = idx.reshape(bsz,L)
        
        nodes = self.norm_node1(nodes.unsqueeze(1))
        nodes = nodes.reshape(bsz,L,58)
        nodes = self.norm_node2(self.embed_node(nodes))
        
        pair = pair.permute(0,3,1,2)
        pair = self.norm_edge1(pair)
        pair = pair.permute(0,2,3,1)
        pair = self.norm_edge2(self.embed_e1(pair))
        
        rbf_feat = rbf(torch.cdist(init_xyz[:,:,0,:], init_xyz[:,:,0,:]))
        rbf_feat = rbf_feat.to(device)

        bond = bond.reshape(1,L,L,1)
        pair = torch.cat((pair, rbf_feat, bond), dim=-1)
        pair = self.norm_edge3(self.embed_e2(pair)) 
        
        # define graph
        G = make_graph(init_xyz, pair, idx, top_k=128)
        l1_feats = init_pos 
        l1_feats = l1_feats.reshape(bsz*L,-1, 3)
        
        # SE(3) Transformer
        shift = self.se3(G, nodes.reshape(bsz*L, -1, 1), l1_feats)
        offset = shift['1'].reshape(bsz, L, -1, 3)
        
        offset = offset[0,:,0,:]
        res_num, _ = init_CA.size()
        start = 0
        end = 0
        xyz_new = []
        for i in range(res_num):
            start = end
            end += init_atom[i]
            xyz_new.append(init_CA[i] + offset[start:end,:])
        
        xyz_new = torch.cat(xyz_new)
        loss = torch.sqrt(self.lossfn(xyz_new, label_xyz)/L)

        batch_dictionary = {'loss': loss}
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        return batch_dictionary

    def validation_step(self, batch, batch_idx):
        nodes, pair, bond, init_pos, init_xyz, init_atom, init_CA, label_xyz, seq_l, pdbs = batch
        bsz = 1
        L = seq_l[0]

        init_xyz = init_xyz.reshape(L, 1, 3)
        init_xyz = init_xyz.reshape(bsz,L,1,3)
        
        init_pos = init_pos.reshape(L, 1, 3)
        init_pos = init_pos.reshape(bsz,L,1,3)
        
        pair = pair.reshape(bsz,L,L,12)
        
        idx = torch.arange(L).long().view(1, L)
        idx = idx.to(device)
        idx = idx.reshape(bsz,L)
        
        nodes = self.norm_node1(nodes.unsqueeze(1))
        nodes = nodes.reshape(bsz,L,58)
        nodes = self.norm_node2(self.embed_node(nodes))
        
        pair = pair.permute(0,3,1,2)
        pair = self.norm_edge1(pair)
        pair = pair.permute(0,2,3,1)
        pair = self.norm_edge2(self.embed_e1(pair))
        
        rbf_feat = rbf(torch.cdist(init_xyz[:,:,0,:], init_xyz[:,:,0,:]))
        rbf_feat = rbf_feat.to(device)

        bond = bond.reshape(1,L,L,1)
        pair = torch.cat((pair, rbf_feat, bond), dim=-1)
        pair = self.norm_edge3(self.embed_e2(pair)) 
        
        # define graph
        G = make_graph(init_xyz, pair, idx, top_k=128)
        l1_feats = init_pos
        l1_feats = l1_feats.reshape(bsz*L,-1, 3)
        
        # SE(3) Transformer
        shift = self.se3(G, nodes.reshape(bsz*L, -1, 1), l1_feats)
        offset = shift['1'].reshape(bsz, L, -1, 3)
        
        offset = offset[0,:,0,:]
        res_num,_ = init_CA.size()
        start = 0
        end = 0
        xyz_new = []
        for i in range(res_num):
            start = end
            end += init_atom[i]
            xyz_new.append(init_CA[i] + offset[start:end,:])
        
        xyz_new = torch.cat(xyz_new)
        loss = torch.sqrt(self.lossfn(xyz_new, label_xyz)/L)
        return {'rmse':loss}

    def validation_epoch_end(self,outputs):
        val_loss = torch.stack([x['rmse'] for x in outputs]).mean()
        log = {'avg_rmse': val_loss}
        self.log('avg_rmse', val_loss)
        return {'log': log}


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, betas=(0.9, 0.999), weight_decay=0.001)
        return [optimizer]