import time
import torch

import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F

from transformers import get_linear_schedule_with_warmup

import dgl
from dgl.nn import GraphConv

import numpy as np

from SE3_network import SE3Transformer,TFN
from Transformer import LayerNorm

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
    '''
    Input:
        - xyz: current backbone cooordinates (B, L, 3, 3)
        - pair: pair features from Trunk (B, L, L, E)
        - idx: residue index from ground truth pdb
    Output:
        - G: defined graph
    '''

    B, L = xyz.shape[:2]
    device = xyz.device
    
    # distance map from current CA coordinates
    D = torch.cdist(xyz[:,:,0,:], xyz[:,:,0,:]) + torch.eye(L, device=device).unsqueeze(0)*999.9  # (B, L, L)
    # seq sep
    sep = idx[:,None,:] - idx[:,:,None]
    sep = sep.abs() + torch.eye(L, device=device).unsqueeze(0)*999.9
    
    # get top_k neighbors
    D_neigh, E_idx = torch.topk(D, min(top_k, L), largest=False) # shape of E_idx: (B, L, top_k)
    topk_matrix = torch.zeros((B, L, L), device=device)
    topk_matrix.scatter_(2, E_idx, 1.0)

    # put an edge if any of the 3 conditions are met:
    #   1) |i-j| <= kmin (connect sequentially adjacent residues)
    #   2) top_k neighbors
    cond = torch.logical_or(topk_matrix > 0.0, sep < kmin)
    b,i,j = torch.where(cond)

    src = b*L+i
    tgt = b*L+j
    G = dgl.graph((src, tgt), num_nodes=B*L).to(device)
    G.edata['d'] = (xyz[b,j,0,:] - xyz[b,i,0,:]).detach() # no gradient through basis function
    G.edata['w'] = pair[b,i,j]
    return G

def get_bonded_neigh(idx):
    '''
    Input:
        - idx: residue indices of given sequence (B,L)
    Output:
        - neighbor: bonded neighbor information with sign (B, L, L, 1)
    '''
    neighbor = idx[:,None,:] - idx[:,:,None]
    neighbor = neighbor.float()
    sign = torch.sign(neighbor) # (B, L, L)
    neighbor = torch.abs(neighbor)
    neighbor[neighbor > 1] = 0.0
    neighbor = sign * neighbor 
    return neighbor.unsqueeze(-1)

def rbf(D):
    # Distance radial basis function
    D_min, D_max, D_count = 0., 20., 36
    D_mu = torch.linspace(D_min, D_max, D_count).to(D.device)
    D_mu = D_mu[None,:]
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)
    RBF = torch.exp(-((D_expand - D_mu) / D_sigma)**2)
    return RBF

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
        #print(pdbs[0])
        bsz = 1
        L = seq_l[0]

        init_xyz = init_xyz.reshape(L, 1, 3)
        #init_xyz = torch.cat([init_xyz, init_xyz*torch.tensor([1,1,-1], dtype=init_xyz.dtype, device=init_xyz.device)])
        init_xyz = init_xyz.reshape(bsz,L,1,3)
        
        init_pos = init_pos.reshape(L, 1, 3)
        #init_pos = torch.cat([init_pos, init_pos*torch.tensor([1,1,-1], dtype=init_pos.dtype, device=init_pos.device)])
        init_pos = init_pos.reshape(bsz,L,1,3)
        
        #nodes = torch.cat([nodes, nodes])
        #pair = torch.cat([pair, pair])
        pair = pair.reshape(bsz,L,L,12)
        
        idx = torch.arange(L).long().view(1, L)
        idx = idx.to(device)
        #idx = torch.cat([idx, idx])
        idx = idx.reshape(bsz,L)
        
        nodes = self.norm_node1(nodes.unsqueeze(1))
        nodes = nodes.reshape(bsz,L,58)
        nodes = self.norm_node2(self.embed_node(nodes))
        
        pair = pair.permute(0,3,1,2)
        pair = self.norm_edge1(pair)
        pair = pair.permute(0,2,3,1)
        pair = self.norm_edge2(self.embed_e1(pair))
        
        #neighbor = get_bonded_neigh(idx)
        #neighbor = neighbor.to(device)
        rbf_feat = rbf(torch.cdist(init_xyz[:,:,0,:], init_xyz[:,:,0,:]))
        rbf_feat = rbf_feat.to(device)

        bond = bond.reshape(1,L,L,1)
        #pair = torch.cat((pair, rbf_feat), dim=-1)
        pair = torch.cat((pair, rbf_feat, bond), dim=-1)
        pair = self.norm_edge3(self.embed_e2(pair)) 
        
        # define graph
        #xyz:[2, 138, 3, 3], pair:[2, 138, 138, 32], idx:[2, 138], top_k:64
        G = make_graph(init_xyz, pair, idx, top_k=128)
        l1_feats = init_pos # l1 features = displacement vector to CA
        l1_feats = l1_feats.reshape(bsz*L,-1, 3)
        
        # apply SE(3) Transformer & update coordinates
        # node.reshape(B*L, -1, 1):[276, 32, 1], l1_feats:[276, 3, 3]
        # print(pdbs[0], nodes.size(), l1_feats.size())
        shift = self.se3(G, nodes.reshape(bsz*L, -1, 1), l1_feats) # 0: [276, 8, 1] 1: [276, 3, 3]
        offset = shift['1'].reshape(bsz, L, -1, 3) # (B, L, 3, 3)  torch.Size([2, 138, 3, 3])
        
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

        batch_dictionary = {'loss': loss}
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        return batch_dictionary

    def validation_step(self, batch, batch_idx):
        nodes, pair, bond, init_pos, init_xyz, init_atom, init_CA, label_xyz, seq_l, pdbs = batch
        bsz = 1
        L = seq_l[0]

        init_xyz = init_xyz.reshape(L, 1, 3)
        #init_xyz = torch.cat([init_xyz, init_xyz*torch.tensor([1,1,-1], dtype=init_xyz.dtype, device=init_xyz.device)])
        init_xyz = init_xyz.reshape(bsz,L,1,3)
        
        init_pos = init_pos.reshape(L, 1, 3)
        #init_pos = torch.cat([init_pos, init_pos*torch.tensor([1,1,-1], dtype=init_pos.dtype, device=init_pos.device)])
        init_pos = init_pos.reshape(bsz,L,1,3)
        
        #nodes = torch.cat([nodes, nodes])
        #pair = torch.cat([pair, pair])
        pair = pair.reshape(bsz,L,L,12)
        
        idx = torch.arange(L).long().view(1, L)
        idx = idx.to(device)
        #idx = torch.cat([idx, idx])
        idx = idx.reshape(bsz,L)
        
        nodes = self.norm_node1(nodes.unsqueeze(1))
        nodes = nodes.reshape(bsz,L,58)
        nodes = self.norm_node2(self.embed_node(nodes))
        
        pair = pair.permute(0,3,1,2)
        pair = self.norm_edge1(pair)
        pair = pair.permute(0,2,3,1)
        pair = self.norm_edge2(self.embed_e1(pair))
        
        #neighbor = get_bonded_neigh(idx)
        #neighbor = neighbor.to(device)
        rbf_feat = rbf(torch.cdist(init_xyz[:,:,0,:], init_xyz[:,:,0,:]))
        rbf_feat = rbf_feat.to(device)

        bond = bond.reshape(1,L,L,1)
        #pair = torch.cat((pair, rbf_feat), dim=-1)
        pair = torch.cat((pair, rbf_feat, bond), dim=-1)
        pair = self.norm_edge3(self.embed_e2(pair)) 
        
        # define graph
        #xyz:[2, 138, 3, 3], pair:[2, 138, 138, 32], idx:[2, 138], top_k:64
        G = make_graph(init_xyz, pair, idx, top_k=128)
        l1_feats = init_pos # l1 features = displacement vector to CA
        l1_feats = l1_feats.reshape(bsz*L,-1, 3)
        
        # apply SE(3) Transformer & update coordinates
        #node.reshape(B*L, -1, 1):[276, 32, 1], l1_feats:[276, 3, 3]
        shift = self.se3(G, nodes.reshape(bsz*L, -1, 1), l1_feats) # 0: [276, 8, 1] 1: [276, 3, 3]
        offset = shift['1'].reshape(bsz, L, -1, 3) # (B, L, 3, 3)  torch.Size([2, 138, 3, 3])
        
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
        
        # batch_dictionary = {'val_loss': loss}
        # self.log('val_loss', loss, on_epoch=True, sync_dist=True)
        # return batch_dictionary
        return {'rmse':loss}

    def validation_epoch_end(self,outputs):
        val_loss = torch.stack([x['rmse'] for x in outputs]).mean()
        log = {'avg_rmse': val_loss}
        self.log('avg_rmse', val_loss)
        return {'log': log}


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, betas=(0.9, 0.999), weight_decay=0.001)
        #scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=15000, num_training_steps = -1)
        #return [optimizer],[scheduler]
        return [optimizer]


    # def optimizer_step(self, current_epoch, batch_idx, optimizer,optimizer_idx, second_order_closure=None, on_tpu=False, using_native_amp=False, using_lbfgs=False):
    #     if self.trainer.global_step > 500:
    #         lr_scale = min(1., float(self.trainer.global_step + 1) / 500.)
    #         for pg in optimizer.param_groups:
    #             #pg['lr'] = lr_scale * self.learning_rate
    #             pg['lr'] = self.learning_rate
    #     optimizer.step()
    #     optimizer.zero_grad()
