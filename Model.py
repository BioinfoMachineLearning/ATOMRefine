#!/usr/bin/env python3
# encoding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl

def activation_func(activation):
    return  nn.ModuleDict([
        ['relu', nn.ReLU(inplace=True)],
        ['leaky_relu', nn.LeakyReLU(negative_slope=0.01, inplace=True)],
    ])[activation]

def norm_func(norm, n_channel):
    return  nn.ModuleDict([
        ['instance', nn.InstanceNorm1d(n_channel)],
    ])[norm]

class AtomEmbLayer(nn.Module):
    def __init__(self, n_in, n_out, 
                        atom_emb_in=7, atom_emb_h=256,
                        norm='instance', activation='relu', 
                        *args, **kwargs
                    ):
        super(AtomEmbLayer, self).__init__()
        self.norm = norm

        self.fn_atom_norm = norm_func(norm, atom_emb_in)
        self.fn_atom_linear = nn.Linear(atom_emb_in, atom_emb_h, bias=False)
        self.fn_atom_activation = activation_func(activation)
        self.fn_atom_norm2 = norm_func(norm, atom_emb_h)
        self.fn_atom_linear2 = nn.Linear(atom_emb_h, atom_emb_h, bias=False)

    def forward(self, G):
        atom_emb = G.ndata['atom_emb']   ## atom_emb: [40, 14, 7]
        # first layer
        atom_emb = self.fn_atom_norm(atom_emb)  ## atom_emb: [40, 14, 7]
        atom_emb = self.fn_atom_linear(atom_emb)  ## atom_emb: [40, 14, 256]
        atom_emb = torch.mean(atom_emb, 1)  ## atom_emb: [40, 256]
        atom_emb = self.fn_atom_activation(atom_emb)  ## atom_emb: [40, 256]
        # second layer
        atom_emb = self.fn_atom_norm2(atom_emb.unsqueeze(1)).squeeze() if self.norm=="instance" else self.fn_atom_norm2(atom_emb)  ## atom_emb: [40, 256]
        atom_emb = self.fn_atom_linear2(atom_emb)  ## atom_emb: [40, 256]
        atom_emb = self.fn_atom_activation(atom_emb)  ## atom_emb: [40, 256]
        # cat
        x = torch.cat((G.ndata['nfeat'], atom_emb), dim=1)
        G.ndata['nfeat'] = x  ## atom_emb: [40, 284]

        return G

class EdgeApplyModule(nn.Module):
    def __init__(self, n_in, n_out, norm='instance', activation='leaky_relu', LSTM=False, *args, **kwargs):
        super(EdgeApplyModule, self).__init__()
        self.norm, self.LSTM = norm, LSTM

        self.fn_norm = norm_func(norm, n_in)
        self.fn_linear = nn.Linear(n_in, n_out)
        self.fn_activation = activation_func(activation)
        if self.LSTM: self.fn_lstm = nn.LSTMCell(n_out, n_out, bias=False)
        self.attn_fc = nn.Linear(n_out, 1, bias=False)

    def forward(self, edges):
        # edges.src['nfeat']:[664, 284], edges.data['efeat']:[664, 15], edges.dst['nfeat']:[664, 284]
        x = torch.cat((edges.src['nfeat'], edges.data['efeat'], edges.dst['nfeat']), 1)  # [664, 583]
        x = self.fn_norm(x.unsqueeze(1)).squeeze() if self.norm=="instance" else self.fn_norm(x)  # [664, 583]
        x = self.fn_linear(x)  # [664, 256]
        x = self.fn_activation(x)  # [664, 256]
        if self.LSTM:
            # hidden: efeat, cell: efeat_c
            if not 'efeat_c' in edges.data: edges.data['efeat_c'] = torch.zeros_like(edges.data['efeat'])
            x, c = self.fn_lstm(x, (edges.data['efeat'], edges.data['efeat_c']))

        attn = self.attn_fc(x)  # [664, 1]
        if self.LSTM: return {'efeat': x, 'attn': attn, 'efeat_c': c}
        return {'efeat': x, 'attn': attn}

def message_func(edge):
    return {'_efeat': edge.data['efeat'], '_attn': edge.data['attn']}

def reduce_func(node):
    alpha = F.softmax(node.mailbox['_attn'], dim=1)  # node.batch_size: dgl will batch nodes with same degrees node.mailbox['_attn']: [1, 6, 1]
    attn_feat = torch.sum(alpha * node.mailbox['_efeat'], dim=1)  # alpha: [1, 6, 1] node.mailbox['_efeat']: [1, 6, 256]  attn_feat: [1,256]
    feat = torch.cat((node.data['nfeat'], attn_feat), 1)  # feat: [1, 540]
    return {'_nfeat': feat}

class NodeApplyModule(nn.Module):
    def __init__(self, n_in, n_out, norm='instance', activation='relu', LSTM=False, 
                        *args, **kwargs
                    ):
        super(NodeApplyModule, self).__init__()
        self.norm, self.LSTM = norm, LSTM

        self.fn_norm = norm_func(norm, n_in)
        self.fn_linear = nn.Linear(n_in, n_out)
        self.fn_activation = activation_func(activation)
        if self.LSTM: self.fn_lstm = nn.LSTMCell(n_out, n_out, bias=False)

    def forward(self, nodes):
        x = nodes.data['_nfeat']
        x = self.fn_norm(x.unsqueeze(1)).squeeze() if self.norm=="instance" else self.fn_norm(x)
        x = self.fn_linear(x)
        x = self.fn_activation(x)

        if self.LSTM:
            if not 'nfeat_c' in nodes.data: nodes.data['nfeat_c'] = torch.zeros_like(nodes.data['nfeat'])
            x, c = self.fn_lstm(x, (nodes.data['nfeat'], nodes.data['nfeat_c']))
            return {'nfeat': x, 'nfeat_c': c}

        return {'nfeat': x}

class MessagePassingLayer(nn.Module):
    def __init__(self,
                node_n_in, node_n_out, edge_n_in, edge_n_out,
                norm='instance', activation='relu', 
                LSTM=False, last_layer=False, 
                atom_emb=False, atom_emb_h=256,
                *args, **kwargs):
        super(MessagePassingLayer, self).__init__()
        self.LSTM, self.last_layer = LSTM, last_layer

        self.edge_update = EdgeApplyModule(edge_n_in+node_n_in*2, edge_n_out, norm=norm, LSTM=LSTM)
        self.node_update = NodeApplyModule(node_n_in+edge_n_out, node_n_out, norm, activation, LSTM=LSTM, )
    
    def forward(self, G):        
        G.apply_edges(self.edge_update)
        G.update_all(message_func, reduce_func, self.node_update)
        if self.last_layer: G.apply_edges(self.edge_update)
        return G

class GNN(nn.Module):
    def __init__(self,
                 node_n_in=28, node_n_hidden=256, edge_n_in=15, edge_n_hidden=256, n_layers=10, n_output=37,
                 LSTM=True, atom_emb_in=7, atom_emb_h=256, norm='instance', activation='relu', 
                 distCB=True, QA=False, *args, **kwargs):
        super(GNN, self).__init__()
        self.distCB, self.QA = distCB, QA

        self.layers = nn.ModuleList()
        # AtomEmbLayer
        self.layers.append(AtomEmbLayer(node_n_in, node_n_hidden, atom_emb_in, atom_emb_h, norm, activation, ))
        # first MessagePassingLayer                 
        self.layers.append(MessagePassingLayer(node_n_in+atom_emb_h, node_n_hidden, edge_n_in, edge_n_hidden, norm, activation, ))
        # intermediate MessagePassingLayers
        for i in range(n_layers - 2):
            self.layers.append(MessagePassingLayer(node_n_hidden, node_n_hidden, edge_n_hidden, edge_n_hidden, norm, activation, LSTM=True))
        # last MessagePassingLayer
        self.layers.append(MessagePassingLayer(node_n_hidden, node_n_hidden, edge_n_hidden, edge_n_hidden, norm, activation, LSTM=True, last_layer=True))

        # distCB layer
        if self.distCB:
            self.output_layer = nn.Sequential(nn.Linear(edge_n_hidden, n_output))

        # QA layer
        if self.QA:
            self.global_qa_linear = nn.Linear(node_n_hidden+edge_n_hidden, 1)
            self.local_qa_linear = nn.Linear(node_n_hidden, 1)
            self.sigmoid = nn.Sigmoid()
            
    def forward(self, G):
        for layer in self.layers:
            G = layer(G)

        output = {}        
        # distCB
        if self.distCB:
            output['distCB'] = self.output_layer(G.edata['efeat'])  #[664, 37]
            print(G.edata['nfeat'])

        # global and local QA
        if self.QA:
            h_global = torch.cat((dgl.mean_nodes(G, 'nfeat'), dgl.mean_edges(G, 'efeat')), 1)
            y_global = self.sigmoid(self.global_qa_linear(h_global))
            output['global_lddt'] = y_global
            y_local = self.sigmoid(self.local_qa_linear(G.ndata['nfeat']))
            output['local_lddt'] = y_local

        return output