#!/usr/bin/env python3
# encoding: utf-8

import numpy as np
import torch
from torch.utils.data import Dataset
import Utils as Utils
from covalent import cal_covalent
import os,re

class Data(Dataset):
    def __init__(self, start_pdbs, test_mode = False
                ):
        self.start_pdbs = start_pdbs
        self.data_len = len(self.start_pdbs)
        self.test_mode = test_mode
    
    def __len__(self):
        return self.data_len
    
    def __get_seq_feature(self, pdb_file):
        seq = Utils.get_seqs_from_pdb(pdb_file)
        # node_feat
        node_feat = {
            'onehot': Utils.get_seq_onehot(seq),
            'rPosition': Utils.get_rPos(seq),
        }
        return node_feat, len(seq)

    def __get_struc_feat(self, pdb_file, true_pdb, seq_len):
        # node feat
        node_feat = {}
        # atom_emb
        embedding, atom_pos, atom_lst, CA_lst, res_atom = Utils.get_atom_emb(pdb_file, true_pdb, [1, seq_len])
        node_feat['atom_emb'] = {
            'embedding': embedding,
            'atom_pos': atom_pos,
            'atom_lst': atom_lst,
            'CA_lst': CA_lst,
            'res_atom': res_atom
        }
        return node_feat

    def __getitem__(self, idx):
        if torch.is_tensor(idx): idx = idx.tolist()
        pdb_file = self.start_pdbs[idx][0]
        #print(pdb_file, 'gen feature...')
        feature = {"node": None, "edge": None}
        
        if self.test_mode:
            true_pdb = pdb_file
        else:
            true_pdb = self.start_pdbs[idx][1]

        # extract feature
        seq_node_feat, seq_len = self.__get_seq_feature(pdb_file)
        struc_node_feat = self.__get_struc_feat(pdb_file, true_pdb, seq_len)
        covalent = cal_covalent(pdb_file)
        
        true_node_feat = self.__get_struc_feat(true_pdb, true_pdb, seq_len)

        res_num = seq_node_feat['onehot'].shape[0]
        com_emb = []
        atom_xyz = []
        for i in range(res_num):
            seq_emb = seq_node_feat['onehot'][i]
            atom_emb = struc_node_feat['atom_emb']['embedding'][i]
            seq_emb = np.tile(seq_emb,(atom_emb.shape[0],1))
            com_emb.append(np.concatenate((seq_emb,atom_emb[:,0:37]),axis=1))
            atom_xyz.append(atom_emb[:,37:40])
        com_emb = np.vstack(com_emb)
        atom_xyz = np.vstack(atom_xyz)
        
        atom_pos = struc_node_feat['atom_emb']['atom_pos']
        atom_pos = np.vstack(atom_pos)
        true_pos = true_node_feat['atom_emb']['atom_pos']
        true_pos = np.vstack(true_pos)
        
        atom_lst = struc_node_feat['atom_emb']['atom_lst']
        res_atom = struc_node_feat['atom_emb']['res_atom']
        CA_lst = struc_node_feat['atom_emb']['CA_lst']
        CA_pos = np.vstack(CA_lst)

        if self.test_mode:
            p, q, k, t = Utils.set_lframe(pdb_file, atom_xyz, atom_lst, [1, seq_len])
        else:
            init_dir = "/storage/htc/bdm/tianqi/data/init_model"
            tar = os.path.basename(pdb_file)
            tar = re.sub("\.pdb","",tar)
            if not os.path.exists(init_dir+"/p/"+tar+".npy"):
                p, q, k, t = Utils.set_lframe(pdb_file, atom_xyz, atom_lst, [1, seq_len])
            else:
                p = np.load(init_dir+"/p/"+tar+".npy")
                q = np.load(init_dir+"/q/"+tar+".npy")
                k = np.load(init_dir+"/k/"+tar+".npy")
                t = np.load(init_dir+"/t/"+tar+".npy")
        pairs = np.concatenate([p,q,k,t],axis=-1)
        return [com_emb, atom_xyz, atom_pos, atom_lst, CA_pos, true_pos, pdb_file, pairs, covalent, res_atom]
