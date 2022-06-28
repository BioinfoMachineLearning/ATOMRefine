import os,re
import time
import argparse
import pytorch_lightning as pl

import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils import data
from torch.utils.data import DataLoader

from Data import Data
from network2 import SE3Refine
from network2 import get_bonded_neigh,rbf,make_graph

import dgl
import numpy as np

import warnings
warnings.filterwarnings("ignore")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def refine_coords(coords, n_steps=1):
    vdw_dist, cov_dist = 3.0, 3.78
    k_vdw, k_cov = 100.0, 100.0
    min_speed = 0.001

    for i in range(n_steps):
        n_res = coords.size(0)
        accels = coords * 0

        # Steric clashing
        crep = coords.unsqueeze(0).expand(n_res, -1, -1)
        diffs = crep - crep.transpose(0, 1)
        dists = diffs.norm(dim=2).clamp(min=0.01, max=10.0)
        norm_diffs = diffs / dists.unsqueeze(2)
        violate = (dists < vdw_dist).to(torch.float) * (vdw_dist - dists)
        forces = k_vdw * violate
        pair_accels = forces.unsqueeze(2) * norm_diffs
        accels += pair_accels.sum(dim=0)

        # Adjacent C-alphas
        diffs = coords[1:] - coords[:-1]
        dists = diffs.norm(dim=1).clamp(min=0.1)
        norm_diffs = diffs / dists.unsqueeze(1)
        violate = (dists - cov_dist).clamp(max=3.0)
        forces = k_cov * violate
        accels_cov = forces.unsqueeze(1) * norm_diffs
        accels[:-1] += accels_cov
        accels[1: ] -= accels_cov

        coords = coords + accels.clamp(min=-100.0, max=100.0) * min_speed

    return coords

def xyz2pdb(tar, xyz, res_atom, outdir):
    f = open(outdir+"/"+tar+".pdb","w")
    i = 1
    j = 1
    for res in res_atom:
        res_name = res[0:3]
        #print(res_name)
        for atom in res_atom[res]:
            atom_seq= j
            res_seq= i
            x = xyz[j-1][0]
            y = xyz[j-1][1]
            z = xyz[j-1][2]
            line="{:6s}{:5d} {:^4s} {:3s} {:1s}{:4d}    {:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}".format('ATOM',atom_seq,atom,res_name,'A',res_seq,x,y,z,0,0)
            f.write(line+"\n")
            j = j+1
        i = i+1
    f.close()

def _collate_fn(batch):
    nodes = []
    pairs = []
    bonds = []
    init_xyz = []
    init_pos = []
    init_atom = []
    init_CA = []
    label_xyz = []
    seq_l = []
    pdbs = []
    #res_atom = []
    for iter,item in enumerate(batch):
        nodes.append(item[0])
        init_xyz.append(item[1])
        init_pos.append(item[2])
        init_atom = item[3]
        init_CA.append(item[4])
        label_xyz.append(item[5])
        pdbs.append(item[6])
        pairs.append(item[7])
        bonds.append(item[8])
        #res_atom = item[9]
        l = item[1].shape[0]
        seq_l.append(l)
    bsz = len(init_xyz)
    nodes = [torch.from_numpy(item).float() for item in nodes]
    nodes = torch.cat(nodes)
    pairs = [torch.from_numpy(item).float() for item in pairs]
    pairs = torch.cat(pairs)
    bonds = [torch.from_numpy(item).float() for item in bonds]
    bonds = torch.cat(bonds)
    init_xyz = [torch.from_numpy(item).float() for item in init_xyz]
    init_xyz = torch.cat(init_xyz)
    init_pos = [torch.from_numpy(item).float() for item in init_pos]
    init_pos = torch.cat(init_pos)
    init_CA = [torch.from_numpy(item).float() for item in init_CA]
    init_CA = torch.cat(init_CA)
    label_xyz = [torch.from_numpy(item).float() for item in label_xyz]
    label_xyz = torch.cat(label_xyz)

    return nodes, pairs, bonds, init_xyz, init_pos, init_atom, init_CA, label_xyz, seq_l, pdbs#, res_atom


def dir_path(string):
    if os.path.isdir(string):
        return os.path.abspath(string)
    else:
        raise NotADirectoryError(string)

def datalist(pdb_dir,true_dir,train_dir,lst,tm_thre=0.0,test_mode=False):
    train_lst = []
    for line in open(train_dir+"/"+lst):
        line = line.rstrip().split()
        tar,l,tm = line[0],line[1],float(line[2])
        if tm >= tm_thre:
            continue
        if test_mode:
            train_lst.append(["/storage/htc/bdm/tianqi/capsule-5769140/data/init_model_casp14/"+tar+".pdb","/storage/htc/bdm/tianqi/capsule-5769140/data/init_model_casp14/"+tar+".npy","/storage/htc/bdm/tianqi/capsule-5769140/data/init_model_casp14/"+tar+".npy",tm])
        else:
            #train_lst.append([pdb_dir+"/all/"+tar+"_rotate.pdb",true_dir+"/pdb/"+tar+".pdb",tm])
            train_lst.append([pdb_dir+"/clean/"+tar+".pdb",true_dir+"/clean/"+tar+".pdb",tm])
    return train_lst

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Refinement')
    ap.add_argument('--data', type=dir_path, required=True, default='training list dir')
    ap.add_argument('--network', type=str, required=False, default='SE3Refine')
    ap.add_argument('--out_path', type=str, required=False,
                    default='output')
    ap.add_argument('--num_gpus', type=int, required=False, default=1)
    ap.add_argument('--lst', type=int, required=False, default=1)
    ap.add_argument('--num_workers', type=int, required=False, default=4)
    ap.add_argument('--test_size', type=int, required=False, default=50)
    ap.add_argument('--epochs', type=int, required=False, default=5)
    ap.add_argument('--time_limit', type=int, required=False, default=0)
    ap.add_argument('--batch_size', type=int, required=False, default=1)
    ap.add_argument('--test_percent_check', type=float, required=False, default=1.0)
    ap.add_argument('--test_seed', type=int, required=False, default=None)
    ap.add_argument('--debug', required=False, action='store_true')
    ap.add_argument('--test', required=False, action='store_true')

    args = ap.parse_args()
    train_dir = args.data
    network = args.network
    out_path = args.out_path
    num_gpus = args.num_gpus
    lst = args.lst
    test_size = args.test_size
    num_workers = args.num_workers
    epochs = args.epochs
    time_limit = args.time_limit
    batch_size = args.batch_size
    test_percent_check = args.test_percent_check
    test_seed = args.test_seed
    debug_mode = args.debug
    test_mode = args.test

    pl.utilities.seed.seed_everything(seed=test_seed)

    src_dir = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))
    pdb_dir = "/storage/htc/bdm/tianqi/capsule-5769140/data/init_model"
    true_dir = "/storage/htc/bdm/tianqi/capsule-5769140/data/true_model"
    train_lst = datalist(pdb_dir, true_dir, train_dir,"train"+str(lst)+".lst",tm_thre=1.1)
    val_lst = datalist(pdb_dir, true_dir, train_dir,"valid"+str(lst)+".lst",tm_thre=1.1)
    test_lst = datalist(pdb_dir, true_dir, train_dir,"test_clean.lst",tm_thre=1.1,test_mode=False)
    
    train_dataset = Data(train_lst)
    train_loader = DataLoader(train_dataset, shuffle=True, pin_memory=False, num_workers=args.num_workers,batch_size=args.batch_size,collate_fn=_collate_fn)

    val_dataset = Data(val_lst)
    val_loader = DataLoader(val_dataset, shuffle=False, pin_memory=False, num_workers=args.num_workers,batch_size=args.batch_size,collate_fn=_collate_fn)

    test_dataset = Data(test_lst,test_mode=True)
    test_loader = DataLoader(test_dataset, shuffle=False, pin_memory=False, num_workers=args.num_workers,batch_size=1,collate_fn=_collate_fn)

    start_time = time.time()

    test_model = globals()[network]()
    for p in test_model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)

    checkpoint_callback = ModelCheckpoint(
        monitor='avg_rmse',
        dirpath=out_path,
        filename=network + '-{epoch:02d}-{avg_rmse:.3f}',
        save_top_k=3,
        mode='min',
        save_last=True
    )
    
    if debug_mode:
        if not test_mode:
            trainer = pl.Trainer(max_epochs=epochs,
                                 accumulate_grad_batches=batch_size,
                                 default_root_dir=out_path,
                                 accelerator='ddp',
                                 callbacks=[checkpoint_callback],
                                 )
    else:
        logger = TensorBoardLogger(out_path, name="log")
        trainer = pl.Trainer(gpus=num_gpus,max_epochs=epochs,
                             accumulate_grad_batches=batch_size,
                             default_root_dir=out_path,
                             accelerator='ddp',
                             logger=logger,
                             callbacks=[checkpoint_callback],
                             num_sanity_val_steps=0,
                             #resume_from_checkpoint=os.path.join(out_path, 'last.ckpt')
                             )

    if test_mode:
        model = test_model.load_from_checkpoint("output/model1/SE3Refine-epoch=13-avg_rmse=2.783.ckpt")
        model = model.to(device)
        model.eval()
        with torch.no_grad():
            for batch in test_loader:
                nodes, pair, bond, init_pos, init_xyz, init_atom, init_CA, label_xyz, seq_l, pdbs, res_atom = batch
                init_CA = init_CA.to(device)
                bsz = 1
                L = seq_l[0]
                
                pdb = os.path.basename(pdbs[0])
                pdb = re.sub("\.pdb","",pdb)
                print(pdb)

                init_xyz = init_xyz.reshape(L, 1, 3)
                #init_xyz = torch.cat([init_xyz, init_xyz*torch.tensor([1,1,-1], dtype=init_xyz.dtype, device=init_xyz.device)])
                init_xyz = init_xyz.reshape(bsz,L,1,3)
                init_xyz = init_xyz.to(device)
                
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
                idx = idx.to(device)
                
                nodes = model.norm_node1(nodes.unsqueeze(1))
                nodes = nodes.reshape(bsz,L,58)
                nodes = nodes.to(device)
                nodes = model.norm_node2(model.embed_node(nodes))
                
                pair = pair.permute(0,3,1,2)
                pair = model.norm_edge1(pair)
                pair = pair.permute(0,2,3,1)
                pair = pair.to(device)
                pair = model.norm_edge2(model.embed_e1(pair))
                
                #neighbor = get_bonded_neigh(idx)
                #neighbor = neighbor.to(device)
                rbf_feat = rbf(torch.cdist(init_xyz[:,:,0,:], init_xyz[:,:,0,:]))
                rbf_feat = rbf_feat.to(device)

                bond = bond.reshape(1,L,L,1)
                bond = bond.to(device)
                #pair = torch.cat((pair, rbf_feat), dim=-1)
                pair = torch.cat((pair, rbf_feat, bond), dim=-1)
                pair = model.norm_edge3(model.embed_e2(pair)) 
                
                # define graph
                #xyz:[2, 138, 3, 3], pair:[2, 138, 138, 32], idx:[2, 138], top_k:64
                G = make_graph(init_xyz, pair, idx, top_k=128)
                l1_feats = init_pos # l1 features = displacement vector to CA
                l1_feats = l1_feats.reshape(bsz*L,-1, 3)
                l1_feats = l1_feats.to(device)
                
                # apply SE(3) Transformer & update coordinates
                #node.reshape(B*L, -1, 1):[276, 32, 1], l1_feats:[276, 3, 3]
                shift = model.se3(G, nodes.reshape(bsz*L, -1, 1), l1_feats) # 0: [276, 8, 1] 1: [276, 3, 3]
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

                os.system("mkdir -p output/pre")
                xyz2pdb(pdb, xyz_new.detach().cpu().numpy(), res_atom, "output/pre")
    else:
        time1 = time.time()
        trainer.fit(test_model, train_loader, val_loader)
        time2 = time.time()
        print('{} epochs takes {} seconds using {} GPUs.'.format(epochs, time2 - time1, num_gpus))
