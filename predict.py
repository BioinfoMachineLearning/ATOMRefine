import os,re
import time
import argparse
import pytorch_lightning as pl

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils import data
from torch.utils.data import DataLoader

from Data import Data
from network2 import SE3Refine
from network2 import get_bonded_neigh,rbf,make_graph

import dgl
import numpy as np

from amber import protein
from amber import relax

import warnings
warnings.filterwarnings("ignore")

import sys

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def xyz2pdb(tar, xyz, res_atom, outdir):
    f = open(outdir+"/"+tar+".pdb","w")
    i = 1
    j = 1
    for res in res_atom:
        res_name = res[0:3]
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
    res_atom = []
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
        res_atom = item[9]
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

    return nodes, pairs, bonds, init_xyz, init_pos, init_atom, init_CA, label_xyz, seq_l, pdbs, res_atom

def amber_relax(input_file, out_file):
    #### amber relaxation from Alphafold2 source code https://github.com/deepmind/alphafold
    test_config = {'max_iterations': 1,
               'tolerance': 2.39,
               'stiffness': 10.0,
               'exclude_residues': [],
               'max_outer_iterations': 1,
               'use_gpu': False}

    amber_relax = relax.AmberRelaxation(**test_config)

    with open(input_file) as f:
      test_prot = protein.from_pdb_string(f.read())
    try:
      pdb_min, debug_info, num_violations = amber_relax.process(prot=test_prot)
      f = open(out_file,"w")
      f.write(pdb_min)
      f.close()
    except Exception as e:
      print(input_file)

def dir_path(string):
    if os.path.isdir(string):
        return os.path.abspath(string)
    else:
        raise NotADirectoryError(string)

def datalist(pdbfile, targetid, seqlen, outdir, test_mode=False):
    train_lst = []
    tar,L, = targetid, int(seqlen)

    if test_mode:
        if L >= 1500:
            window = 900
            shift = 850
            grids = np.arange(0, L-window+shift, shift)
            ngrids = grids.shape[0]
            for i in range(ngrids):
                start_1 = grids[i]
                end_1 = min(grids[i]+window, L)
                print(start_1, end_1)
                os.system("python "+src_dir+"/pdb_selres.py -"+str(start_1+1)+":"+str(end_1)+" "+pdbfile+" > "+outdir+"/"+tar+"_"+str(i)+".pdb")
                os.system("python "+src_dir+"/pdb_reres.py -1 "+outdir+"/"+tar+"_"+str(i)+".pdb > "+outdir+"/"+tar+"_"+str(i)+".tmp")
                os.system("python "+src_dir+"/pdb_reatom.py -1 "+outdir+"/"+tar+"_"+str(i)+".tmp > "+outdir+"/"+tar+"_"+str(i)+".pdb")
                os.system("rm "+outdir+"/"+tar+"_"+str(i)+".tmp")
                train_lst.append([outdir+"/"+tar+"_"+str(i)+".pdb",outdir+"/"+tar+"_"+str(i)+".pdb"])
        else:
            train_lst.append([pdbfile,pdbfile])
    return train_lst

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Refinement')
    ap.add_argument('--network', type=str, required=False, default='SE3Refine')
    ap.add_argument('--init', type=str, required=False, default='initial pdbfile')
    ap.add_argument('--id', type=str, required=False, default='pdb id')
    ap.add_argument('--seql', type=int, required=False, default='sequence length')
    ap.add_argument('--out_path', type=str, required=False, default='output')
    ap.add_argument('--num_gpus', type=int, required=False, default=1)
    ap.add_argument('--num_workers', type=int, required=False, default=4)
    ap.add_argument('--test_seed', type=int, required=False, default=None)
    ap.add_argument('--test', required=False, action='store_true')
    ap.add_argument('--amber', required=False, action='store_true')

    args = ap.parse_args()
    network = args.network
    pdbfile = os.path.abspath(args.init)
    targetid = args.id
    seqlen = int(args.seql)
    out_path = os.path.abspath(args.out_path)
    num_gpus = args.num_gpus
    num_workers = args.num_workers
    test_mode = args.test
    amber_mode = args.amber
    test_seed = args.test_seed

    pl.utilities.seed.seed_everything(seed=test_seed)

    src_dir = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))

    test_lst = datalist(pdbfile, targetid, seqlen, out_path, test_mode=test_mode)
    test_dataset = Data(test_lst,test_mode=True)
    test_loader = DataLoader(test_dataset, shuffle=False, pin_memory=False, num_workers=args.num_workers,batch_size=1,collate_fn=_collate_fn)

    start_time = time.time()

    test_model = globals()[network]()
    for p in test_model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)

    if test_mode:
        model_dir = src_dir+"/model/"
        model_dict = {1:"SE3Refine-epoch=03-avg_rmse=2.946.ckpt", 2:"SE3Refine-epoch=04-avg_rmse=2.631.ckpt", 3:"SE3Refine-epoch=11-avg_rmse=2.746.ckpt", 4:"SE3Refine-epoch=22-avg_rmse=2.836.ckpt", 5:"SE3Refine-epoch=07-avg_rmse=2.744.ckpt"}
        print("Start prediction ...Predict each model may take 4 or 5 mins...In total, 5 models will be generated....")
        for lst,modelfile in model_dict.items():
            model = test_model.load_from_checkpoint(model_dir+"/model"+str(lst)+"/"+modelfile)
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
                    if os.path.exists(out_path +"/"+pdb+"_"+str(lst)+".pdb"):
                        continue
                    print(pdb+"....model"+str(lst))

                    init_xyz = init_xyz.reshape(L, 1, 3)
                    init_xyz = init_xyz.reshape(bsz,L,1,3)
                    init_xyz = init_xyz.to(device)
                    
                    init_pos = init_pos.reshape(L, 1, 3)
                    init_pos = init_pos.reshape(bsz,L,1,3)
                    
                    pair = pair.reshape(bsz,L,L,12)
                    
                    idx = torch.arange(L).long().view(1, L)
                    idx = idx.to(device)
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
                    
                    rbf_feat = rbf(torch.cdist(init_xyz[:,:,0,:], init_xyz[:,:,0,:]))
                    rbf_feat = rbf_feat.to(device)

                    bond = bond.reshape(1,L,L,1)
                    bond = bond.to(device)
                    pair = torch.cat((pair, rbf_feat, bond), dim=-1)
                    pair = model.norm_edge3(model.embed_e2(pair)) 
                    
                    # define graph
                    G = make_graph(init_xyz, pair, idx, top_k=128)
                    l1_feats = init_pos # l1 features = displacement vector to CA
                    l1_feats = l1_feats.reshape(bsz*L,-1, 3)
                    l1_feats = l1_feats.to(device)
                    
                    # SE(3) Transformer
                    shift = model.se3(G, nodes.reshape(bsz*L, -1, 1), l1_feats)
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
                    if not os.path.exists(out_path +"/tmp"):
                        os.makedirs(out_path +"/tmp")
                    xyz2pdb(pdb+"_"+str(lst), xyz_new.detach().cpu().numpy(), res_atom, out_path +"/tmp")

        tar = targetid
        L = seqlen
        if L >= 1500:
            window = 900
            shift = 850
            grids = np.arange(0, L-window+shift, shift)
            ngrids = grids.shape[0]
            for i in range(ngrids):
                start_1 = grids[i]
                end_1 = min(grids[i]+window, L)
                print(start_1, end_1)
                for lst in model_dict.keys():
                    os.system("python "+src_dir+"/pdb_reres.py -"+str(start_1+1)+" "+out_path+"/tmp/"+tar+"_"+str(i)+"_"+str(lst)+".pdb > "+out_path+"/tmp/"+tar+"_"+str(i)+"_"+str(lst)+".tmp")
                    os.system("python "+src_dir+"/pdb_reatom.py -1 "+out_path+"/tmp/"+tar+"_"+str(i)+"_"+str(lst)+".tmp > "+out_path+"/tmp/"+tar+"_"+str(i)+"_"+str(lst)+".pdb")
                    os.system("rm "+out_path+"/tmp/"+tar+"_"+str(i)+"_"+str(lst)+".tmp")
            for lst in model_dict.keys():
                os.system("cp "+out_path+"/tmp/"+tar+"_0_"+str(lst)+".pdb "+out_path+"/tmp/"+tar+"_"+str(lst)+".tmp")
                for i in range(ngrids - 1):
                    os.system("python "+src_dir+"/pdb_intersect.py "+out_path+"/tmp/"+tar+"_"+str(lst)+".tmp "+out_path+"/tmp/"+tar+"_"+str(i + 1)+"_"+str(lst)+".pdb > "+out_path+"/tmp/"+tar+"_"+str(lst)+".pdb")
                    os.system("cp "+out_path+"/tmp/"+tar+"_"+str(lst)+".pdb "+out_path+"/tmp/"+tar+"_"+str(lst)+".tmp")
                os.system("cp "+out_path+"/tmp/"+tar+"_"+str(lst)+".tmp "+out_path+"/"+tar+"_"+str(lst)+".pdb")
        else:
            for lst in model_dict.keys():
                os.system("cp "+out_path+"/tmp/"+tar+"_"+str(lst)+".pdb "+out_path+"/"+tar+"_"+str(lst)+".pdb")

        if amber_mode:
            for lst in model_dict.keys():
                amber_relax(out_path+"/"+tar+"_"+str(lst)+".pdb", out_path+"/tmp/"+tar+"_"+str(lst)+".pdb")
                os.system("cp "+out_path+"/tmp/"+tar+"_"+str(lst)+".pdb "+out_path+"/"+tar+"_"+str(lst)+".pdb")
        os.system("rm -r "+out_path+"/tmp")
        print("Prediction finished......")
