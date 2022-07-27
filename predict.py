import os,re,glob
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

def _collate_fn(batch):
    nodes = []
    pairs = []
    label_init = []
    label_coord = []
    seq_l = []
    pdbs = []
    for iter,item in enumerate(batch):
        nodes.append(item[0])
        pairs.append(item[1])
        label_init.append(item[2])
        label_coord.append(item[3])
        pdbs.append(item[4])
        l = item[2].shape[0]
        seq_l.append(l)
    bsz = len(label_init)
    nodes = [torch.from_numpy(item).float() for item in nodes]
    nodes = torch.cat(nodes)
    pairs = [torch.from_numpy(item).float() for item in pairs]
    pairs = torch.cat(pairs)
    label_init = [torch.from_numpy(item).float() for item in label_init]
    label_init = torch.cat(label_init)
    label_coord = [torch.from_numpy(item).float() for item in label_coord]
    label_coord = torch.cat(label_coord)
    #batched_graph = dgl.batch(graphs)
    return torch.tensor(nodes),torch.tensor(pairs),torch.tensor(label_init),torch.tensor(label_coord),seq_l, pdbs

def dir_path(string):
    if os.path.isdir(string):
        return os.path.abspath(string)
    else:
        raise NotADirectoryError(string)

def datalist(pdb_dir,true_dir,train_dir,lst,tm_thre=0.0,test_mode=None):
    train_lst = []
    for line in open(train_dir+"/"+lst):
        line = line.rstrip().split()
        tar,l,tm = line[0],line[1],float(line[2])
        if tm >= tm_thre:
            continue
        if test_mode == "casp14r":
            train_lst.append(["casp14_refine/init_model/"+tar+".pdb","casp14_refine/init_model/"+tar+".npy","casp14_refine/init_model/"+tar+".npy",tm])
        elif test_mode == "casp14t":
            train_lst.append(["init_model_casp14/"+tar+".pdb","init_model_casp14/"+tar+".npy","init_model_casp14/"+tar+".npy",tm])
        else:
            train_lst.append([pdb_dir+"/all/"+tar+".pdb",pdb_dir+"/all/"+tar+".npy",true_dir+"/xyz/"+tar+".npy",tm])
    return train_lst

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Refinement')
    ap.add_argument('--data', type=dir_path, required=True, default='training list dir')
    ap.add_argument('--network', type=str, required=False, default='SE3Refine')
    ap.add_argument('--out_path', type=str, required=False,
                    default='output')
    ap.add_argument('--num_gpus', type=int, required=False, default=1)
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
    pdb_dir = "/storage/htc/bdm/tianqi/data/init_model"
    true_dir = "/storage/htc/bdm/tianqi/data/true_model"
    train_lst = datalist(pdb_dir, true_dir, train_dir,"all_train500.lst",tm_thre=0.9)
    val_lst = datalist(pdb_dir, true_dir, train_dir,"valid.lst",tm_thre=0.8)
    test_lst = datalist(pdb_dir, true_dir, train_dir,"test.lst",tm_thre=1.1,test_mode=None)
    
    train_dataset = Data(train_lst)
    train_loader = DataLoader(train_dataset, shuffle=True, pin_memory=False, num_workers=args.num_workers,batch_size=args.batch_size,collate_fn=_collate_fn)

    val_dataset = Data(val_lst)
    val_loader = DataLoader(val_dataset, shuffle=False, pin_memory=False, num_workers=args.num_workers,batch_size=args.batch_size,collate_fn=_collate_fn)

    test_dataset = Data(test_lst)
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
        loss_init = []
        loss_refine = []
        with torch.no_grad():
            for batch in test_loader:
                node, pair, y_init, y_coord, seq_l, pdbs = batch
                pdb = os.path.basename(pdbs[0])
                pdb = re.sub("\.pdb","",pdb)

                print(pdb)

                node_iter = node.to(device)
                pair_iter = pair.to(device)
                y_init = y_init.to(device)
                y_coord = y_coord.to(device)
                y_init = y_init.squeeze(0).to(device)
                y_coord = y_coord.squeeze(0).to(device)
                L, _ = y_coord.size()
                bsz = 2

                xyz = y_init.reshape(L, 3, 3)
                xyz = torch.cat([xyz, xyz*torch.tensor([1,1,-1], dtype=xyz.dtype, device=xyz.device)])

                offset_lst = []
                count = 1
                model_lst = ["model1/SE3Refine-epoch=26-avg_rmse=5.271.ckpt","model3/SE3Refine-epoch=29-avg_rmse=4.907.ckpt","model4/SE3Refine-epoch=23-avg_rmse=4.138.ckpt","last.ckpt.bkp","SE3Refine-epoch=20-avg_rmse=6.436.ckpt"]
                for model_file in model_lst:
                    model = test_model.load_from_checkpoint(src_dir+"/output/"+model_file)
                    model = model.to(device)
                    model.eval()

                    node = torch.cat([node_iter, node_iter])
                    node = node.to(device)
                    pair = torch.cat([pair_iter, pair_iter])
                    pair = pair.to(device)
                    idx = torch.arange(L).long().view(1, L)
                    idx = idx.to(device)
                    idx = torch.cat([idx, idx])

                    pair = pair.reshape(bsz,L,L,6)
                    idx = idx.reshape(bsz,L)
                    xyz = xyz.reshape(bsz,L,3,3)
                    xyz = xyz.to(device)

                    node = model.norm_node1(node.unsqueeze(1))
                    node = node.reshape(bsz,L,28)
                    node = model.norm_node2(model.embed_node(node))
                    
                    pair = pair.permute(0,3,1,2)
                    pair = model.norm_edge1(pair)
                    pair = pair.permute(0,2,3,1)
                    pair = model.norm_edge2(model.embed_e1(pair))
                    
                    neighbor = get_bonded_neigh(idx)
                    neighbor = neighbor.to(device)
                    rbf_feat = rbf(torch.cdist(xyz[:,:,1,:], xyz[:,:,1,:]))
                    rbf_feat = rbf_feat.to(device)

                    pair = pair.to(device)
                    pair = torch.cat((pair, rbf_feat, neighbor), dim=-1)
                    pair = model.norm_edge3(model.embed_e2(pair))
                    
                    # define graph
                    for i in range(1):
                        if i >0:
                            xyz = xyz_new_tmp.reshape(bsz,L,3,3)
                        G = make_graph(xyz, pair, idx, top_k=18)
                        l1_feats = xyz - xyz[:,:,1,:].unsqueeze(2)
                        l1_feats = l1_feats.reshape(bsz*L, -1, 3)
                        
                        # SE(3) Transformer
                        shift = model.se3(G, node.reshape(bsz*L, -1, 1), l1_feats)
                        offset = shift['1'].reshape(bsz, L, -1, 3)
                        offset_lst.append(offset)

                        CA_new = xyz[:,:,1] + offset[:,:,1]
                        N_new = CA_new + offset[:,:,0]
                        C_new = CA_new + offset[:,:,2]
                        xyz_new = torch.stack([N_new, CA_new, C_new], dim=2)

                        xyz_new1 = xyz_new[0]
                        xyz_new1 = xyz_new1.reshape(L,9)
                        convert_weight = torch.tensor([1,1,-1])
                        convert_weight = convert_weight.to(device)
                        xyz_new2 = xyz_new[1]*convert_weight
                        xyz_new2 = xyz_new2.reshape(L,9)
                        xyz_new = (xyz_new1+xyz_new2)/2

                        os.system("mkdir -p output/pre")
                        np.save("output/pre/"+pdb,xyz_new.cpu().numpy())
                        os.system("python "+src_dir+"/xyz2pdb.py "+pdb+".fasta "+src_dir+"/output/pre/"+pdb+".npy "+src_dir+"/output/pre/")
                        os.system("mv output/pre/"+pdb+".pdb output/pre/model/"+pdb+str(count)+".pdb")
                        count = count+1
                        os.system("rm output/pre/"+pdb+".npy")
                offset = torch.mean(torch.stack(offset_lst), dim=0)
                CA_new = xyz[:,:,1] + offset[:,:,1]
                N_new = CA_new + offset[:,:,0]
                C_new = CA_new + offset[:,:,2]
                xyz_new = torch.stack([N_new, CA_new, C_new], dim=2)
                
                xyz_new1 = xyz_new[0]
                xyz_new1 = xyz_new1.reshape(L,9)
                convert_weight = torch.tensor([1,1,-1])
                convert_weight = convert_weight.to(device)
                xyz_new2 = xyz_new[1]*convert_weight
                xyz_new2 = xyz_new2.reshape(L,9)
                xyz_new = (xyz_new1+xyz_new2)/2

                loss = torch.sqrt(model.lossfn(xyz_new, y_coord)/seq_l[0])
                loss_refine.append(loss)
                loss = torch.sqrt(model.lossfn(y_init, y_coord)/seq_l[0])
                loss_init.append(loss)
                
                os.system("mkdir -p output/pre")
                np.save("output/pre/"+pdb,xyz_new.cpu().numpy())
                os.system("python "+src_dir+"/xyz2pdb.py "+pdb+".fasta "+src_dir+"/output/pre/"+pdb+".npy "+src_dir+"/output/pre/")
                os.system("rm output/pre/"+pdb+".npy")
            loss_avg_init = torch.mean(torch.stack(loss_init))
            loss_avg = torch.mean(torch.stack(loss_refine))
            print("init rmsd:",loss_avg_init)
            print("refine rmsd:",loss_avg)
    else:
        time1 = time.time()
        trainer.fit(test_model, train_loader, val_loader)
        time2 = time.time()
        print('{} epochs takes {} seconds using {} GPUs.'.format(epochs, time2 - time1, num_gpus))
