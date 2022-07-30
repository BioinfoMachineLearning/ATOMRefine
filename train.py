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
            train_lst.append(["data/init_model_casp14/"+tar+".pdb","data/init_model_casp14/"+tar+".npy","data/init_model_casp14/"+tar+".npy",tm])
        else:
            train_lst.append([pdb_dir+"/"+tar+".pdb",true_dir+"/"+tar+".pdb",tm])
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
    pdb_dir = "data/AF2_model"
    true_dir = "data/true_model"
    train_lst = datalist(pdb_dir, true_dir, train_dir,"train"+str(lst)+".lst",tm_thre=1.1)
    val_lst = datalist(pdb_dir, true_dir, train_dir,"valid"+str(lst)+".lst",tm_thre=1.1)
    test_lst = datalist(pdb_dir, true_dir, train_dir,"test.lst",tm_thre=1.1,test_mode=False)
    
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
        trainer = pl.Trainer(max_epochs=epochs,
                             accumulate_grad_batches=batch_size,
                             default_root_dir=out_path,
                             accelerator='ddp',
                             logger=logger,
                             callbacks=[checkpoint_callback],
                             num_sanity_val_steps=0,
                             #resume_from_checkpoint=os.path.join(out_path, 'last.ckpt')
                             )

    time1 = time.time()
    trainer.fit(test_model, train_loader, val_loader)
    time2 = time.time()
    print('{} epochs takes {} seconds using {} GPUs.'.format(epochs, time2 - time1, num_gpus))
