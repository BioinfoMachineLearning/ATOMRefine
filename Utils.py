#!/usr/bin/env python3
# encoding: utf-8
# ATOM Embedding function and local frame construct function

import os, pickle
import numpy as np
import Bio.PDB
from Bio import SeqIO
from Bio.PDB.DSSP import DSSP
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")

__MYPATH__ = os.path.split(os.path.realpath(__file__))[0]
pdb_parser = Bio.PDB.PDBParser(QUIET = True)

def get_seqs_from_pdb(pdb_file):
    for record in SeqIO.parse(pdb_file, "pdb-atom"):
        return str(record.seq).upper()

RESIDUE_TYPES = ['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y','X']
def get_seq_onehot(seq):
    seq_onehot = np.zeros((len(seq), len(RESIDUE_TYPES)))
    for i, res in enumerate(seq.upper()):
        if res not in RESIDUE_TYPES: res = "X"
        seq_onehot[i, RESIDUE_TYPES.index(res)] = 1
    return seq_onehot

def get_rPos(seq):
    seq_len= len(seq)
    r_pos = np.linspace(0, 1, num=seq_len).reshape(seq_len, -1)
    return r_pos

def get_one_hot(targets, nb_classes):
    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape)+[nb_classes])


heavy_atoms = pickle.load(open(__MYPATH__+"/heavy_atoms.pkl", "rb"))
#atom_dict = {'C':0, 'N':1, 'O':2, 'S':3}
atom_dict = {"N":0, "CA":1, "C":2, "O":3, "CB":4, "OG":5, "CG":6, "CD1":7, "CD2":8, "CE1":9, "CE2":10, "CZ":11, "OD1":12, "ND2":13, "CG1":14, "CG2":15, "CD":16, "CE":17, "NZ":18, "OD2":19, "OE1":20, "NE2":21, "OE2":22, "OH":23, "NE":24, "NH1":25, "NH2":26, "OG1":27, "SD":28, "ND1":29, "SG":30, "NE1":31, "CE3":32, "CZ2":33, "CZ3":34, "CH2":35, "OXT":36}
def get_atom_emb(pdb_file, true_pdb, res_range, model_id=0, chain_id=0):
    '''
    Generate the atom embedding from coordinates and type for each residue.
    '''
    res_num = res_range[1]-res_range[0]+1

    structure_true = pdb_parser.get_structure('native', true_pdb)
    model_true = structure_true.get_list()[model_id]
    chain_true = model_true.get_list()[chain_id]
    residue_list_true = chain_true.get_list()

    structure = pdb_parser.get_structure('tmp', pdb_file)
    model = structure.get_list()[model_id]
    chain = model.get_list()[chain_id]
    residue_list = chain.get_list()
    atom_embs = [-1 for _ in range(res_num)]
    atom_xyz = [-1 for _ in range(res_num)]
    
    atom_lst = []
    CA_lst = []
    res_atom = defaultdict(list)
    for residue,residue_true in zip(residue_list,residue_list_true):
        if residue_true.id[1]<res_range[0] or residue_true.id[1]>res_range[1]: continue
        res_index = residue.id[1]-res_range[0]
        atom_pos, onehot = [], []
        _resname = residue_true.get_resname() if residue_true.get_resname() in heavy_atoms else 'GLY'
        for _atom in heavy_atoms[_resname]['atoms']:
            if residue_true.has_id(_atom):
                res_atom[_resname+str(residue.id[1]-res_range[0])].append(_atom)
                atom_pos.append(residue[_atom].coord)
                _onehot = np.zeros(len(atom_dict))
                _onehot[atom_dict[_atom]] = 1
                onehot.append(_onehot)
        atom_lst.append(len(atom_pos))
        CA_pose = residue['CA'].coord
        CA_lst.append(CA_pose)
        atom_emb = np.concatenate((np.array(onehot), np.array(atom_pos)-CA_pose[None,:]), axis=1)
        atom_embs[residue.id[1]-res_range[0]] = atom_emb.astype(np.float16)
        atom_xyz[residue.id[1]-res_range[0]] = np.array(atom_pos).astype(np.float16)

    atom_nums = np.zeros((res_num))
    for i, _item in enumerate(atom_embs):
        if not np.isscalar(_item):
            atom_nums[i] = _item.shape[0]

    return atom_embs,atom_xyz,atom_lst,CA_lst,res_atom

def set_lframe(structure_file, atom_xyz, atom_lst, res_range=None):
    '''
    Agrs:
        structure_file (string): the path of pdb structure file.
        res_range [int, int]: the start and end residue index, e.g. [1, 100].
    '''
    # load pdb file
    structure = pdb_parser.get_structure("tmp_stru", structure_file)
    residues = [_ for _ in structure.get_residues()]

    # the residue num
    res_num = res_range[1]-res_range[0]+1

    # set local frame for each residue in pdb
    pdict = dict()
    pdict['N'] = np.stack([np.array(residue['N'].coord) for residue in residues])
    pdict['Ca'] = np.stack([np.array(residue['CA'].coord) for residue in residues])
    pdict['C'] = np.stack([np.array(residue['C'].coord) for residue in residues])

    # recreate Cb given N,Ca,C
    ca = -0.58273431
    cb = 0.56802827
    cc = -0.54067466
    
    b = pdict['Ca'] - pdict['N']
    c = pdict['C'] - pdict['Ca']
    a = np.cross(b, c)
    pdict['Cb'] = ca * a + cb * b + cc * c

    # local frame
    z = pdict['Cb'] - pdict['Ca']
    z /= np.linalg.norm(z, axis=-1)[:,None]
    x = np.cross(pdict['Ca']-pdict['N'], z)
    x /= np.linalg.norm(x, axis=-1)[:,None]
    y = np.cross(z, x)
    y /= np.linalg.norm(y, axis=-1)[:,None]

    xyz = np.stack([x,y,z])

    pdict['lfr'] = np.transpose(xyz, [1,0,2])
    
    start, end, j = 0, 0, 0
    atom_idx = [-1 for _ in range(atom_xyz.shape[0])]
    for i in range(len(atom_lst)):
        start = end
        end += atom_lst[i]
        atom_idx[start:end] = [j]*atom_lst[i]
        j = j+1
    
    p = np.zeros((atom_xyz.shape[0], atom_xyz.shape[0],3))
    q = np.zeros((atom_xyz.shape[0], atom_xyz.shape[0],3))
    k = np.zeros((atom_xyz.shape[0], atom_xyz.shape[0],3))
    t = np.zeros((atom_xyz.shape[0], atom_xyz.shape[0],3))
    for i in range(atom_xyz.shape[0]):
        res_idx = atom_idx[i]
        for j in range(atom_xyz.shape[0]):
            p[i,j,:] = np.matmul(pdict['lfr'][res_idx],atom_xyz[j]-atom_xyz[i])
            q[i,j,:] = np.matmul(pdict['lfr'][atom_idx[i]],pdict['lfr'][atom_idx[j]][0])
            k[i,j,:] = np.matmul(pdict['lfr'][atom_idx[i]],pdict['lfr'][atom_idx[j]][1])
            t[i,j,:] = np.matmul(pdict['lfr'][atom_idx[i]],pdict['lfr'][atom_idx[j]][2])
    
    return p,q,k,t