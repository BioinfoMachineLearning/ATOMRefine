#!/usr/bin/env python3
# encoding: utf-8
# Original script from Jing, X. GNNRefine: fast and effective protein model refinement by deep graph neural networks (Code Ocean, 2021); https://doi.org/10.24433/CO.8813669.v1
# With local frame construct and relative position/orentation function added 

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

def get_SepEnc(seq):
    seq_len = len(seq)
    sep = np.abs(np.linspace(0,seq_len,num=seq_len,endpoint=False)[:, None] - np.linspace(0,seq_len,num=seq_len,endpoint=False)[None, :])
    for i,step in enumerate(np.linspace(5, 20, num=3, endpoint=False)):
        sep[np.where((sep>step) & (sep<=step+5))] = 6+i
    sep[np.where(sep>step+5)] = 6+i+1
    sep = sep-1
    sep[np.where(sep<0)] = 0
    sep_enc = get_one_hot(sep.astype(np.int), 9)    

    return sep_enc


# {G,H,I: H}, {S,T,C: C}, {B,E: E}
SS3_TYPES = {'H':0, 'B':2, 'E':2, 'G':0, 'I':0, 'T':1, 'S':1, '-':1}
def get_DSSP_label(decoy_file, res_range, invalid_value=-1):
    '''
    Extract the SS, RSA, Dihedral from pdb file using DSSP
    Agrs:
        decoy_file (string): the path of pdb structure file.
        res_range (int, int): rasidue id range, e.g. [1,100].
    '''
    res_num = res_range[1]-res_range[0]+1

    structure = pdb_parser.get_structure("tmp_stru", decoy_file)
    model = structure.get_list()[0]
    dssp = DSSP(model, decoy_file, dssp='/storage/htc/bdm/tools/dssp/dssp')
    SS3s, RSAs, Dihedrals = np.zeros((res_num, 3)), np.zeros((res_num, 1)), np.zeros((res_num, 2))
    for _key in dssp.keys():
        res_index = _key[1][1]
        if res_index<res_range[0] or res_index>res_range[1]: continue
        SS, RSA = dssp[_key][2], dssp[_key][3]        
        SS3s[res_index-res_range[0], SS3_TYPES[SS]] = 1
        if not RSA=='NA': RSAs[res_index-res_range[0]] = [RSA]
        phi, psi = dssp[_key][4], dssp[_key][5]
        Dihedrals[res_index-res_range[0]] = [phi, psi]

    # convert degree to radian
    Dihedrals[Dihedrals==360.0] = 0
    Dihedrals = Dihedrals/180*np.pi

    feature_dict = {'SS3': SS3s, 'RSA': RSAs, 'Dihedral': Dihedrals, }
    return feature_dict
    
def calc_dist_map(data, nan_fill=-1):
    """
    Calc the dist for a map with two points.
        data: shape: [N, N, 2, 3].
    """
    dist_map = np.linalg.norm((data[:,:,0,:]-data[:,:,1,:]), axis=-1)
    return np.nan_to_num(dist_map, nan=nan_fill)


def calc_angle_map(data, nan_fill=-4):
    """
    Calc the ange for a map with three points.
        data: shape: [N, N, 3, 3].
    """
    ba, bc = (data[:,:,0,:]-data[:,:,1,:]), (data[:,:,2,:]-data[:,:,1,:])
    angle_radian = np.arccos(np.einsum('ijk,ijk->ij', ba, bc) / (np.linalg.norm(ba, axis=-1)*np.linalg.norm(bc, axis=-1)))
    # angle_degree = np.degrees(angle_radian)
    return np.nan_to_num(angle_radian, nan=nan_fill)

def calc_dihedral_map(data, nan_fill=-4):
    """
    Calc the dihedral ange for a map with four points.
        data: shape: [N, N, 4, 3].
    """
    b01 = -1.0*(data[:,:,1,:] - data[:,:,0,:])
    b12 = data[:,:,2,:] - data[:,:,1,:]
    b23 = data[:,:,3,:] - data[:,:,2,:]

    b12 = b12/np.linalg.norm(b12, axis=-1)[:, :, None]

    v = b01 - np.einsum('ijk,ijk->ij', b01, b12)[:, :, None]*b12
    w = b23 - np.einsum('ijk,ijk->ij', b23, b12)[:, :, None]*b12

    x = np.einsum('ijk,ijk->ij', v, w)
    y = np.einsum('ijk,ijk->ij', np.cross(b12, v, axis=-1), w)

    return np.nan_to_num(np.arctan2(y, x), nan=nan_fill)

DIST_FEATURE_SCALE = {'Ca1-Ca2': 1.0, 'Cb1-Cb2': 1.0, 'N1-O2': 1.0, }
def calc_geometry_maps(structure_file, res_range=None,
                        geometry_types=['Ca1-Ca2', 'Cb1-Cb2', 'N1-O2', 'Ca1-Cb1-Cb2', 'N1-Ca1-Cb1-Cb2', 'Ca1-Cb1-Cb2-Ca2'],
                        nan_fill={2: -1, 3: -1, 4: -4,}):
    '''
    Agrs:
        structure_file (string): the path of pdb structure file.
        res_range [int, int]: the start and end residue index, e.g. [1, 100].
        geometry_types (list): the target atom types of geometry map.
            distance map: 'Ca1-Ca2', 'Cb1-Cb2', 'N1-O2'
            orientation map: 'Ca1-Cb1-Cb2', 'N1-Ca1-Cb1-Cb2', 'Ca1-Cb1-Cb2-Ca2'
        nan_fill (float): the default value of invalid value.
    '''
    # filter out unsupport types
    geometry_types = set(geometry_types) & set(['Ca1-Ca2','Cb1-Cb2','N1-O2','Ca1-Cb1-Cb2','N1-Ca1-Cb1-Cb2','Ca1-Cb1-Cb2-Ca2'])
    # load pdb file
    structure = pdb_parser.get_structure("tmp_stru", structure_file)
    residues = [_ for _ in structure.get_residues()]
    if not res_range: res_range = [1, len(residues)]

    # the residue num
    res_num = res_range[1]-res_range[0]+1
    
    # target atom types to extract coordinates
    atom_types = set(['CA'])
    for otp in geometry_types:
        for _atom in otp.split('-'):
            atom_types.add(_atom[:-1].upper())

    # generate empty coordinates
    coordinates = {}
    for atom_type in atom_types: coordinates[atom_type] = np.zeros((res_num, 3))

    # extract coordinates from pdb
    res_tags = np.zeros((res_num)).astype(np.int8)
    CB_tags = np.zeros((res_num)).astype(np.int8)
    for residue in residues:
        if residue.id[1]<res_range[0] or residue.id[1]>res_range[1]: continue
        res_index = residue.id[1]-res_range[0]
        res_tags[res_index] = 1
        _CB_tag = 0
        for atom in residue:
            if atom.name in atom_types:
                coordinates[atom.name][res_index] = atom.coord
                if atom.name=="CB": _CB_tag=1
        if _CB_tag==0: coordinates['CB'][res_index] = coordinates['CA'][res_index]
        CB_tags[res_index] = _CB_tag

    geometry_dict = dict()
    # ['Ca1-Ca2', 'Cb1-Cb2', 'N1-O2', 'Ca1-Cb1-Cb2', 'N1-Ca1-Cb1-Cb2', 'Ca1-Cb1-Cb2-Ca2']
    for gmt_type in geometry_types:        
        points_map = None
        id_atom = {'1': [], '2':[]}
        for _atom in gmt_type.split('-'):
            _atom_type = _atom[:-1].upper()
            id_atom[_atom[-1]].append(_atom_type)
            _data = np.repeat(coordinates[_atom_type], res_num, axis=0).reshape((res_num, res_num, 3))
            if _atom[-1]=='2':
                _data = np.transpose(_data, (1, 0, 2))
            if points_map is None:
                points_map = _data[:,:,None,:]
            else:
                points_map = np.concatenate((points_map, _data[:,:,None,:]), axis=2)

        if len(gmt_type.split('-')) == 2: # dist
            data_map = calc_dist_map(points_map, nan_fill[2])
        elif len(gmt_type.split('-')) == 3: # angle
            data_map = calc_angle_map(points_map, nan_fill[3])
        elif len(gmt_type.split('-')) == 4: # dihedral
            data_map = calc_dihedral_map(points_map, nan_fill[4])
        
        # mask the no residue and CB sites
        idx = np.where(res_tags==0)[0].tolist() # no residue
        if len(idx)>0:
            data_map[np.array(idx), :] = nan_fill[len(gmt_type.split('-'))]
            data_map[:, np.array(idx)] = nan_fill[len(gmt_type.split('-'))]
        if len(gmt_type.split('-'))>2:
            # row
            if ("CA" in id_atom['1']) and ("CB" in id_atom['1']):
                idx_row = np.where(CB_tags==0)[0].tolist()
                if len(idx_row)>0: data_map[np.array(idx_row), :] = nan_fill[len(gmt_type.split('-'))]
            # col
            if ("CA" in id_atom['2']) and ("CB" in id_atom['2']):
                idx_col = np.where(CB_tags==0)[0].tolist()
                if len(idx_col)>0: data_map[:, np.array(idx_col)] = nan_fill[len(gmt_type.split('-'))]
        # save
        scale = DIST_FEATURE_SCALE[gmt_type] if DIST_FEATURE_SCALE.__contains__(gmt_type) else 1.0
        geometry_dict[gmt_type] = (data_map*scale).astype(np.float16)[:,:,None]

    return geometry_dict

heavy_atoms = pickle.load(open(__MYPATH__+"/heavy_atoms.pkl", "rb"))
#atom_dict = {'C':0, 'N':1, 'O':2, 'S':3}
atom_dict = {"N":0, "CA":1, "C":2, "O":3, "CB":4, "OG":5, "CG":6, "CD1":7, "CD2":8, "CE1":9, "CE2":10, "CZ":11, "OD1":12, "ND2":13, "CG1":14, "CG2":15, "CD":16, "CE":17, "NZ":18, "OD2":19, "OE1":20, "NE2":21, "OE2":22, "OH":23, "NE":24, "NH1":25, "NH2":26, "OG1":27, "SD":28, "ND1":29, "SG":30, "NE1":31, "CE3":32, "CZ2":33, "CZ3":34, "CH2":35, "OXT":36}
def get_atom_emb(pdb_file, true_pdb, res_range, model_id=0, chain_id=0):
    '''
    Generate the atom embedding from coordinates and type for each residue.
    Agrs:
        decoy_file (string): the path of pdb structure file.
        res_range (int, int): rasidue id range, start from 1, e.g. [1,100].
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
    for residue,residue_true in zip(residue_list,residue_list_true):
        if residue_true.id[1]<res_range[0] or residue_true.id[1]>res_range[1]: continue
        res_index = residue.id[1]-res_range[0]
        atom_pos, onehot = [], []
        _resname = residue_true.get_resname() if residue_true.get_resname() in heavy_atoms else 'GLY'
        for _atom in heavy_atoms[_resname]['atoms']:
            if residue_true.has_id(_atom):
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
        if not np.isscalar(_item): # not -1, no data
            atom_nums[i] = _item.shape[0]

    #return embedding
    return atom_embs,atom_xyz,atom_lst,CA_lst#s,res_atom

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