import numpy as np
import math

import os,sys,re

d = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
     'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N', 
     'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W', 
     'ALA': 'A', 'VAL':'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M'}

d_inverse = {'C': 'CYS', 'D': 'ASP', 'S': 'SER', 'Q': 'GLN', 'K': 'LYS',
     'I': 'ILE', 'P': 'PRO', 'T': 'THR', 'F': 'PHE', 'N': 'ASN', 
     'G': 'GLY', 'H': 'HIS', 'L': 'LEU', 'R': 'ARG', 'W': 'TRP', 
     'A': 'ALA', 'V': 'VAL', 'E': 'GLU', 'Y': 'TYR', 'M': 'MET'}

def get_length(fasta):
    seq = ""
    for line in open(fasta,"r"):
        line = line.rstrip()
        if (line.startswith(">")) or (line in ['\n', '\r\n']):
            continue
        else:
            seq += line 
    return seq,len(seq)

def get_xyz(xyz_file):
    xyz = np.load(xyz_file)
    N = xyz[:,0:3].tolist()
    CA = xyz[:,3:6].tolist()
    C = xyz[:,6:9].tolist()
    return N,CA,C

def xyz2pdb(fasta_name, fasta, N_lst, CA_lst, C_lst, outdir):
    f = open(outdir+"/"+fasta_name+".pdb","w")
    seq,l = get_length(fasta)
    i = 1
    j = 1
    for aa in seq:
        ## N ccordinates
        atom_seq= j
        res_name = d_inverse[aa]
        res_seq= i
        x = N_lst[i-1][0]
        y = N_lst[i-1][1]
        z = N_lst[i-1][2]
        line="{:6s}{:5d} {:^4s} {:3s} {:1s}{:4d}    {:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}".format('ATOM',atom_seq,'N',res_name,'A',res_seq,x,y,z,0,0)
        f.write(line+"\n")
        j = j+1
        
        ## CA ccordinates
        atom_seq= j
        res_name = d_inverse[aa]
        res_seq= i
        x = CA_lst[i-1][0]
        y = CA_lst[i-1][1]
        z = CA_lst[i-1][2]
        line="{:6s}{:5d} {:^4s} {:3s} {:1s}{:4d}    {:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}".format('ATOM',atom_seq,'CA',res_name,'A',res_seq,x,y,z,0,0)
        f.write(line+"\n")
        j = j+1

        ## C ccordinates
        atom_seq= j
        res_name = d_inverse[aa]
        res_seq= i
        x = C_lst[i-1][0]
        y = C_lst[i-1][1]
        z = C_lst[i-1][2]
        line="{:6s}{:5d} {:^4s} {:3s} {:1s}{:4d}    {:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}".format('ATOM',atom_seq,'C',res_name,'A',res_seq,x,y,z,0,0)
        f.write(line+"\n")
        j = j+1

        i = i+1

def makedir_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    directory = os.path.abspath(directory)
    return directory


fasta = sys.argv[1]
xyz_file = sys.argv[2]
outdir = sys.argv[3]
makedir_if_not_exists(outdir)    
os.chdir(outdir)

fasta_name = os.path.splitext(os.path.basename(fasta))[0]
xyz_name = os.path.splitext(os.path.basename(xyz_file))[0]

N_lst, CA_lst, C_lst = get_xyz(xyz_file)
xyz2pdb(fasta_name, fasta, N_lst, CA_lst, C_lst, outdir)
