import numpy as np
import math

import os,sys

d = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
     'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N', 
     'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W', 
     'ALA': 'A', 'VAL':'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M'}


def get_backbone(pdb1_name,pdb1,outdir):
    XYZ = []
    f = open(outdir+"/"+pdb1_name+".pdb",'w')
    for line in open(pdb1,"r"):
        if line.startswith('ATOM'):
            line_f=[line[:6].strip(), line[6:11].strip(), line[12:16].strip(), line[17:20].strip(), line[21].strip(), line[22:26].strip(), line[30:38].strip(), line[38:46].strip(), line[46:54].strip(),line[54:60].strip(),line[60:66].strip(),line[76:78].strip(),line[78:80].strip()]
            atom='ATOM'
            atom_seq=int(line_f[1])
            atom_name=line_f[2]
            res_name =line_f[3]
            chain = line_f[4]
            res_seq=int(line_f[5])
            x=float(line_f[6])
            y=float(line_f[7])
            z=float(line_f[8])
            occ=line_f[9]
            tmp=line_f[10]
            ele=line_f[11]
            if (atom_name == "N") or (atom_name == "CA") or (atom_name == "C"):
                f.write(line)
        if line.startswith('TER'):
            f.write('TER\n')
            break


def makedir_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    directory = os.path.abspath(directory)
    return directory


pdb1 = sys.argv[1]
outdir = sys.argv[2]

makedir_if_not_exists(outdir)
os.chdir(outdir)

pdb1_name = os.path.splitext(os.path.basename(pdb1))[0]

get_backbone(pdb1_name,pdb1,outdir)
