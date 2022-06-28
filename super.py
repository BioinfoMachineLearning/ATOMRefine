import numpy as np
import math

import os,sys

d = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
     'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N', 
     'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W', 
     'ALA': 'A', 'VAL':'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M'}

def get_rot_mat(rot_mat):
    rotation = np.zeros((3,4))
    count = 0
    flag = 0
    for line in open(rot_mat):
        if "rotation matrix to rotate Chain-1 to Chain-2" in line:
            count = 1
            flag =1 
        if count == 3:
            arr = line.strip().split()
            rotation[0,:] = (float(arr[1]),float(arr[2]),float(arr[3]),float(arr[4]))
        if count == 4:
            arr = line.strip().split()
            rotation[1,:] = (float(arr[1]),float(arr[2]),float(arr[3]),float(arr[4]))
        if count == 5:
            arr = line.strip().split()
            rotation[2,:] = (float(arr[1]),float(arr[2]),float(arr[3]),float(arr[4]))
            break
        if flag:
            count = count+1
    return rotation

def rotate_pdb(pdb1_name,pdb1,rotation):
    XYZ = []
    #f = open(pdb1_name+"_rotate.pdb",'w')
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
            X=rotation[0,0]+rotation[0,1]*x+rotation[0,2]*y+rotation[0,3]*z
            Y=rotation[1,0]+rotation[1,1]*x+rotation[1,2]*y+rotation[1,3]*z
            Z=rotation[2,0]+rotation[2,1]*x+rotation[2,2]*y+rotation[2,3]*z
            if (atom_name == "N") or (atom_name == "CA") or (atom_name == "C"):
                XYZ.append([X,Y,Z])
            occ=line_f[9]
            tmp=line_f[10]
            ele=line_f[11]
            line="{:6s}{:5d} {:^4s} {:3s} {:1s}{:4d}    {:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}          {:>2s}  ".format(atom,atom_seq,atom_name,res_name,chain,res_seq,X,Y,Z,float(occ),float(tmp),ele)
            #f.write(line+'\n')
        if line.startswith('TER'):
            #f.write('TER\n')
            break
    #rot_pdb = pdb1_name+"_rotate.pdb"
    return XYZ

def get_coord(pdb2):
    XYZ2 = []
    for line in open(pdb2,"r"):
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
            if (atom_name == "N") or (atom_name == "CA") or (atom_name == "C"):
                XYZ2.append([x,y,z])
            occ=line_f[9]
            tmp=line_f[10]
            ele=line_f[11]
            line="{:6s}{:5d} {:^4s} {:3s} {:1s}{:4d}    {:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}          {:>2s}  ".format(atom,atom_seq,atom_name,res_name,chain,res_seq,x,y,z,float(occ),float(tmp),ele)
        if line.startswith('TER'):
            break
    return XYZ2

def makedir_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    directory = os.path.abspath(directory)
    return directory


pdb1 = sys.argv[1]
pdb2 = sys.argv[2]
outdir = sys.argv[3]
#pdb1 = "/storage/htc/bdm/tianqi/test/3d/Deep3D/data/predict/T1027_MULTICOM-DIST_TS1"
#pdb2 = "/storage/htc/bdm/tianqi/test/3d/Deep3D/data/true/pdb/full/T1027.pdb"
#outdir = "/storage/htc/bdm/tianqi/test/3d/Deep3D/data"
makedir_if_not_exists(outdir)    
os.chdir(outdir)

pdb1_name = os.path.splitext(os.path.basename(pdb1))[0]
pdb2_name = os.path.splitext(os.path.basename(pdb2))[0]

TMscore = "/storage/htc/bdm/tools/TMscore"

XYZ_lst = []

XYZ_init = get_coord(pdb2)
XYZ_init = np.array(XYZ_init)
L,_ = XYZ_init.shape
XYZ_init = XYZ_init.reshape(int(L/3),9)
XYZ_lst.append(XYZ_init)

for i in range(1,6):
    pdb1 = pdb1_name+str(i)+".pdb"
    cmd = TMscore+" "+pdb1+" "+pdb2+" > "+pdb1_name+"_"+pdb2_name+".txt"
    print(cmd)
    os.system(cmd)

    rotation = get_rot_mat(pdb1_name+"_"+pdb2_name+".txt")
    XYZ = rotate_pdb(pdb1_name,pdb1,rotation)
    XYZ = np.array(XYZ)
    L,_ = XYZ.shape
    XYZ = XYZ.reshape(int(L/3),9)
    XYZ_lst.append(XYZ)
    os.system("rm "+pdb1_name+"_"+pdb2_name+".txt")

XYZ = np.mean(np.array(XYZ_lst),axis=0)
np.save(pdb2_name,np.array(XYZ))