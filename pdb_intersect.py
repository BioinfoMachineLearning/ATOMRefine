import collections
import sys
atom_data = collections.OrderedDict()  # atom_uid: line
records = ('ATOM', 'TER')

pdb1 = sys.argv[1]
pdb2 = sys.argv[2]

ref = open(pdb1,"r")
for line in ref:
    if line.startswith(records):
        atom_uid = line[12:27]
        atom_data[atom_uid] = line
ref.close()

common_atoms = set(atom_data.keys())

file_atoms = set()
file_data = collections.OrderedDict()
fhandle = open(pdb2,"r")
for line in fhandle:
    atom_uid = line[12:27]
    file_data[atom_uid] = line
    file_atoms.add(atom_uid)
fhandle.close()

common_atoms = common_atoms & file_atoms

for atom in atom_data:
    if atom not in common_atoms:
        print(atom_data[atom].rstrip())
    else:
        #print(atom_data[atom].rstrip() , file_data[atom].rstrip())
        line = atom_data[atom]
        line_f=[line[:6].strip(), line[6:11].strip(), line[12:16].strip(), line[17:20].strip(), line[21].strip(), line[22:26].strip(), line[30:38].strip(), line[38:46].strip(), line[46:54].strip(
),line[54:60].strip(),line[60:66].strip(),line[76:78].strip(),line[78:80].strip()]
        atom_seq=int(line_f[1])
        atom_name=line_f[2]
        res_name =line_f[3]
        chain = line_f[4]
        res_seq=int(line_f[5])
        x=float(line_f[6])
        y=float(line_f[7])
        z=float(line_f[8])
        line = file_data[atom]
        line_f=[line[:6].strip(), line[6:11].strip(), line[12:16].strip(), line[17:20].strip(), line[21].strip(), line[22:26].strip(), line[30:38].strip(), line[38:46].strip(), line[46:54].strip(
),line[54:60].strip(),line[60:66].strip(),line[76:78].strip(),line[78:80].strip()]
        x2 = float(line_f[6])
        y2 = float(line_f[7])
        z2 = float(line_f[8])
        line="{:6s}{:5d} {:^4s} {:3s} {:1s}{:4d}    {:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}".format('ATOM',atom_seq,atom_name,res_name,chain,res_seq,(x+x2)/2,(y+y2)/2,(z+z2)/2,0,0)
        print(line)

for atom in file_data:
    if atom not in common_atoms:
        print(file_data[atom].rstrip())