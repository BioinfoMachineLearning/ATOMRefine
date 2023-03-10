import os, glob, re, sys
import numpy as np
from amber import protein
from amber import relax

if __name__ == '__main__':
  input_file = sys.argv[1]
  out_file = sys.argv[2]

  test_config = {'max_iterations': 1,
                 'tolerance': 2.39,
                 'stiffness': 10.0,
                 'exclude_residues': [],
                 'max_outer_iterations': 1,
                 'use_gpu': False}


  amber_relax = relax.AmberRelaxation(**test_config)

  with open(input_file) as f:
    test_prot = protein.from_pdb_string(f.read())
  pdb_min, debug_info, num_violations = amber_relax.process(prot=test_prot)
  f = open(out_file,"w")
  f.write(pdb_min)
  f.close()
  # try:
  #   pdb_min, debug_info, num_violations = amber_relax.process(prot=test_prot)
  #   f = open(out_file,"w")
  #   f.write(pdb_min)
  #   f.close()
  # except Exception as e:
  #   print(input_file)
