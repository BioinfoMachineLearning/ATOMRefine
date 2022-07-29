<div align="center">
  
# ATOMRefine: 3D equivariant neural networks for all-atom refinement of protein tertiary structures

![ATOMRefine Architecture](https://github.com/BioinfoMachineLearning/ATOMRefine/blob/main/img/ATOMRefine_Architecture.png)
  
</div>

## Description
Atomic protein structure refinement using all-atom graph representations and SE(3)–equivariant graph neural networks

## Installation
```bash
git clone https://github.com/BioinfoMachineLearning/ATOMRefine.git
cd ATOMRefine
conda env create -f ATOMRefine-linux-cu101.yml
```

## Prediction
```bash
conda activate ATOMRefine
sh refine.sh <init_pdb> <target_id> <seq_length> <outdir>

Inputs:
init_pdb: starting model in pdb format
target_id: protein target id
seq_length: protein sequence seq_length
outdir: output folder

e.g.  sh refine.sh example/T1062.pdb T1062 35 output
Expected outputs:
Five refined models in pdb format
```

## Data


## Training
```bash
sh train.sh

python train.py --data <data_dir> --out_path <out_dir> --lst 1
lst: training set id (1 - 10) as 10 folds
```

## References

Tianqi Wu and Jianlin Cheng. Atomic protein structure refinement using all-atom graph representations and SE(3)-equivariant graph neural networks. bioRxiv, 2022. [link](https://doi.org/10.1101/2022.05.06.490934)
