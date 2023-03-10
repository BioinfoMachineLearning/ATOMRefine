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
cd YOUR_ENV/lib/python3.8/site-packages
patch -p0 < ATOMRefine/amber/openmm.patch
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
All the required data for training are provided as below and avaiable at [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6944368.svg)](https://doi.org/10.5281/zenodo.6944368):
* Alphafold2 models (AF2_model.tar.gz)
* target.lst for training (AF2 id and its corresponding true pdb id)
* True experimental structures (true_experimental_structure.tar.gz)

## Training
```bash
cd data
wget https://zenodo.org/record/6944368/files/AF2_model.tar.gz
wget https://zenodo.org/record/6944368/files/true_experimental_structure.tar.gz
tar xvzf AF2_model.tar.gz
tar xvzf true_experimental_structure.tar.gz

conda activate ATOMRefine
python train.py --data <data_dir> --out_path <out_dir> --lst 1
lst: training set id (1 - 10) as 10 folds
e.g. python train.py --data data/train_lst --out_path model --lst 1
```

## References
[Tianqi Wu and Jianlin Cheng. Atomic protein structure refinement using all-atom graph representations and SE(3)-equivariant graph neural networks. bioRxiv, 2022.](https://doi.org/10.1101/2022.05.06.490934)

## Declaration:
The code in this repository's folder ./amber reuse the source code from [AlphaFold](https://github.com/deepmind/alphafold), which has been used under Apache-2.0 license, see the license ./amber/LICENSE.


