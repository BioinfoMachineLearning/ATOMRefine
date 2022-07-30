#!/bin/bash -e

module load cuda/11.1.0
module load miniconda3
source activate RoseTTAFold

python train.py --data data/train_lst/model --out_path model --lst 1