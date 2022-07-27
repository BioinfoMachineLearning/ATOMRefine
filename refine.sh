#!/bin/bash -e
# Tianqi Wu, 01-31-2022
# The main script for SE3 all-atom level refinement

if [ $# != 4 ]; then
        echo "sh $0 <init_pdb> <target_id> <seq_length> <outdir>"
        exit
fi

ROOT=$(dirname $0)
ROOT=$(realpath $ROOT)

init_pdb=$1
targetid=$2
seq_l=$3
outdir=$4

conda activate ATOMRefine

### Check if the initial pdb file does not exist ###
if [ ! -f $init_pdb ] 
then
    echo "Initial pdb file $init_pdb DOES NOT exists....Please check!!!" 
    exit
fi
init_pdb=$(realpath $init_pdb)

[ ! -d $outdir ] && mkdir -p $outdir

CUDA_VISIBLE_DEVICES=-1 python $ROOT/predict.py --init $init_pdb --id $targetid --seql $seq_l --out_path $outdir --test

