#!/bin/bash

#SBATCH --mail-user=roda9210@colorado.edu
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --time=01:00:00
#SBATCH --partition=blanca-kann
#SBATCH --gres=gpu:1
#SBATCH --mem=150G
#SBATCH --output=logs/coref.%j.log

module load anaconda
conda activate coref

# mkdir -p "/scratch/alpine/roda9210/cache/HF/transformers"
# mkdir -p "/scratch/alpine/roda9210/cache/HF/datasets"

# export TRANSFORMERS_CACHE="/scratch/alpine/roda9210/cache/HF/transformers"
# export HF_DATASETS_CACHE="/scratch/alpine/roda9210/cache/HF/datasets"

mkdir -p "$SLURM_SCRATCH/cache/HF/transformers"
mkdir -p "$SLURM_SCRATCH/cache/HF/datasets"

export TRANSFORMERS_CACHE="$SLURM_SCRATCH/cache/HF/transformers"
export HF_DATASETS_CACHE="$SLURM_SCRATCH/cache/HF/datasets"

# python3 evaluate.py -c xlm-roberta-base -p 'xlm-roberta-base-fp32/ckpt_epoch-059.pt.tar' --amp

python3 evaluate.py -c multilingual-bert-base-russian -p 'multilingual-bert-base-fp32/exp1/russian/ckpt_epoch-249.pt.tar'

# python3 evaluate.py -c multilingual-bert-base-polish -p 'multilingual-bert-base-fp32/exp1/polish/ckpt_epoch-059.pt.tar'