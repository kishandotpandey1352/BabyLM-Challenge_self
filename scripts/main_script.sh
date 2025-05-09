#!/bin/bash


SCORING="composite"
INIT_BETA="0.3"
SCHEDULE_TYPE="sigmoid"

#SBATCH --job-name=babylm-gpt
#SBATCH --output=logs/hps.out
#SBATCH --error=logs/hps.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=30:00:00
#SBATCH --gres=gpu:1          
#SBATCH --partition=gpu  

source ~/.bashrc 
conda activate babylm

python main/gpt_model.py \
    --data_path tokenizers/10M_data_token.pkl \
    --scoring ${SCORING} \
    --init_beta ${INIT_BETA} \
    --schedule_type ${SCHEDULE_TYPE}