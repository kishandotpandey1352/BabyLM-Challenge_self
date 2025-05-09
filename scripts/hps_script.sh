#!/bin/bash

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

# adjust path for ur hpc user
cd /users/acp24mw/BabyLM-Challenge

export PYTHONPATH=$(pwd)

# Run the script
python hps/param_tuning.py \
    --data_path tokenizers/10M_data_token.pkl \
    --toggle_scheduler 'off' \
    --score_type 'composite' \
    --n_tokens 1000 \
    --proxy_n_trials 1 \
    --main_n_trials 1
