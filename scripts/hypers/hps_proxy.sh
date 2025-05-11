#!/bin/bash

#SBATCH --job-name=proxy_hp
#SBATCH --output=logs/proxy_hps.out
#SBATCH --error=logs/proxy_hps.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=312G
#SBATCH --time=30:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

source ~/.bashrc 
conda activate babylm
module load CUDA/11.7.0


# Step into the project root
cd /users/#username#/BabyLM-Challenge # change user specific for your user

nvidia-smi
free -h
nvidia-smi --query-compute-apps=pid,used_memory --format=csv

# export PYTHONPATH to make sure python finds local packages (change for your user) so bash can find it
export PYTHONPATH="/users/#username#/BabyLM-Challenge:$PYTHONPATH"

# Run the script from BabyLM-Challenge
python hps/param_tuning.py \
    --data_path tokenizers/100M_data_token.pkl \
    --toggle_scheduler off \
    --score_type composite \
    --n_tokens 500_000 \
    --proxy_n_trials 10 \
    --main_n_trials 50 \
    --proxy_train on \
    --main_train off
