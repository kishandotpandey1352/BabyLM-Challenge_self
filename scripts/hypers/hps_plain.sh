#!/bin/bash

#SBATCH --job-name=plain_main
#SBATCH --output=logs/plain_hps.out
#SBATCH --error=logs/plain_hps.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=256G
#SBATCH --time=30:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

source ~/.bashrc 
conda activate babylm-gpu
module load CUDA/12.4


# Step into the project root
cd /users/{username}/BabyLM-Challenge # change user specific for your user
# export PYTHONPATH to make sure python finds local packages (change for your user) so bash can find it
export PYTHONPATH="/users/{username}/BabyLM-Challenge:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=1



nvidia-smi
free -h
nvidia-smi --query-compute-apps=pid,used_memory --format=csv
nvidia-smi --loop=30 > logs/gpu_usage.log &


# Run the script from BabyLM-Challenge
python hps/param_tuning.py \
    --data_path tokenizers/100M_data_token.pkl \
    --toggle_scheduler off \
    --score_type composite \
    --n_tokens 500_000 \
    --proxy_n_trials 10 \
    --main_n_trials 50 \
    --proxy_train off \
    --main_train on
