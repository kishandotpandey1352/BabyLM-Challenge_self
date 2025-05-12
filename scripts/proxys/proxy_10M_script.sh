#!/bin/bash

#SBATCH --job-name=proxy10
#SBATCH --output=logs/proxy10M.out
#SBATCH --error=logs/proxy10M.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=128G
#SBATCH --time=30:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

# Accept arguments from command line
DATA_PATH=$1
DATA_SIZE=$2

source ~/.bashrc
conda activate babylm
module load CUDA/11.7.0

# Step into the project root
cd /users/acp24kp/babylm/BabyLM-Challenge

nvidia-smi
free -h
nvidia-smi --query-compute-apps=pid,used_memory --format=csv

# Make local project discoverable
export PYTHONPATH="/users/acp24kp/babylm/BabyLM-Challenge:$PYTHONPATH"
export CUDA_LAUNCH_BLOCKING=1

# Run with passed arguments
python main/proxy_main.py \
    --data_path "$DATA_PATH" \
    --data_size "$DATA_SIZE"