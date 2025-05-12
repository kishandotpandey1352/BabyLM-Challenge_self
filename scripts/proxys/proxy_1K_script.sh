#!/bin/bash

#SBATCH --job-name=proxy1K
#SBATCH --output=logs/proxy1K.out
#SBATCH --error=logs/proxy1K.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=01:00:00
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

# Monitor resources
nvidia-smi
free -h
nvidia-smi --query-compute-apps=pid,used_memory --format=csv

# Ensure local packages are found
export PYTHONPATH="/users/acp24kp/babylm/BabyLM-Challenge:$PYTHONPATH"

# Run training
python main/proxy_main.py \
    --data_path "$DATA_PATH" \
    --data_size "$DATA_SIZE"
