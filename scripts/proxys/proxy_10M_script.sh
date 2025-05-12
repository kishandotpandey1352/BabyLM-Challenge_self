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
cd /users/{username}/BabyLM-Challenge # change user specific for your user
# export PYTHONPATH to make sure python finds local packages (change for your user) so bash can find it
export PYTHONPATH="/users/{username}/BabyLM-Challenge:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=1



nvidia-smi
free -h
nvidia-smi --query-compute-apps=pid,used_memory --format=csv
nvidia-smi --loop=1000 > logs/gpu_usage.log &

# Run with passed arguments
python main/proxy_main.py \
    --data_path "$DATA_PATH" \
    --data_size "$DATA_SIZE"
