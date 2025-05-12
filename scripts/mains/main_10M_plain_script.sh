#!/bin/bash

#SBATCH --job-name=main10
#SBATCH --output=logs/main10M.out
#SBATCH --error=logs/main10M.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=128G
#SBATCH --time=30:00:00
#SBATCH --gres=gpu:1          
#SBATCH --partition=gpu  

source ~/.bashrc 
conda activate babylm
module load CUDA/11.7.0

# Step into the project root
cd /users/acp24kp/BabyLM-Challenge # change user specific for your user
# export PYTHONPATH to make sure python finds local packages (change for your user) so bash can find it
export PYTHONPATH="/users/acp24kp/BabyLM-Challenge:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=1



nvidia-smi
free -h
nvidia-smi --query-compute-apps=pid,used_memory --format=csv
nvidia-smi --loop=1000 > logs/gpu_usage.log &

python main/gpt_model.py \
    --data_path tokenizers/10M_data_token.pkl \
    --scoring composite \
    --schedule_type sigmoid \
    --curriculum off \
    --data_size 10M
