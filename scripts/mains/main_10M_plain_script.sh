#!/bin/bash

#SBATCH --job-name=main10_plain
#SBATCH --output=logs/main10M)plain.out
#SBATCH --error=logs/main10M_plain.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
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
nvidia-smi --loop=1000 > logs/gpu_usage.log &

python main/gpt_model.py \
    --data_path tokenizers/10M_data_token.pkl \
    --scoring comp_score \
    --curriculum off \
    --data_size 10M
