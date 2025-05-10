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

source ~/.bashrc
conda activate babylm
module load CUDA/11.7.0

# Step into the project root
cd /users/acp24mw/BabyLM-Challenge # change user specific for your user

nvidia-smi
free -h

nvidia-smi --query-compute-apps=pid,used_memory --format=csv

# export PYTHONPATH to make sure python finds local packages (change for your user) so bash can find it
export PYTHONPATH="/users/acp24mw/BabyLM-Challenge:$PYTHONPATH"

python main/proxy_main.py \
    --data_path tokenizers/10M_data_token.pkl \
    --data_size 10M