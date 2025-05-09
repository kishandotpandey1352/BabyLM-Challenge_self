#!/bin/bash


#SBATCH --job-name=babylm-gpt-proxy-training
#SBATCH --output=logs/proxy.out
#SBATCH --error=logs/proxy.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=30:00:00
#SBATCH --gres=gpu:1          
#SBATCH --partition=gpu  

source ~/.bashrc 
conda activate babylm

python main/proxy_main.py \
    --data_path tokenizers/10M_data_token.pkl