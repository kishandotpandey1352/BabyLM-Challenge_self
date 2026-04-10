#!/bin/bash

#SBATCH --job-name=export_hf_model
#SBATCH --output=logs/export_model.out
#SBATCH --error=logs/export_model.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --time=00:15:00

# === Load environment ===
source ~/.bashrc
conda activate babylm

# === Limit threading to avoid OpenBLAS crash ===
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# === Move to project directory ===
cd /users/acp24kp/BabyLM-Challenge

# === Run export script ===
python export_custom_to_hf_compat.py
