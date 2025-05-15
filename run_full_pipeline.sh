#!/bin/bash

# === Request resources non-interactively ===
# This script should be submitted using sbatch, NOT srun or bash
# Submit it with: sbatch run_full_pipeline.sh

#SBATCH --job-name=full_pipeline
#SBATCH --output=logs/full_pipeline.out
#SBATCH --error=logs/full_pipeline.err
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

# === Environment Setup ===
source ~/.bashrc
conda activate babylm

# === Fix line endings and make script executable ===
dos2unix scripts/proxys/proxy_10M_script.sh
chmod +x scripts/proxys/proxy_10M_script.sh

# === Submit Main Training (Composite Scoring) ===
dos2unix scripts/mains/main_10M_comp_script.sh
chmod +x scripts/mains/main_10M_comp_script.sh
echo "Submitting main model training job with composite scoring..."
sbatch scripts/mains/main_10M_comp_script.sh
