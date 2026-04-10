#!/bin/bash

# === Request resources non-interactively ===
# This script should be submitted using sbatch, NOT srun or bash
# Submit it with: sbatch run_full_pipeline.sh

#SBATCH --job-name=5_full_pipeline_epoch
#SBATCH --output=logs/full_pipeline_5.out
#SBATCH --error=logs/full_pipeline_5.err
#SBATCH --time=96:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

# === Environment Setup ===
source ~/.bashrc
conda activate babylm



# === Submit Main Training (Comp_score Scoring) ===
dos2unix scripts/mains/main_10M_plain_script.sh
chmod +x scripts/mains/main_10M_plain_script.sh
echo "Submitting main model 10M (plain) training job with  comp_score..."
sbatch scripts/mains/main_10M_plain_script.sh
