#!/bin/bash
#SBATCH --job-name=gpt_curriculum
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --partition=gpu

# Load modules and activate environment
module load anaconda/2023a
source activate myenv  # Replace with your actual environment name

# Run training script
DATA_PATH="tokenizers/10M_data_token.pkl"   # Or use 100M version
OUTPUT_DIR="outputs/10M_run_$(date +%Y%m%d_%H%M%S)"

mkdir -p $OUTPUT_DIR

srun python main_pipeline.py \
    --data_path $DATA_PATH \
    --output_dir $OUTPUT_DIR \
    --batch_size 32
