#!/bin/bash
#SBATCH --job-name=gpu_test
#SBATCH --output=logs/gpu_test_%j.out
#SBATCH --error=logs/gpu_test_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --partition=gpu

# Load environment
module load anaconda/2023a
source activate gptcurriculum

# Output basic info
echo "Hostname: $HOSTNAME"
echo "SLURM Job GPUs: $SLURM_JOB_GPUS"

# Run GPU check
nvidia-smi