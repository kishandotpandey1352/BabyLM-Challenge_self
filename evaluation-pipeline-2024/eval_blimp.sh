#!/bin/bash

MODEL_PATH=$1
DEVICE=${2:-cuda:0}  # Use 2nd argument if provided, else default to cuda:0
MODEL_BASENAME=$(basename $MODEL_PATH)

# Optional: safer execution
export CUDA_LAUNCH_BLOCKING=1

python -m lm_eval \
  --model custom_gpt2 \
  --model_args pretrained=$MODEL_PATH,backend=causal \
  --tasks blimp_filtered,blimp_supplement \
  --device $DEVICE \
  --batch_size 1 \
  --log_samples \
  --output_path results/blimp/${MODEL_BASENAME}/blimp_results.json
