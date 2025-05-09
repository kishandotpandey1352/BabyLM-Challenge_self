#!/bin/bash

# evaluate_script.sh

# === Configuration ===
MODEL_PATH="../models/my_babyLM_model"   # Change to your model path (local folder or Hugging Face repo)
DEVICE="cuda:0"                 # Change to "cpu" if no GPU is available

# Specify output directory
OUTPUT_DIR="../Evaluation_output"
mkdir -p $OUTPUT_DIR

# === Evaluate on BLiMP ===
echo "Running BLiMP evaluation..."

lm_eval \
    --model hf \
    --model_args pretrained=$MODEL_PATH,backend=causal \
    --tasks blimp \
    --device $DEVICE \
    --output_path $OUTPUT_DIR/blimp_results.json

# === Evaluate on EWoK ===
echo "Running EWoK evaluation..."

lm_eval \
    --model hf \
    --model_args pretrained=$MODEL_PATH,backend=causal \
    --tasks ewok \
    --device $DEVICE \
    --output_path $OUTPUT_DIR/ewok_results.json

echo "Evaluation completed. Results saved in '$OUTPUT_DIR'."
