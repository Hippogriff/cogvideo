#!/bin/bash

# Ensure diffusers and other dependencies are installed
# pip install -r requirements.txt

# Usage: ./run_inference_i2v.sh <image_path> [prompt]

IMAGE_PATH=${1:-"resources/web_demo.png"}
PROMPT=${2:-"A girl riding a bike."}
MODEL_PATH="THUDM/CogVideoX-5b-I2V" # or THUDM/CogVideoX1.5-5b-I2V

echo "Running Image-to-Video Inference with:"
echo "Image: $IMAGE_PATH"
echo "Prompt: $PROMPT"
echo "Model: $MODEL_PATH"

python inference/cli_demo.py \
    --prompt "$PROMPT" \
    --model_path "$MODEL_PATH" \
    --image_or_video_path "$IMAGE_PATH" \
    --generate_type "i2v" \
    --guidance_scale 6.0 \
    --num_inference_steps 50 \
    --fps 16
