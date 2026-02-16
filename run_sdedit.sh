#!/bin/bash

# Usage: ./run_sdedit.sh <video_path> [prompt] [strength]

VIDEO_PATH=${1:-"resources/videos/1.mp4"}
PROMPT=${2:-"A cyberpunk city transformation"}
STRENGTH=${3:-0.7}
MODEL_PATH="THUDM/CogVideoX1.5-5b" 

echo "Running SDEdit with:"
echo "Video: $VIDEO_PATH"
echo "Prompt: $PROMPT"
echo "Strength: $STRENGTH"
echo "Model: $MODEL_PATH"

python inference/cli_sdedit.py \
    --prompt "$PROMPT" \
    --video_path "$VIDEO_PATH" \
    --model_path "$MODEL_PATH" \
    --strength "$STRENGTH" \
    --guidance_scale 6.0 \
    --num_inference_steps 50 \
    --fps 8 \
    --debug_sdedit
