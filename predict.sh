#!/bin/bash

# Inference script for receipt text detection

CUDA_VISIBLE_DEVICES=0 python tools/predict.py \
    --model_path "output/DBNet_Receipt_Detection/checkpoint/model_best.pth" \
    --input_folder "test_images/" \
    --output_folder "results/" \
    --thre 0.3 \
    --polygon \
    --save_result
    # Remove --show for batch processing
