#!/bin/bash

# Evaluation script for receipt detection

CUDA_VISIBLE_DEVICES=0 python tools/eval.py \
    --model_path "output/DBNet_Receipt_Detection/checkpoint/model_best.pth" \
    --gpu_id 0
