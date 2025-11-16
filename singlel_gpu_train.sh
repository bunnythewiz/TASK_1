#!/bin/bash

# Single GPU training for receipt detection
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
    --config_file "config/receipt_detection.yaml"
