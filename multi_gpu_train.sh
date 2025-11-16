#!/bin/bash

# Multi-GPU training for receipt detection
export NGPUS=2  # Adjust based on available GPUs

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
    --nproc_per_node=$NGPUS \
    --master_port=29500 \
    tools/train.py --config_file "config/receipt_detection.yaml"
