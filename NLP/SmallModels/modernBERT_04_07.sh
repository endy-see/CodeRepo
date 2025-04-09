#!/bin/bash

# Script to launch distributed training on 4 A100 GPUs
accelerate launch \
    --multi_gpu \
    --num_processes 4 \
    --gpu_ids "0,1,2,3" \
    --mixed_precision "fp16" \
    modernBERT_04_07.py \
    --train_file "/cosmos/local/IndexQuality/ContentModels/DataAugmentation/data/ModernBERT/data/TrainingSet.tsv" \
    --output_dir "./checkpoints" \
    --cache_dir "./url_cache" \
    --num_labels 2 \
    --batch_size 32 \
    --gradient_accumulation_steps 1 \
    --num_epochs 3 \
    --max_length 512 \
    --learning_rate 2e-5