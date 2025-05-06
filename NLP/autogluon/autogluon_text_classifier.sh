#!/bin/bash

# Environment setup
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Training parameters
# TRAIN_FILE="/cosmos/local/IndexQuality/ContentModels/DataAugmentation/data/CBSpam_v3/QuerySensitivity/data_all5.tsv"  # Replace with your data file path
TRAIN_FILE='/cosmos/local/IndexQuality/ContentModels/DataAugmentation/data/CBSpam_v3/QuerySensitivity/data_all6_for_autogluon1.tsv'
TEXT_COL="prompt"  # Replace with your text column name
LABEL_COL="collapsed_label"  # Replace with your label column name
# OUTPUT_DIR="models/autogluon_multimodal_classifier"  # Output directory for all models
OUTPUT_DIR="models/autogluon_multimodal_classifier_for_collapsed_label"  # Output directory for all models
MAX_LENGTH=2048
BATCH_SIZE=256    # Fixed batch size as requested  64对deberta OOM 
LR=2e-5
NUM_EPOCHS=5     # Fixed number of epochs as requested
CHECKPOINT_NAME='deberta-v3-base'  # Replace with your desired checkpoint name

# Create output directory
mkdir -p $OUTPUT_DIR

# Run training with all models
python autogluon_text_classifier.py \
    --train_file $TRAIN_FILE \
    --train_columns "${TEXT_COL},${LABEL_COL}" \
    --output_dir $OUTPUT_DIR \
    --batch_size $BATCH_SIZE \
    --max_length $MAX_LENGTH \
    --learning_rate $LR \
    --num_epochs $NUM_EPOCHS \
    --checkpoint_name $CHECKPOINT_NAME

# Note: Required packages
# rm -rf /cosmos/local/IndexQuality/ContentModels/DataAugmentation/data/CBSpam_v3/Code/GeneralCodeRepo/Project/QuerySensitivity/AutoGluOn/models/autogluon_multimodal_classifier/ModernBERT-base_risk_type_label
# rm -rf /cosmos/local/IndexQuality/ContentModels/DataAugmentation/data/CBSpam_v3/Code/GeneralCodeRepo/Project/QuerySensitivity/AutoGluOn/models/autogluon_multimodal_classifier_for_collapsed_label/deberta-v3-large_collapsed_label
# 对ModernBERT-base，如果单设env.per_gpu_batch_size，最大只能是32了(32*4=128)，而如果单设env.batch_size大于env.per_gpu_batch_size * env.num_gpus，则会自动accumulate gradients. 所以还是设整体的合适，目前先设置成256跑着吧