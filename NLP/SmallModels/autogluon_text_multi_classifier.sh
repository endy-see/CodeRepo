#!/bin/bash

# Environment setup
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Training parameters
TRAIN_FILE="/cosmos/local/IndexQuality/ContentModels/DataAugmentation/data/CBSpam_v3/QuerySensitivity/data_all5.tsv"  # Replace with your data file path
TEXT_COL="prompt"  # Replace with your text column name
LABEL_COL="risk_type_label"  # Replace with your label column name
OUTPUT_DIR="models/autogluon_classifier"
MAX_LENGTH=2048
BATCH_SIZE=8
LR=2e-5
NUM_EPOCHS=10

# Create output directory
mkdir -p $OUTPUT_DIR

# Run training with all models
python autogluon_text_classifier.py \
    --train_file $TRAIN_FILE \
    --train_columns "${TEXT_COL},${LABEL_COL}" \
    --output_dir $OUTPUT_DIR \
    --max_length $MAX_LENGTH \
    --batch_size $BATCH_SIZE \
    --learning_rate $LR \
    --num_epochs $NUM_EPOCHS

# conda create -n autogluon_env python=3.10
# pip install 'thinc>=8.3.4,<8.4.0'

# pip install --no-deps \
#     autogluon.multimodal \
#     lightgbm \
#     xgboost \
#     catboost \
#     fasttext-wheel \
#     scikit-learn \
#     pandas \
#     numpy

# pip install --no-deps spacy==3.8.5