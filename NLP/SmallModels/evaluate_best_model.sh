#!/bin/bash

# Set environment variables for GPU usage
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Set paths
MODEL_PATH="checkpoints/query_sensitivity_2025_0414_080050/best_model"
DATA_FILE="/cosmos/local/IndexQuality/ContentModels/DataAugmentation/data/CBSpam_v3/QuerySensitivity/data_all1.tsv"

# Create logs directory
LOGS_DIR="logs"
mkdir -p $LOGS_DIR

timestamp=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOGS_DIR/evaluation_${timestamp}.log"

echo "Starting evaluation of best model..."
echo "Model path: $MODEL_PATH"
echo "Data file: $DATA_FILE"

# Run evaluation with all GPUs
python evaluate_best_model.py \
    --model_path "$MODEL_PATH" \
    --data_file "$DATA_FILE" \
    --max_length 1024 \
    --batch_size 32 \
    --num_workers 4 2>&1 | tee "$LOG_FILE"

# Check if evaluation was successful
if [ $? -eq 0 ]; then
    echo "Evaluation completed successfully. Results saved to $LOG_FILE"
else
    echo "Evaluation failed. Check $LOG_FILE for details"
    exit 1
fi