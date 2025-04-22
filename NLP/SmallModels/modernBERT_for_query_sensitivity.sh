#!/bin/bash

# Get the directory of the script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Create output directory with timestamp
timestamp=$(date +%Y%m%d_%H%M%S)
# OUTPUT_DIR="$SCRIPT_DIR/checkpoints/query_sensitivity_${timestamp}"
OUTPUT_DIR="$SCRIPT_DIR/checkpoints/models_modernBERT_optimize1_1"
mkdir -p $OUTPUT_DIR

# Create cache directory
CACHE_DIR="$SCRIPT_DIR/cache_modernBERT/query_sensitivity_for_collapsed_label"
mkdir -p $CACHE_DIR

# Set training parameters with optimized values
TRAIN_FILE="/cosmos/local/IndexQuality/ContentModels/DataAugmentation/data/CBSpam_v3/QuerySensitivity/data_all5.tsv"
BATCH_SIZE=64                         # Reduced batch size for stability
NUM_EPOCHS=5
LEARNING_RATE=2e-5                    # Reduced learning rate for stability
MAX_LENGTH=2048
GRAD_ACCUM_STEPS=4                    # Increased gradient accumulation steps
NUM_WORKERS=4

# Set NCCL environment variables for better multi-GPU performance
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_TIMEOUT=30
export NCCL_SOCKET_NTHREADS=8         # Increased socket threads
export NCCL_NSOCKS_PERTHREAD=8        # Increased socks per thread
export NCCL_MAX_RINGS=8               # Enable multiple communication rings
export NCCL_BUFFSIZE=2097152          # 2MB buffer size
export NCCL_NET_GDR_LEVEL=5           # Enable GPUDirect RDMA
export CUDA_DEVICE_MAX_CONNECTIONS=1   # Optimize CUDA connections
export TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS=1

# Configure accelerate with optimized settings
cat > accelerate_config.yaml << EOF
compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
gpu_ids: all
machine_rank: 0
main_process_ip: null
main_process_port: null
main_training_function: main
num_machines: 1
num_processes: 4
use_cpu: false
rdzv_backend: static
same_network: true
nproc_per_node: 4
deepspeed_config:
  zero_optimization:
    stage: 1
  gradient_accumulation_steps: ${GRAD_ACCUM_STEPS}
  gradient_clipping: 1.0
  offload_optimizer_device: none
  offload_param_device: none
  aio:
    block_size: 1048576
    queue_depth: 8
    overlap_events: true
EOF
# mixed_precision: bf16  # Using bfloat16 for better numerical stability

# Launch distributed training using accelerate with optimized settings
ACCELERATE_CONFIG_FILE=accelerate_config.yaml accelerate launch \
    --multi_gpu \
    --num_processes 4 \
    --num_machines 1 \
    --dynamo_backend inductor \
    --gradient_accumulation_steps ${GRAD_ACCUM_STEPS} \
    modernBERT_for_query_sensitivity.py \
    --train_file $TRAIN_FILE \
    --train_columns "prompt,collapsed_label" \
    --output_dir $OUTPUT_DIR \
    --cache_dir $CACHE_DIR \
    --batch_size $BATCH_SIZE \
    --num_epochs $NUM_EPOCHS \
    --num_labels 3 \
    --learning_rate $LEARNING_RATE \
    --max_length $MAX_LENGTH \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --num_workers $NUM_WORKERS \
    --seed 42

        # --num_workers $NUM_WORKERS \
# --mixed_precision bf16 \
# ./modernBERT_04_07_optimize1.sh
# conda install pytorch torchvision transformers accelerator
# pip install --upgrade torch torchvision transformers accelerator

