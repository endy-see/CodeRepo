#!/bin/bash

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
mixed_precision: "no"
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

# Specify paths
MODEL_PATH="/cosmos/local/IndexQuality/ContentModels/DataAugmentation/data/CBSpam_v3/Code/GeneralCodeRepo/Project/QuerySensitivity/checkpoints/models_modernBERT_optimize4/best_model_f1_0.7428_epoch_7"  # Path to your best model directory
# MODEL_PATH="/cosmos/local/IndexQuality/ContentModels/DataAugmentation/data/CBSpam_v3/Code/GeneralCodeRepo/Project/QuerySensitivity/checkpoints/models_modernBERT_optimize4/latest_model"
# TEST_FILE="/cosmos/local/IndexQuality/ContentModels/DataAugmentation/data/CBSpam_v3/QuerySensitivity/valid_set_prompt_risk_type_label.tsv"   # Path to your test_set2.tsv file
# TEST_FILE="/cosmos/local/IndexQuality/ContentModels/DataAugmentation/data/CBSpam_v3/QuerySensitivity/test_set_prompt_risk_type_label.tsv"   # Path to your test_set2.tsv file
TEST_FILE="/cosmos/local/IndexQuality/ContentModels/DataAugmentation/data/CBSpam_v3/QuerySensitivity/FY25H2_testsets_processed.tsv"   # Path to your test_set2.tsv file

CACHE_DIR="cache_modernBERT_eval"  # Optional cache directory
OUTPUT_FILE="/cosmos/local/IndexQuality/ContentModels/DataAugmentation/data/CBSpam_v3/Code/GeneralCodeRepo/Project/QuerySensitivity/checkpoints/models_modernBERT_optimize4/eval_results.txt" 

# python modernBERT_04_07_optimized1_eval.py \
#     --model_path $MODEL_PATH \
#     --test_file $TEST_FILE \
#     --max_length 2048 \
#     --batch_size 256 \
#     --output_file $OUTPUT_FILE \

ACCELERATE_CONFIG_FILE=accelerate_config.yaml accelerate launch \
    --multi_gpu \
    --num_processes 4 \
    --num_machines 1 \
    --dynamo_backend inductor \
    modernBERT_04_07_optimized1_eval.py \
    --model_path $MODEL_PATH \
    --test_file $TEST_FILE \
    --max_length 2048 \
    --batch_size 256 \
    --output_file $OUTPUT_FILE \