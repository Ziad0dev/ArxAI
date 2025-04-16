#!/bin/bash

# Set environment variables
export CUDA_VISIBLE_DEVICES=0,1,2,3  # Adjust based on available GPUs
export TOKENIZERS_PARALLELISM=false

# Create directories
mkdir -p output
mkdir -p cache
mkdir -p logs

# Set timestamp for this run
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/training_${TIMESTAMP}.log"

# Default configuration
TRAIN_DATA="data/papers_train.json"
VAL_DATA="data/papers_val.json"
TEST_DATA="data/papers_test.json"
DOMAIN_FILE="data/domains/domain_mapping.json"
OUTPUT_DIR="checkpoints/physics_ai_large_run"
CACHE_DIR="cache"

# Model configuration - even larger model for better performance
MODEL_NAME="sentence-transformers/all-mpnet-base-v2"  # Better base model
D_MODEL=768
N_HEADS=12
N_LAYERS=10     # Increased from 8 to 10
D_FF=3072
DROPOUT=0.1
MAX_SEQ_LENGTH=1024  # Increased from 768 to 1024

# Training configuration - bigger run
BATCH_SIZE=16        # Adjusted based on memory constraints
EVAL_BATCH_SIZE=32
NUM_EPOCHS=15        # Increased from 10 to 15
LEARNING_RATE=2e-5   # Slightly reduced for more stable training
WEIGHT_DECAY=0.01
WARMUP_STEPS=3000    # Increased from 2000 to 3000
NUM_WORKERS=4
GRADIENT_ACCUMULATION_STEPS=8  # Increased for larger effective batch size
SAVE_STEPS=1000      # More frequent checkpoints
EVAL_STEPS=500       # More frequent evaluation
SAVE_TOTAL_LIMIT=3   # Keep only the 3 most recent checkpoints

# Run training with progress monitoring
echo "Starting LARGE training run at $(date)"
echo "Logs will be saved to ${LOG_FILE}"

python -u train.py \
  --train_data ${TRAIN_DATA} \
  --val_data ${VAL_DATA} \
  --test_data ${TEST_DATA} \
  --domain_file ${DOMAIN_FILE} \
  --output_dir ${OUTPUT_DIR} \
  --cache_dir ${CACHE_DIR} \
  --model_name ${MODEL_NAME} \
  --d_model ${D_MODEL} \
  --n_heads ${N_HEADS} \
  --n_layers ${N_LAYERS} \
  --d_ff ${D_FF} \
  --dropout ${DROPOUT} \
  --max_seq_length ${MAX_SEQ_LENGTH} \
  --batch_size ${BATCH_SIZE} \
  --eval_batch_size ${EVAL_BATCH_SIZE} \
  --num_epochs ${NUM_EPOCHS} \
  --learning_rate ${LEARNING_RATE} \
  --weight_decay ${WEIGHT_DECAY} \
  --warmup_steps ${WARMUP_STEPS} \
  --num_workers ${NUM_WORKERS} \
  --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
  --save_steps ${SAVE_STEPS} \
  --eval_steps ${EVAL_STEPS} \
  --save_total_limit ${SAVE_TOTAL_LIMIT} \
  --fp16 \
  2>&1 | tee ${LOG_FILE}

echo "Training completed at $(date)"

# For multi-GPU training with DDP, uncomment and adjust the following:
# WORLD_SIZE=$NGPU
# python -m torch.distributed.launch --nproc_per_node=$NGPU train.py \
#   --use_ddp \
#   [all the same parameters as above] 