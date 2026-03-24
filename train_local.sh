#!/bin/bash
# Local multi-GPU training via HF Accelerate.
#
# Auto-detects GPUs from CUDA_VISIBLE_DEVICES (or all available).
# Adjusts gradient accumulation to keep global batch = 256.
#
# Usage:
#   ./train_local.sh mlm              # Classic BERT MLM
#   ./train_local.sh dlm              # Modern DLM
#   CUDA_VISIBLE_DEVICES=0,1 ./train_local.sh dlm   # Use specific GPUs
#
# Environment variables (optional):
#   MAX_STEPS=100000    BATCH_SIZE=64    LR=1e-4
#   SEED=42             SAVE_STEPS=10000
#   GLOBAL_BATCH=256    NUM_GPUS=<auto>  GRAD_ACCUM=<auto>

set -e

METHOD="${1:?Usage: $0 <mlm|dlm>}"

MAX_STEPS="${MAX_STEPS:-100000}"
BATCH_SIZE="${BATCH_SIZE:-128}"
LR="${LR:-1e-4}"
SEED="${SEED:-42}"
SAVE_STEPS="${SAVE_STEPS:-10000}"
GLOBAL_BATCH="${GLOBAL_BATCH:-256}"

# Auto-detect GPU count from CUDA_VISIBLE_DEVICES or torch
if [ -z "$NUM_GPUS" ]; then
    if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
        NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
    else
        NUM_GPUS=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 1)
    fi
fi

# Auto-compute gradient accumulation to hit GLOBAL_BATCH
if [ -z "$GRAD_ACCUM" ]; then
    GRAD_ACCUM=$((GLOBAL_BATCH / (BATCH_SIZE * NUM_GPUS)))
    if [ "$GRAD_ACCUM" -lt 1 ]; then
        GRAD_ACCUM=1
    fi
fi

EFFECTIVE_BATCH=$((BATCH_SIZE * GRAD_ACCUM * NUM_GPUS))

OUTPUT_DIR="_saved_models/${METHOD}_$(date +%Y%m%d_%H%M%S)"

echo "========================================"
echo "dBERT Training"
echo "  Method:    $METHOD"
echo "  GPUs:      $NUM_GPUS"
echo "  Steps:     $MAX_STEPS"
echo "  Batch:     ${BATCH_SIZE} x ${GRAD_ACCUM} x ${NUM_GPUS} = ${EFFECTIVE_BATCH} global"
echo "  Output:    $OUTPUT_DIR"
echo "========================================"

accelerate launch \
    --num_processes "$NUM_GPUS" \
    --mixed_precision bf16 \
    train.py \
    --method "$METHOD" \
    --output_dir "$OUTPUT_DIR" \
    --max_steps "$MAX_STEPS" \
    --batch_size "$BATCH_SIZE" \
    --gradient_accumulation_steps "$GRAD_ACCUM" \
    --learning_rate "$LR" \
    --save_steps "$SAVE_STEPS" \
    --seed "$SEED"
