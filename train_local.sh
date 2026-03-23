#!/bin/bash
# Local multi-GPU training via HF Accelerate.
#
# Default: batch_size=128 x 2 GPUs = 256 global batch (matches original BERT).
#
# Usage:
#   ./train_local.sh bert_mlm   # Classic BERT MLM
#   ./train_local.sh dlm        # Modern DLM
#
# Environment variables (optional):
#   MAX_STEPS=100000    BATCH_SIZE=128   NUM_GPUS=2
#   LR=1e-4             SEED=42          GRAD_ACCUM=1
#   SAVE_STEPS=10000

set -e

METHOD="${1:?Usage: $0 <bert_mlm|dlm>}"

MAX_STEPS="${MAX_STEPS:-100000}"
BATCH_SIZE="${BATCH_SIZE:-128}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
LR="${LR:-1e-4}"
SEED="${SEED:-42}"
SAVE_STEPS="${SAVE_STEPS:-10000}"
NUM_GPUS="${NUM_GPUS:-2}"

OUTPUT_DIR="_saved_models/${METHOD}_$(date +%Y%m%d_%H%M%S)"

GLOBAL_BATCH=$((BATCH_SIZE * GRAD_ACCUM * NUM_GPUS))

echo "========================================"
echo "dBERT Training"
echo "  Method:    $METHOD"
echo "  GPUs:      $NUM_GPUS"
echo "  Steps:     $MAX_STEPS"
echo "  Batch:     ${BATCH_SIZE} x ${GRAD_ACCUM} x ${NUM_GPUS} = ${GLOBAL_BATCH} global"
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
