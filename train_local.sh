#!/bin/bash
# Local multi-GPU training via HF Accelerate.
#
# Usage:
#   ./train_local.sh bert_mlm   # Classic BERT MLM
#   ./train_local.sh dlm        # Modern DLM
#
# Environment variables (optional):
#   MAX_STEPS=100000    BATCH_SIZE=16    GRAD_ACCUM=4
#   LR=1e-4             SEED=42          MAX_LENGTH=512
#   SAVE_STEPS=10000    NUM_GPUS=auto

set -e

METHOD="${1:?Usage: $0 <bert_mlm|dlm>}"

MAX_STEPS="${MAX_STEPS:-100000}"
BATCH_SIZE="${BATCH_SIZE:-16}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
LR="${LR:-1e-4}"
SEED="${SEED:-42}"
MAX_LENGTH="${MAX_LENGTH:-512}"
SAVE_STEPS="${SAVE_STEPS:-10000}"

OUTPUT_DIR="_saved_models/${METHOD}_$(date +%Y%m%d_%H%M%S)"

if [ "${NUM_GPUS}" = "auto" ] || [ -z "${NUM_GPUS}" ]; then
    NUM_GPUS=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 1)
fi

echo "========================================"
echo "dBERT Training"
echo "  Method:    $METHOD"
echo "  GPUs:      $NUM_GPUS"
echo "  Steps:     $MAX_STEPS"
echo "  Batch:     ${BATCH_SIZE} x ${GRAD_ACCUM} x ${NUM_GPUS}"
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
    --max_length "$MAX_LENGTH" \
    --save_steps "$SAVE_STEPS" \
    --seed "$SEED"
