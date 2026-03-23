#!/bin/bash
#SBATCH -p gh
#SBATCH -t 24:00:00
#SBATCH --ntasks-per-node=1
#SBATCH -A ASC25023
#SBATCH --output=_slurm_out/%x_%A_%a.out
#
# Usage:
#   sbatch -N 4 --array=0-1 --job-name=dbert train_slurm.sh
#   sbatch -N 4 --array=0   --job-name=dmlm train_slurm.sh   # MLM only
#   sbatch -N 4 --array=1   --job-name=dbert_dlm train_slurm.sh   # DLM only

set -e

# --- Two methods ---
METHODS=("mlm" "dlm")

# --- Auto-submit as job array if not already inside one ---
if [ -z "$SLURM_ARRAY_TASK_ID" ] && [ -z "$SLURM_JOB_ID" ]; then
    echo "Auto-submitting as job array with ${#METHODS[@]} tasks (0-$((${#METHODS[@]} - 1)))"
    exec sbatch --array=0-$((${#METHODS[@]} - 1)) "$@" "$0"
fi

# --- Environment ---
mkdir -p "_slurm_out"
source ~/.bashrc

if conda info --envs | awk '{print $1}' | grep -w -q "adlmc"; then
    conda activate adlmc
else
    echo "CRITICAL: No 'adlmc' environment found."
    exit 1
fi

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export CUDA_HOME=$CONDA_PREFIX
export PATH=$CUDA_HOME/bin:$PATH
export SCRATCH=${SCRATCH:-/ssd1/ayx98}

export TRANSFORMERS_VERBOSITY=error
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export HF_HUB_DISABLE_PROGRESS_BARS=1

export MASTER_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$MASTER_NODE
export MASTER_PORT=29500

export NCCL_SOCKET_IFNAME="^lo,docker0"
export GLOO_SOCKET_IFNAME="^lo,docker0"
export NCCL_P2P_LEVEL=NVL
export NCCL_TIMEOUT=1800
export NCCL_IB_TIMEOUT=22
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_SHM_DISABLE=0
export NCCL_BUFFSIZE=16777216
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

ulimit -l unlimited
ulimit -s unlimited

# --- Cache ---
SHARED_CACHE="$SCRATCH/cache/huggingface"
LOCAL_NVME="$TMPDIR/${USER}_${SLURM_JOB_ID}"
export HF_HOME="$SHARED_CACHE"
export TRITON_CACHE_DIR="$LOCAL_NVME/triton_cache"
export HF_DATASETS_CACHE="$LOCAL_NVME/datasets_cache"

setup_node_storage() {
    mkdir -p "$TRITON_CACHE_DIR" "$HF_DATASETS_CACHE"
}
export -f setup_node_storage
export LOCAL_NVME TRITON_CACHE_DIR HF_DATASETS_CACHE
srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 bash -c setup_node_storage

# --- Validate array task ---
if [ $SLURM_ARRAY_TASK_ID -ge ${#METHODS[@]} ]; then
    echo "Error: SLURM_ARRAY_TASK_ID ($SLURM_ARRAY_TASK_ID) out of range (${#METHODS[@]} methods)"
    exit 1
fi

# --- Parse selected method ---
METHOD="${METHODS[$SLURM_ARRAY_TASK_ID]}"

# --- Training config (overridable via env) ---
MAX_STEPS="${MAX_STEPS:-200000}"
BATCH_SIZE="${BATCH_SIZE:-16}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
LEARNING_RATE="${LR:-1e-4}"
SEED="${SEED:-42}"
MAX_LENGTH="${MAX_LENGTH:-512}"
SAVE_STEPS="${SAVE_STEPS:-10000}"

if [ -n "$SLURM_ARRAY_JOB_ID" ]; then
    JOB_TAG="job_${SLURM_ARRAY_JOB_ID}"
else
    JOB_TAG="$(date +%Y%m%d_%H%M%S)"
fi
OUTPUT_ROOT="${OUTPUT_ROOT:-$SCRATCH/cache/dBERT}"
OUTPUT_DIR="${OUTPUT_ROOT}/${JOB_TAG}/${METHOD}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SCRIPT_PATH="${SCRIPT_DIR}/train.py"

echo "========================================"
echo "dBERT Training (task $SLURM_ARRAY_TASK_ID / $((${#METHODS[@]} - 1)))"
echo "  method=$METHOD"
echo "  steps=$MAX_STEPS, batch=${BATCH_SIZE}x${GRAD_ACCUM}x${SLURM_NNODES}N"
echo "  Output: $OUTPUT_DIR"
echo "========================================"

mkdir -p "$OUTPUT_DIR"

CMD="torchrun \
    --nproc_per_node=gpu \
    --nnodes=$SLURM_NNODES \
    --node_rank=\$SLURM_PROCID \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    $SCRIPT_PATH \
    --method $METHOD \
    --output_dir \"$OUTPUT_DIR\" \
    --max_steps $MAX_STEPS \
    --batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --learning_rate $LEARNING_RATE \
    --max_length $MAX_LENGTH \
    --save_steps $SAVE_STEPS \
    --seed $SEED"

START_TIME=$(date +%s)

srun \
    --nodes=$SLURM_NNODES \
    --ntasks=$SLURM_NNODES \
    --ntasks-per-node=1 \
    bash -c "stdbuf -oL -eL $CMD" 2>&1 | tee "$OUTPUT_DIR/slurm.log"

RESULT=$?
END_TIME=$(date +%s)
RUNTIME=$((END_TIME - START_TIME))

echo "========================================"
if [ $RESULT -eq 0 ]; then
    echo "DONE: $METHOD ($((RUNTIME/3600))h $((RUNTIME%3600/60))m)"
else
    echo "FAILED: $METHOD"
fi
echo "========================================"

srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 bash -c "rm -rf $LOCAL_NVME"

exit $RESULT
