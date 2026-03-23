# dBERT Training Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement training infrastructure to compare classic BERT MLM vs modern DLM training on the same BERT architecture and dataset.

**Architecture:** Single `BertForMaskedLM` (fresh random weights, 110M params) trained with two methods switchable via `--method bert_mlm|dlm`. Shared data pipeline loads BookCorpusOpen + Wikipedia via HuggingFace streaming. Training scales from single-GPU (accelerate) to multi-node (SLURM + torchrun).

**Tech Stack:** PyTorch, HuggingFace transformers + datasets, HF Accelerate, SLURM/torchrun

**Spec:** `docs/superpowers/specs/2026-03-23-dbert-experiment-design.md`

**Reference implementation:** `~/foo/ADLMC/ablations/qwen2_vs_qwen3/train.py` (DLM training loop), `~/foo/ADLMC/ablations/ablation_utils.py` (data loading, logging), `~/foo/ADLMC/ablations/adlm_arch/train_slurm.sh` (SLURM launcher)

---

## File Structure

```
dBERT/
├── data.py               # Dataset loading (BookCorpusOpen + Wikipedia), collator
├── train.py              # Main training script (--method bert_mlm|dlm)
├── train_local.sh        # Local multi-GPU launcher via HF accelerate
├── train_slurm.sh        # Multi-node SLURM launcher via torchrun
├── eval.py               # Accuracy-vs-mask-rate evaluation (sketch)
├── _saved_models/        # Checkpoints
├── _eval_results/        # Evaluation outputs
└── notebooks/            # Visualization
```

**Responsibilities:**
- `data.py`: Loads and interleaves BookCorpusOpen + Wikipedia in streaming mode. Exports a collator function that tokenizes and pads to `max_length`. Also provides a held-out eval iterator for `eval.py`.
- `train.py`: Creates fresh `BertForMaskedLM`, sets up HF `Trainer` with a custom `compute_loss` that switches between BERT MLM and DLM objectives based on `--method`. Logs loss/accuracy to JSONL. Saves checkpoints.
- `train_local.sh`: Wraps `accelerate launch train.py ...` for local multi-GPU.
- `train_slurm.sh`: SLURM sbatch script using `srun` + `torchrun` for multi-node training.
- `eval.py`: Loads a checkpoint, runs accuracy-vs-mask-rate evaluation with three inference modes. Sketch implementation — will be refined later.

---

## Chunk 1: Data Pipeline

### Task 1: Create `data.py` — dataset loading and collation

**Files:**
- Create: `data.py`

- [ ] **Step 1: Write `data.py` with dataset loading and collator**

```python
"""Dataset loading for dBERT: BookCorpusOpen + English Wikipedia."""

from datasets import load_dataset, interleave_datasets
from transformers import BertTokenizer

TOKENIZER_NAME = "google-bert/bert-base-uncased"


def load_tokenizer():
    """Load BERT tokenizer. Provides [MASK] at id 103, [PAD] at id 0."""
    return BertTokenizer.from_pretrained(TOKENIZER_NAME)


def load_train_data():
    """Load BookCorpusOpen + Wikipedia interleaved, streaming mode.

    BookCorpusOpen: ~74M sentences from ~17K books.
    Wikipedia: English Wikipedia articles (20231101 snapshot).
    Both have a 'text' column.
    """
    bookcorpus = load_dataset(
        "bookcorpusopen",
        split="train",
        streaming=True,
        trust_remote_code=True,
    )
    wikipedia = load_dataset(
        "wikimedia/wikipedia",
        "20231101.en",
        split="train",
        streaming=True,
        trust_remote_code=True,
    )
    # Keep only 'text' column from both
    bookcorpus = bookcorpus.select_columns(["text"])
    wikipedia = wikipedia.select_columns(["text"])

    # Interleave with equal probability (roughly matches original BERT mix)
    dataset = interleave_datasets(
        [bookcorpus, wikipedia],
        probabilities=[0.5, 0.5],
        seed=42,
    )
    return dataset


def load_eval_data(num_samples=10000):
    """Load a small eval set from Wikipedia for held-out evaluation.

    Uses a deterministic skip to get samples not seen early in training.
    """
    wiki = load_dataset(
        "wikimedia/wikipedia",
        "20231101.en",
        split="train",
        streaming=True,
        trust_remote_code=True,
    )
    wiki = wiki.select_columns(["text"])
    # Skip ahead to get held-out-ish samples
    eval_samples = []
    for i, ex in enumerate(wiki.skip(1_000_000)):
        if i >= num_samples:
            break
        eval_samples.append(ex)
    return eval_samples


def make_collator(tokenizer, max_length=512):
    """Collator that tokenizes and pads text to fixed length.

    Returns dict with 'input_ids' tensor of shape (B, max_length).
    """
    def collate_fn(batch):
        texts = [ex["text"] for ex in batch]
        encodings = tokenizer(
            texts,
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {"input_ids": encodings["input_ids"]}
    return collate_fn
```

- [ ] **Step 2: Verify data loading works**

Run: `python -c "from data import load_tokenizer, load_train_data, make_collator; tok = load_tokenizer(); ds = load_train_data(); c = make_collator(tok); batch = c([next(iter(ds)) for _ in range(4)]); print(batch['input_ids'].shape); print('MASK id:', tok.mask_token_id, 'PAD id:', tok.pad_token_id)"`

Expected: `torch.Size([4, 512])`, `MASK id: 103`, `PAD id: 0`

- [ ] **Step 3: Commit**

```bash
git add data.py
git commit -m "feat: add data pipeline for BookCorpusOpen + Wikipedia"
```

---

## Chunk 2: Training Script

### Task 2: Create `train.py` — BERT MLM training method

**Files:**
- Create: `train.py`

- [ ] **Step 1: Write `train.py` with BERT MLM trainer**

This is the core training script. Start with the BERT MLM method, since it's simpler (uses built-in loss). The DLM method will be added in Task 3.

```python
"""dBERT: Compare classic BERT MLM vs modern DLM training.

Usage:
  python train.py --method bert_mlm --output_dir _saved_models/bert_mlm
  python train.py --method dlm --output_dir _saved_models/dlm
"""

import argparse
import json
import os

import torch
import torch.nn.functional as F
from transformers import (
    BertConfig,
    BertForMaskedLM,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)
from pathlib import Path

from data import load_tokenizer, load_train_data, make_collator


def is_main_process():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank() == 0
    return int(os.environ.get("RANK", "0")) == 0


class JSONLLogger(TrainerCallback):
    def __init__(self, output_dir):
        self.path = Path(output_dir) / "training_log.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and state.is_local_process_zero:
            with open(self.path, "a") as f:
                f.write(json.dumps({"step": state.global_step, **logs}) + "\n")


class MLMTrainer(Trainer):
    """Classic BERT MLM: 15% masking with 80/10/10 corruption."""

    def __init__(self, mask_token_id, vocab_size, **kwargs):
        super().__init__(**kwargs)
        self.mask_token_id = mask_token_id
        self.vocab_size = vocab_size
        self._steps = 0
        self._loss_sum = 0.0
        self._acc_sum = 0.0

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        input_ids = inputs["input_ids"]
        B, L = input_ids.shape
        device = input_ids.device

        # Select 15% of tokens as targets
        rand = torch.rand(B, L, device=device)
        target_mask = rand < 0.15
        # Don't mask special tokens ([CLS]=101, [SEP]=102, [PAD]=0)
        special = (input_ids == 0) | (input_ids == 101) | (input_ids == 102)
        target_mask = target_mask & ~special

        # 80/10/10 corruption
        corruption_rand = torch.rand(B, L, device=device)
        # 80%: replace with [MASK]
        mask_replace = target_mask & (corruption_rand < 0.8)
        # 10%: replace with random token
        random_replace = target_mask & (corruption_rand >= 0.8) & (corruption_rand < 0.9)
        # 10%: keep original (no replacement needed)

        corrupted_ids = input_ids.clone()
        corrupted_ids[mask_replace] = self.mask_token_id
        corrupted_ids[random_replace] = torch.randint(
            0, self.vocab_size, (random_replace.sum(),), device=device
        )

        # Labels: -100 for non-target positions (ignored by CE loss)
        labels = input_ids.clone()
        labels[~target_mask] = -100

        outputs = model(input_ids=corrupted_ids, labels=labels)
        loss = outputs.loss

        with torch.no_grad():
            preds = outputs.logits.argmax(dim=-1)
            acc_denom = target_mask.sum().clamp(min=1)
            acc = ((preds == input_ids) & target_mask).sum().float() / acc_denom

            self._loss_sum += loss.item()
            self._acc_sum += acc.item()
            self._steps += 1

        return (loss, outputs) if return_outputs else loss

    def log(self, logs, *args, **kwargs):
        if logs is not None and self._steps > 0:
            logs["mlm_loss"] = f"{self._loss_sum / self._steps:.3e}"
            logs["mlm_acc"] = f"{self._acc_sum / self._steps:.3e}"
            self._loss_sum = self._acc_sum = 0.0
            self._steps = 0
        if logs is not None:
            for key in ("learning_rate", "grad_norm"):
                if key in logs and isinstance(logs[key], float):
                    logs[key] = f"{logs[key]:.3e}"
        super().log(logs, *args, **kwargs)


class DLMTrainer(Trainer):
    """Absorbing-state diffusion LM: uniform schedule, variable mask rate."""

    def __init__(self, mask_token_id, pad_token_id, **kwargs):
        super().__init__(**kwargs)
        self.mask_token_id = mask_token_id
        self.pad_token_id = pad_token_id
        self._steps = 0
        self._loss_sum = 0.0
        self._acc_sum = 0.0

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        input_ids = inputs["input_ids"]
        B, L = input_ids.shape
        device = input_ids.device

        # Uniform schedule: t ~ U(0,1), mask_prob = t
        t = torch.rand(B, 1, device=device)
        is_masked = torch.rand(B, L, device=device) < t
        noisy_ids = torch.where(is_masked, self.mask_token_id, input_ids)

        logits = model(input_ids=noisy_ids).logits  # (B, L, V)
        V = logits.size(-1)

        loss_full = F.cross_entropy(
            logits.view(B * L, V), input_ids.view(B * L), reduction="none"
        ).view(B, L)

        # Uniform schedule => time_weight = 1
        loss = (loss_full * is_masked).sum(dim=1).mean() / L

        with torch.no_grad():
            acc_mask = is_masked & (input_ids != self.pad_token_id)
            acc_denom = acc_mask.sum().clamp(min=1)
            preds = logits.argmax(dim=-1)
            acc = ((preds == input_ids) & acc_mask).sum().float() / acc_denom

            self._loss_sum += loss.item()
            self._acc_sum += acc.item()
            self._steps += 1

        return (loss, logits) if return_outputs else loss

    def log(self, logs, *args, **kwargs):
        if logs is not None and self._steps > 0:
            logs["dlm_loss"] = f"{self._loss_sum / self._steps:.3e}"
            logs["dlm_acc"] = f"{self._acc_sum / self._steps:.3e}"
            self._loss_sum = self._acc_sum = 0.0
            self._steps = 0
        if logs is not None:
            for key in ("learning_rate", "grad_norm"):
                if key in logs and isinstance(logs[key], float):
                    logs[key] = f"{logs[key]:.3e}"
        super().log(logs, *args, **kwargs)


def main():
    p = argparse.ArgumentParser(description="dBERT: BERT MLM vs DLM training")
    p.add_argument("--method", type=str, required=True, choices=["bert_mlm", "dlm"])
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--max_steps", type=int, default=100000)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--warmup_ratio", type=float, default=0.01)
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--save_steps", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)

    # Fresh BERT model — random init, no pretrained weights
    config = BertConfig()
    model = BertForMaskedLM(config)
    tokenizer = load_tokenizer()

    if is_main_process():
        n = sum(p.numel() for p in model.parameters())
        print(f"Method: {args.method}")
        print(f"Model: BertForMaskedLM (fresh init, {n/1e6:.1f}M params)")
        print(f"MASK token: {tokenizer.mask_token_id} ({tokenizer.mask_token!r})")
        print(f"PAD token: {tokenizer.pad_token_id} ({tokenizer.pad_token!r})")

    dataset = load_train_data()
    collator = make_collator(tokenizer, args.max_length)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="constant_with_warmup",
        warmup_ratio=args.warmup_ratio,
        weight_decay=0.01,
        max_grad_norm=1.0,
        max_steps=args.max_steps,
        optim="adamw_torch",
        adam_beta1=0.9,
        adam_beta2=0.95,
        bf16=True,
        tf32=True,
        logging_steps=args.logging_steps,
        logging_first_step=True,
        save_steps=args.save_steps,
        save_total_limit=3,
        remove_unused_columns=False,
        dataloader_num_workers=4,
        ignore_data_skip=True,
        report_to="none",
        seed=args.seed,
    )

    if args.method == "bert_mlm":
        trainer = MLMTrainer(
            mask_token_id=tokenizer.mask_token_id,
            vocab_size=config.vocab_size,
            model=model,
            args=training_args,
            train_dataset=dataset,
            data_collator=collator,
            callbacks=[JSONLLogger(args.output_dir)] if is_main_process() else [],
        )
    else:
        trainer = DLMTrainer(
            mask_token_id=tokenizer.mask_token_id,
            pad_token_id=tokenizer.pad_token_id,
            model=model,
            args=training_args,
            train_dataset=dataset,
            data_collator=collator,
            callbacks=[JSONLLogger(args.output_dir)] if is_main_process() else [],
        )

    os.makedirs(args.output_dir, exist_ok=True)
    if is_main_process():
        with open(os.path.join(args.output_dir, "args.json"), "w") as f:
            json.dump(vars(args), f, indent=2)

    trainer.train()

    # Save final model
    if is_main_process():
        model.save_pretrained(os.path.join(args.output_dir, "final"))
        tokenizer.save_pretrained(os.path.join(args.output_dir, "final"))
        print(f"Done. Logs at {args.output_dir}/training_log.jsonl")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke test BERT MLM training (CPU, few steps)**

Run: `python train.py --method bert_mlm --output_dir /tmp/dbert_test_mlm --max_steps 5 --batch_size 2 --gradient_accumulation_steps 1 --logging_steps 1`

Expected: Runs 5 steps, prints loss/accuracy, creates `training_log.jsonl`.

- [ ] **Step 3: Smoke test DLM training (CPU, few steps)**

Run: `python train.py --method dlm --output_dir /tmp/dbert_test_dlm --max_steps 5 --batch_size 2 --gradient_accumulation_steps 1 --logging_steps 1`

Expected: Runs 5 steps, prints loss/accuracy, creates `training_log.jsonl`.

- [ ] **Step 4: Commit**

```bash
git add train.py
git commit -m "feat: add training script with BERT MLM and DLM methods"
```

---

## Chunk 3: Launch Scripts

### Task 3: Create `train_local.sh` — local multi-GPU launcher

**Files:**
- Create: `train_local.sh`

- [ ] **Step 1: Write `train_local.sh`**

```bash
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
```

- [ ] **Step 2: Make executable**

Run: `chmod +x train_local.sh`

- [ ] **Step 3: Commit**

```bash
git add train_local.sh
git commit -m "feat: add local multi-GPU launcher script"
```

### Task 4: Create `train_slurm.sh` — SLURM multi-node launcher

**Files:**
- Create: `train_slurm.sh`

- [ ] **Step 1: Write `train_slurm.sh`**

Adapted from `~/foo/ADLMC/ablations/adlm_arch/train_slurm.sh` — same environment setup patterns, simplified for a single 2-config ablation (bert_mlm vs dlm).

```bash
#!/bin/bash
#SBATCH -p gh
#SBATCH -t 24:00:00
#SBATCH --ntasks-per-node=1
#SBATCH -A ASC25023
#SBATCH --output=_slurm_out/%x_%A_%a.out
#
# Usage:
#   sbatch -N 4 --array=0-1 --job-name=dbert train_slurm.sh
#   sbatch -N 4 --array=0   --job-name=dbert_mlm train_slurm.sh   # MLM only
#   sbatch -N 4 --array=1   --job-name=dbert_dlm train_slurm.sh   # DLM only

set -e

# --- Two methods ---
METHODS=("bert_mlm" "dlm")

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
MAX_STEPS="${MAX_STEPS:-100000}"
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
```

- [ ] **Step 2: Make executable**

Run: `chmod +x train_slurm.sh`

- [ ] **Step 3: Commit**

```bash
git add train_slurm.sh
git commit -m "feat: add SLURM multi-node launcher script"
```

---

## Chunk 4: Evaluation Sketch

### Task 5: Create `eval.py` — accuracy-vs-mask-rate evaluation (sketch)

**Files:**
- Create: `eval.py`

This is a sketch — enough to run, but will be refined later per the spec.

- [ ] **Step 1: Write `eval.py`**

```python
"""dBERT evaluation: accuracy vs mask rate with three inference modes.

Modes:
  1. independent: Single forward pass, predict all masks at once (classic BERT usage)
  2. iterative:   Iterative unmasking loop (DLM generation method)

Three-way comparison:
  - BERT model + independent
  - BERT model + iterative
  - DLM model  + iterative

Usage:
  python eval.py --checkpoint _saved_models/bert_mlm/final --mode independent
  python eval.py --checkpoint _saved_models/dlm/final --mode iterative
"""

import argparse
import json
import os

import torch
import torch.nn.functional as F
from transformers import BertForMaskedLM, BertTokenizer

from data import load_eval_data, make_collator, load_tokenizer


@torch.no_grad()
def evaluate_independent(model, input_ids, mask_rate, mask_token_id):
    """Mask tokens at given rate, predict all in one forward pass."""
    B, L = input_ids.shape
    device = input_ids.device

    # Don't mask special tokens ([CLS]=101, [SEP]=102, [PAD]=0)
    special = (input_ids == 0) | (input_ids == 101) | (input_ids == 102)
    maskable = ~special

    is_masked = (torch.rand(B, L, device=device) < mask_rate) & maskable
    masked_ids = torch.where(is_masked, mask_token_id, input_ids)

    logits = model(input_ids=masked_ids).logits
    preds = logits.argmax(dim=-1)

    correct = ((preds == input_ids) & is_masked).sum().item()
    total = is_masked.sum().item()
    return correct, total


@torch.no_grad()
def evaluate_iterative(model, input_ids, mask_rate, mask_token_id,
                       num_steps=64, temperature=0.0):
    """Mask tokens at given rate, recover via iterative unmasking."""
    B, L = input_ids.shape
    device = input_ids.device

    special = (input_ids == 0) | (input_ids == 101) | (input_ids == 102)
    maskable = ~special

    is_target = (torch.rand(B, L, device=device) < mask_rate) & maskable
    x = torch.where(is_target, mask_token_id, input_ids)

    # Iterative unmasking
    for step in range(num_steps):
        is_masked = (x == mask_token_id) & is_target
        if not is_masked.any():
            break

        logits = model(input_ids=x).logits

        # Fraction to unmask this step
        frac = 1.0 / (num_steps - step)
        unmask = is_masked & (torch.rand(B, L, device=device) < frac)

        if step == num_steps - 1:
            unmask = is_masked  # Unmask all remaining

        if temperature < 1e-6:
            sampled = logits.argmax(dim=-1)
        else:
            probs = F.softmax(logits / temperature, dim=-1)
            sampled = torch.multinomial(probs.view(-1, probs.size(-1)), 1).view(B, L)

        x = torch.where(unmask, sampled, x)

    correct = ((x == input_ids) & is_target).sum().item()
    total = is_target.sum().item()
    return correct, total


def main():
    p = argparse.ArgumentParser(description="dBERT evaluation")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--mode", type=str, required=True, choices=["independent", "iterative"])
    p.add_argument("--mask_rates", type=str, default="0.05,0.15,0.30,0.50,0.70,0.90")
    p.add_argument("--num_eval_samples", type=int, default=1000)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_steps", type=int, default=64, help="Steps for iterative mode")
    p.add_argument("--output_dir", type=str, default="_eval_results")
    args = p.parse_args()

    mask_rates = [float(r) for r in args.mask_rates.split(",")]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BertForMaskedLM.from_pretrained(args.checkpoint).to(device).eval()
    tokenizer = load_tokenizer()

    eval_data = load_eval_data(num_samples=args.num_eval_samples)
    collator = make_collator(tokenizer, max_length=512)

    results = {}
    for mask_rate in mask_rates:
        total_correct = 0
        total_masked = 0

        for i in range(0, len(eval_data), args.batch_size):
            batch = collator(eval_data[i:i + args.batch_size])
            input_ids = batch["input_ids"].to(device)

            if args.mode == "independent":
                c, t = evaluate_independent(model, input_ids, mask_rate, tokenizer.mask_token_id)
            else:
                c, t = evaluate_iterative(model, input_ids, mask_rate, tokenizer.mask_token_id,
                                          num_steps=args.num_steps)
            total_correct += c
            total_masked += t

        acc = total_correct / max(total_masked, 1)
        results[str(mask_rate)] = {"accuracy": acc, "correct": total_correct, "total": total_masked}
        print(f"  mask_rate={mask_rate:.2f}  acc={acc:.4f}  ({total_correct}/{total_masked})")

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_name = os.path.basename(os.path.dirname(args.checkpoint))
    out_path = os.path.join(args.output_dir, f"{ckpt_name}_{args.mode}.json")
    with open(out_path, "w") as f:
        json.dump({"checkpoint": args.checkpoint, "mode": args.mode, "results": results}, f, indent=2)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add eval.py
git commit -m "feat: add evaluation script sketch (accuracy vs mask rate)"
```

---

## Chunk 5: Scaffolding and Final Verification

### Task 6: Create directory scaffolding and gitignore

**Files:**
- Create: `.gitignore`
- Create: `_saved_models/.gitkeep`
- Create: `_eval_results/.gitkeep`
- Create: `notebooks/.gitkeep`

- [ ] **Step 1: Create `.gitignore`**

```
# Checkpoints and model weights
_saved_models/*/
!_saved_models/.gitkeep

# Eval outputs
_eval_results/*/
!_eval_results/.gitkeep

# SLURM logs
_slurm_out/

# Python
__pycache__/
*.pyc
*.egg-info/

# Jupyter
.ipynb_checkpoints/
```

- [ ] **Step 2: Create directory placeholders**

```bash
mkdir -p _saved_models _eval_results notebooks
touch _saved_models/.gitkeep _eval_results/.gitkeep notebooks/.gitkeep
```

- [ ] **Step 3: Commit**

```bash
git add .gitignore _saved_models/.gitkeep _eval_results/.gitkeep notebooks/.gitkeep
git commit -m "feat: add project scaffolding and gitignore"
```

### Task 7: End-to-end smoke test

- [ ] **Step 1: Run BERT MLM training for 5 steps**

Run: `cd /home/ayx98/foo/dBERT && python train.py --method bert_mlm --output_dir /tmp/dbert_smoke_mlm --max_steps 5 --batch_size 2 --gradient_accumulation_steps 1 --logging_steps 1`

Expected: Completes without errors, prints `mlm_loss` and `mlm_acc` for each step.

- [ ] **Step 2: Run DLM training for 5 steps**

Run: `cd /home/ayx98/foo/dBERT && python train.py --method dlm --output_dir /tmp/dbert_smoke_dlm --max_steps 5 --batch_size 2 --gradient_accumulation_steps 1 --logging_steps 1`

Expected: Completes without errors, prints `dlm_loss` and `dlm_acc` for each step.

- [ ] **Step 3: Verify training logs are well-formed**

Run: `cat /tmp/dbert_smoke_mlm/training_log.jsonl | python -m json.tool --no-ensure-ascii > /dev/null && echo "MLM logs OK" && cat /tmp/dbert_smoke_dlm/training_log.jsonl | python -m json.tool --no-ensure-ascii > /dev/null && echo "DLM logs OK"`

Expected: Both print "OK".
