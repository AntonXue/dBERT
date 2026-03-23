"""dBERT: Compare classic BERT MLM vs modern DLM training.

Assumes data has been preprocessed via `python data.py`.

Usage:
  python train.py --method mlm --output_dir _saved_models/mlm
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

from data import load_data, make_collator

# BERT special token IDs (fixed by vocabulary)
MASK_TOKEN_ID = 103
VOCAB_SIZE = 30522


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

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._steps = 0
        self._loss_sum = 0.0
        self._acc_sum = 0.0

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        input_ids = inputs["input_ids"]
        B, L = input_ids.shape
        device = input_ids.device

        # Select 15% of tokens as targets
        target_mask = torch.rand(B, L, device=device) < 0.15

        # 80/10/10 corruption
        corruption_rand = torch.rand(B, L, device=device)
        mask_replace = target_mask & (corruption_rand < 0.8)
        random_replace = target_mask & (corruption_rand >= 0.8) & (corruption_rand < 0.9)

        corrupted_ids = input_ids.clone()
        corrupted_ids[mask_replace] = MASK_TOKEN_ID
        corrupted_ids[random_replace] = torch.randint(
            0, VOCAB_SIZE, (random_replace.sum(),), device=device
        )

        logits = model(input_ids=corrupted_ids).logits
        V = logits.size(-1)

        loss_full = F.cross_entropy(
            logits.view(B * L, V), input_ids.view(B * L), reduction="none"
        ).view(B, L)

        target_count = target_mask.sum().clamp(min=1)
        loss = (loss_full * target_mask).sum() / target_count

        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            acc = ((preds == input_ids) & target_mask).sum().float() / target_count
            self._loss_sum += loss.item()
            self._acc_sum += acc.item()
            self._steps += 1

        return (loss, logits) if return_outputs else loss

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

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
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
        noisy_ids = torch.where(is_masked, MASK_TOKEN_ID, input_ids)

        logits = model(input_ids=noisy_ids).logits
        V = logits.size(-1)

        loss_full = F.cross_entropy(
            logits.view(B * L, V), input_ids.view(B * L), reduction="none"
        ).view(B, L)

        # Uniform schedule => time_weight = 1
        loss = (loss_full * is_masked).sum(dim=1).mean() / L

        with torch.no_grad():
            masked_count = is_masked.sum().clamp(min=1)
            preds = logits.argmax(dim=-1)
            acc = ((preds == input_ids) & is_masked).sum().float() / masked_count
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
    p.add_argument("--method", type=str, required=True, choices=["mlm", "dlm"])
    p.add_argument("--output_dir", type=str, required=True)
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
    model = BertForMaskedLM(BertConfig())

    if is_main_process():
        n = sum(p.numel() for p in model.parameters())
        print(f"Method: {args.method}")
        print(f"Model: BertForMaskedLM (fresh init, {n/1e6:.1f}M params)")

    dataset, _ = load_data()
    collator = make_collator()

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

    TrainerClass = MLMTrainer if args.method == "mlm" else DLMTrainer
    trainer = TrainerClass(
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

    if is_main_process():
        model.save_pretrained(os.path.join(args.output_dir, "final"))
        print(f"Done. Logs at {args.output_dir}/training_log.jsonl")


if __name__ == "__main__":
    main()
