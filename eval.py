"""dBERT evaluation: accuracy vs mask rate with three inference modes.

Assumes data has been preprocessed via `python data.py`.

Modes:
  1. independent: Single forward pass, predict all masks at once (classic BERT usage)
  2. iterative:   Iterative unmasking loop (DLM generation method)

Three-way comparison:
  - BERT model + independent
  - BERT model + iterative
  - DLM model  + iterative

Usage:
  python eval.py --checkpoint _saved_models/mlm/final --mode independent
  python eval.py --checkpoint _saved_models/dlm/final --mode iterative
"""

import argparse
import json
import os

import torch
import torch.nn.functional as F
from transformers import BertForMaskedLM

from data import load_data, make_collator

MASK_TOKEN_ID = 103


@torch.no_grad()
def evaluate_independent(model, input_ids, mask_rate):
    """Mask tokens at given rate, predict all in one forward pass."""
    B, L = input_ids.shape
    device = input_ids.device

    is_masked = torch.rand(B, L, device=device) < mask_rate
    masked_ids = torch.where(is_masked, MASK_TOKEN_ID, input_ids)

    logits = model(input_ids=masked_ids).logits
    preds = logits.argmax(dim=-1)

    correct = ((preds == input_ids) & is_masked).sum().item()
    total = is_masked.sum().item()
    return correct, total


@torch.no_grad()
def evaluate_iterative(model, input_ids, mask_rate, num_steps=64, temperature=0.0):
    """Mask tokens at given rate, recover via iterative unmasking."""
    B, L = input_ids.shape
    device = input_ids.device

    is_target = torch.rand(B, L, device=device) < mask_rate
    x = torch.where(is_target, MASK_TOKEN_ID, input_ids)

    for step in range(num_steps):
        is_masked = (x == MASK_TOKEN_ID) & is_target
        if not is_masked.any():
            break

        logits = model(input_ids=x).logits

        frac = 1.0 / (num_steps - step)
        unmask = is_masked & (torch.rand(B, L, device=device) < frac)

        if step == num_steps - 1:
            unmask = is_masked

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
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_steps", type=int, default=64, help="Steps for iterative mode")
    p.add_argument("--output_dir", type=str, default="_eval_results")
    args = p.parse_args()

    mask_rates = [float(r) for r in args.mask_rates.split(",")]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BertForMaskedLM.from_pretrained(args.checkpoint).to(device).eval()

    _, eval_data = load_data()
    collator = make_collator()

    results = {}
    for mask_rate in mask_rates:
        total_correct = 0
        total_masked = 0

        for i in range(0, len(eval_data), args.batch_size):
            batch = collator(eval_data[i:i + args.batch_size])
            input_ids = batch["input_ids"].to(device)

            if args.mode == "independent":
                c, t = evaluate_independent(model, input_ids, mask_rate)
            else:
                c, t = evaluate_iterative(model, input_ids, mask_rate,
                                          num_steps=args.num_steps)
            total_correct += c
            total_masked += t

        acc = total_correct / max(total_masked, 1)
        results[str(mask_rate)] = {"accuracy": acc, "correct": total_correct, "total": total_masked}
        print(f"  mask_rate={mask_rate:.2f}  acc={acc:.4f}  ({total_correct}/{total_masked})")

    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_name = os.path.basename(os.path.dirname(args.checkpoint))
    out_path = os.path.join(args.output_dir, f"{ckpt_name}_{args.mode}.json")
    with open(out_path, "w") as f:
        json.dump({"checkpoint": args.checkpoint, "mode": args.mode, "results": results}, f, indent=2)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
