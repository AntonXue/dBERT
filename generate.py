"""dBERT evaluation: mask + fill, save reconstructions to _outputs/.

Assumes data has been preprocessed via `python data.py`.

Modes:
  1. independent: Single forward pass, predict all masks at once (classic BERT usage)
  2. iterative:   Iterative unmasking loop (DLM-style cosine schedule + remasking)

Usage:
  python eval.py --model AntonXue/BERT-MLM --mode independent
  python eval.py --model AntonXue/BERT-DLM --mode iterative
  python eval.py --model AntonXue/BERT-MLM --mode iterative

Outputs saved to _outputs/<model_name>_<mode>.pt containing:
  - original_ids: (N, L) original token IDs
  - reconstructed_ids: dict of mask_rate -> (N, L) reconstructed token IDs
  - mask_flags: dict of mask_rate -> (N, L) bool mask of which positions were masked
  - args: the eval arguments
"""

import argparse
import json
import math
import os

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import BertForMaskedLM

from data import load_data

MASK_TOKEN_ID = 103


def _sample_tokens(logits, temperature, top_k=None):
    """Sample tokens from logits, never sampling the MASK token."""
    B, L, V = logits.shape
    logits = logits.clone()
    logits[:, :, MASK_TOKEN_ID] = -float("inf")

    if temperature < 1e-6:
        return logits.argmax(dim=-1)
    if abs(temperature - 1.0) > 1e-6:
        logits = logits / temperature

    if top_k and 0 < top_k < V:
        logits_flat = logits.view(B * L, V)
        topk_vals, topk_idx = logits_flat.topk(top_k, dim=-1)
        sampled_k = torch.multinomial(F.softmax(topk_vals, dim=-1), 1).view(B * L)
        return topk_idx[torch.arange(B * L, device=logits.device), sampled_k].view(B, L)
    else:
        probs = F.softmax(logits, dim=-1).view(B * L, V)
        return torch.multinomial(probs, 1).view(B, L)


@torch.no_grad()
def fill_independent(model, input_ids, mask_rate):
    """Mask tokens at given rate, fill all in one forward pass (argmax)."""
    B, L = input_ids.shape
    device = input_ids.device

    is_masked = torch.rand(B, L, device=device) < mask_rate
    masked_ids = torch.where(is_masked, MASK_TOKEN_ID, input_ids)

    logits = model(input_ids=masked_ids).logits
    preds = logits.argmax(dim=-1)

    reconstructed = torch.where(is_masked, preds, input_ids)
    return reconstructed, is_masked


@torch.no_grad()
def fill_iterative(model, input_ids, mask_rate, num_steps=64,
                   temperature=0.8, top_k=1000, remask_rate=0.1, eps=1e-2):
    """Mask tokens at given rate, recover via DLM-style iterative unmasking.

    Uses the cosine schedule from alan-dlm/generation_utils.py:
      alpha[0] ~ 0 (clean), alpha[T] = 1 (noisy)
      Denoising walks t from T down to 1, transitioning alpha_t -> alpha_s
      where s = t-1 < t (noisier -> cleaner).
    """
    B, L = input_ids.shape
    device = input_ids.device

    is_target = torch.rand(B, L, device=device) < mask_rate
    is_prompt = ~is_target
    x = torch.where(is_target, MASK_TOKEN_ID, input_ids)

    # Cosine schedule: alpha[0] ~ 0 (clean), alpha[T] = 1 (noisy)
    ticks = torch.linspace(0.0, (math.pi / 2.0) - eps, num_steps + 1, device=device)
    alpha = torch.cos(ticks) ** 2
    alpha = alpha.flip(0)

    # Walk t from T down to 1
    for t in reversed(range(1, num_steps + 1)):
        is_masked = (x == MASK_TOKEN_ID)
        if not is_masked.any():
            break

        logits = model(input_ids=x).logits
        alpha_t = alpha[t]      # current (noisier)
        alpha_s = alpha[t - 1]  # target  (cleaner)

        if t > 1:
            p_unmask = (alpha_t - alpha_s) / (alpha_t + eps)
            to_unmask = torch.rand_like(is_masked.float()) < p_unmask
        else:
            to_unmask = is_masked  # final step: unmask everything

        sampled = _sample_tokens(logits, temperature, top_k)
        sampled = torch.where(to_unmask, sampled, MASK_TOKEN_ID)
        x = torch.where(is_masked, sampled, x)

        # Remasking: let the model reconsider earlier choices.
        # Remask more when noise is high (alpha_t close to 1), taper off near clean.
        if t > 1 and remask_rate > 0.0:
            p_remask = remask_rate * alpha_t
            to_remask = (torch.rand(B, L, device=device) < p_remask) & ~is_masked & ~is_prompt
            x[to_remask] = MASK_TOKEN_ID

    return x, is_target


def main():
    p = argparse.ArgumentParser(description="dBERT evaluation: mask, fill, dump")
    p.add_argument("--model", type=str, required=True,
                    help="HF model ID (e.g. AntonXue/BERT-DLM) or local path")
    p.add_argument("--mode", type=str, required=True, choices=["independent", "iterative"])
    p.add_argument("--mask_rates", type=str, default="0.05,0.15,0.30,0.50,0.70,0.90")
    p.add_argument("--num_samples", type=int, default=10000)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_steps", type=int, default=64, help="Steps for iterative mode")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=1000)
    p.add_argument("--remask_rate", type=float, default=0.1)
    p.add_argument("--dump_dir", type=str, default="_outputs")
    args = p.parse_args()

    mask_rates = [float(r) for r in args.mask_rates.split(",")]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BertForMaskedLM.from_pretrained(args.model).to(device).eval()
    print(f"Model: {args.model} ({sum(p.numel() for p in model.parameters())/1e6:.1f}M params)")
    print(f"Mode: {args.mode}")

    _, eval_data = load_data()
    if args.num_samples < len(eval_data):
        eval_data = eval_data.select(range(args.num_samples))
    print(f"Eval sequences: {len(eval_data):,}")

    # Collect all original IDs
    all_original = torch.tensor(eval_data[:len(eval_data)]["input_ids"], dtype=torch.long)

    reconstructed_ids = {}
    mask_flags = {}

    for mask_rate in mask_rates:
        all_recon = []
        all_masks = []

        n_batches = (len(eval_data) + args.batch_size - 1) // args.batch_size
        for i in tqdm(range(0, len(eval_data), args.batch_size),
                      total=n_batches, desc=f"mask_rate={mask_rate:.2f}"):
            slice_data = eval_data[i:i + args.batch_size]
            input_ids = torch.tensor(slice_data["input_ids"], dtype=torch.long).to(device)

            if args.mode == "independent":
                recon, masked = fill_independent(model, input_ids, mask_rate)
            else:
                recon, masked = fill_iterative(
                    model, input_ids, mask_rate,
                    num_steps=args.num_steps,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    remask_rate=args.remask_rate,
                )

            all_recon.append(recon.cpu())
            all_masks.append(masked.cpu())

        reconstructed_ids[mask_rate] = torch.cat(all_recon, dim=0)
        mask_flags[mask_rate] = torch.cat(all_masks, dim=0)

        # Quick accuracy summary
        correct = ((reconstructed_ids[mask_rate] == all_original) & mask_flags[mask_rate]).sum().item()
        total = mask_flags[mask_rate].sum().item()
        acc = correct / max(total, 1)
        print(f"  -> acc={acc:.4f}  ({correct}/{total})")

    # Save dump
    os.makedirs(args.dump_dir, exist_ok=True)
    model_name = args.model.replace("/", "_")
    dump_path = os.path.join(args.dump_dir, f"{model_name}_{args.mode}.pt")
    torch.save({
        "original_ids": all_original,
        "reconstructed_ids": reconstructed_ids,
        "mask_flags": mask_flags,
        "args": vars(args),
    }, dump_path)
    print(f"Dump saved to {dump_path}")


if __name__ == "__main__":
    main()
