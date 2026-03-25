"""Score generation dumps with GPT-2 XL perplexity and MAUVE.

Loads .pt dumps from generate.py, decodes reconstructed sequences,
and scores them. Reports perplexity (median/mean) and MAUVE.

Usage:
  python eval.py --dump _outputs/AntonXue_BERT-DLM_iterative.pt
  python eval.py --dump _outputs/*.pt   # Score all dumps
"""

import argparse
import glob
import json
import math
import os

import mauve
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BertTokenizer


def compute_perplexity(gpt2_model, gpt2_tokenizer, texts, device, batch_size=16):
    """Compute per-sequence NLL under GPT-2, return list of NLLs."""
    seq_nlls = []
    for start in tqdm(range(0, len(texts), batch_size), desc="  GPT-2 ppl",
                      total=(len(texts) + batch_size - 1) // batch_size):
        batch_texts = texts[start:start + batch_size]

        gpt2_inputs = gpt2_tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(device)

        input_ids = gpt2_inputs["input_ids"]
        attention_mask = gpt2_inputs["attention_mask"]

        with torch.no_grad():
            outputs = gpt2_model(input_ids=input_ids, attention_mask=attention_mask)

        shift_logits = outputs.logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        shift_mask = attention_mask[:, 1:].contiguous()

        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="none",
        ).view(shift_labels.shape)

        per_seq = (loss * shift_mask).sum(dim=1) / shift_mask.sum(dim=1).clamp(min=1)
        seq_nlls.extend(per_seq.tolist())

    return seq_nlls


def compute_mauve_score(reference_texts, generated_texts, device_id=0):
    """Compute MAUVE score between reference and generated text distributions."""
    result = mauve.compute_mauve(
        p_text=reference_texts,
        q_text=generated_texts,
        max_text_length=512,
        device_id=device_id,
        batch_size=64,
        verbose=False,
    )
    return result.mauve


def main():
    p = argparse.ArgumentParser(description="Score generation dumps with GPT-2 ppl + MAUVE")
    p.add_argument("--dump", type=str, nargs="+", required=True,
                    help="Path(s) to .pt dump files (supports glob)")
    p.add_argument("--gpt2_model", type=str, default="openai-community/gpt2-xl")
    p.add_argument("--gpt2_batch_size", type=int, default=16)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--output_dir", type=str, default="_outputs")
    args = p.parse_args()

    # Expand globs
    dump_paths = []
    for pattern in args.dump:
        dump_paths.extend(glob.glob(pattern))
    dump_paths = sorted(set(dump_paths))
    print(f"Scoring {len(dump_paths)} dump(s)")

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    # Load GPT-2 XL
    print(f"Loading {args.gpt2_model}...")
    gpt2_model = AutoModelForCausalLM.from_pretrained(
        args.gpt2_model, dtype=torch.float16
    ).to(device).eval()
    gpt2_tokenizer = AutoTokenizer.from_pretrained(args.gpt2_model)
    gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token
    print(f"GPT-2: {sum(p.numel() for p in gpt2_model.parameters())/1e6:.0f}M params")

    # Load BERT tokenizer for decoding
    bert_tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased")

    for dump_path in dump_paths:
        print(f"\n{'='*60}")
        print(f"Dump: {dump_path}")
        dump = torch.load(dump_path, weights_only=False)

        dump_args = dump["args"]
        print(f"  Model: {dump_args['model']}, Mode: {dump_args['mode']}")

        original_ids = dump["original_ids"]
        reconstructed_ids = dump["reconstructed_ids"]
        mask_flags = dump["mask_flags"]

        # Decode original texts (shared across mask rates)
        original_texts = bert_tokenizer.batch_decode(original_ids, skip_special_tokens=True)

        # Score original text (baseline perplexity)
        print("Scoring baseline (original text)...")
        orig_nlls = compute_perplexity(
            gpt2_model, gpt2_tokenizer, original_texts, device, args.gpt2_batch_size
        )
        orig_nlls_t = torch.tensor(orig_nlls)
        print(f"  Baseline: median_ppl={math.exp(orig_nlls_t.median()):.2f}  mean_ppl={math.exp(orig_nlls_t.mean()):.2f}")

        results = {
            "model": dump_args["model"],
            "mode": dump_args["mode"],
            "baseline": {
                "median_ppl": math.exp(orig_nlls_t.median().item()),
                "mean_ppl": math.exp(orig_nlls_t.mean().item()),
            },
            "mask_rates": {},
        }

        for mask_rate, recon_ids in reconstructed_ids.items():
            masks = mask_flags[mask_rate]
            correct = ((recon_ids == original_ids) & masks).sum().item()
            total = masks.sum().item()
            acc = correct / max(total, 1)

            recon_texts = bert_tokenizer.batch_decode(recon_ids, skip_special_tokens=True)

            print(f"\nScoring mask_rate={mask_rate:.2f}...")

            # GPT-2 perplexity
            nlls = compute_perplexity(
                gpt2_model, gpt2_tokenizer, recon_texts, device, args.gpt2_batch_size
            )
            nlls_t = torch.tensor(nlls)
            med_ppl = math.exp(nlls_t.median().item())
            mean_ppl = math.exp(nlls_t.mean().item())

            # MAUVE
            print("  Computing MAUVE...")
            mauve_score = compute_mauve_score(original_texts, recon_texts)

            results["mask_rates"][str(mask_rate)] = {
                "accuracy": acc,
                "median_ppl": med_ppl,
                "mean_ppl": mean_ppl,
                "mauve": mauve_score,
            }
            print(f"  acc={acc:.4f}  median_ppl={med_ppl:.2f}  mean_ppl={mean_ppl:.2f}  mauve={mauve_score:.4f}")

        # Save scores
        os.makedirs(args.output_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(dump_path))[0]
        out_path = os.path.join(args.output_dir, f"{base}_scores.json")
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nScores saved to {out_path}")


if __name__ == "__main__":
    main()
