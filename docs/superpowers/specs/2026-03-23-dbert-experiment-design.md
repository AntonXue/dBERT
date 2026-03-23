# dBERT: Modern DLM Training on a 2018 Architecture

## Thesis

Modern diffusion language models are essentially masked language modeling — what BERT was doing in 2018. The key improvements since then are the training objective and generation method. This experiment demonstrates that applying modern DLM training techniques to the original BERT architecture yields meaningfully better results than the original BERT training regime.

## Experiment Overview

Train two models with identical architecture and data, differing only in training objective:
1. **Classic BERT MLM** — 15% fixed masking with 80/10/10 corruption
2. **Modern DLM** — Absorbing-state diffusion with uniform noise schedule

Evaluate with a three-way inference comparison that disentangles the contributions of training objective vs generation method.

## Architecture

- **Model:** `BertForMaskedLM` from HuggingFace (`google-bert/bert-base-uncased` config)
- **Initialization:** Fresh random weights via `BertForMaskedLM(BertConfig())` — no pretrained weights
- **Tokenizer:** `BertTokenizer.from_pretrained("google-bert/bert-base-uncased")` — provides `[MASK]` token (id 103)
- **Parameters:** ~110M (12 layers, 768 hidden, 12 heads, vocab 30522)
- **Attention:** Bidirectional (native BERT behavior, no modifications needed)
- **Max sequence length:** 512 (BERT default)

No custom model code required. `BertForMaskedLM` supports both training methods out of the box:
- Classic MLM: pass `labels` with `-100` at unmasked positions, use `output.loss`
- DLM: use `output.logits` directly, compute custom loss on masked positions

## Dataset

- **BookCorpusOpen** (`bookcorpusopen` on HuggingFace) — open reproduction of the original BookCorpus (~17K books)
- **English Wikipedia** (`wikimedia/wikipedia`, English subset) — matches original BERT data
- **Loading:** HuggingFace datasets, streaming mode
- **Held-out split:** Reserve a portion for evaluation (pseudo-perplexity, accuracy-vs-mask-rate)
- **Collation:** Tokenize, truncate/pad to max_length, return `input_ids` tensors

Both methods use identical data pipeline (shared `data.py`).

## Training Method A: Classic BERT MLM

The original 2018 BERT masked language modeling objective:

1. Select 15% of tokens uniformly at random as prediction targets
2. For each selected token:
   - 80% chance: replace with `[MASK]`
   - 10% chance: replace with a random vocabulary token
   - 10% chance: keep the original token
3. Forward pass through bidirectional transformer
4. Cross-entropy loss on the 15% selected positions only (unmasked positions labeled `-100`)

Uses `BertForMaskedLM`'s built-in loss computation via the `labels` argument.

## Training Method B: Modern DLM (Absorbing-State Diffusion)

Adapted from the ADLMC reference implementation (`ablations/qwen2_vs_qwen3/train.py`):

1. Sample `t ~ Uniform(0, 1)` per batch
2. For each token, independently mask with probability `t` (replace with `[MASK]`)
3. Forward pass through bidirectional transformer
4. Cross-entropy loss on masked positions only
5. Time weight = 1 (uniform weighting — no cosine schedule)

Key differences from classic BERT MLM:
- **Variable mask rate** (0-100%) vs fixed 15% — model sees the full spectrum from nearly clean to nearly destroyed
- **Always `[MASK]` replacement** (absorbing state) vs 80/10/10 corruption scheme
- **Uniform noise schedule** — no cosine time weighting, which empirically hurts inference at high-noise settings

Loss formula: `loss = (CE(logits, targets) * is_masked).sum(dim=1).mean() / L`

## Training Configuration

Shared across both methods (mirroring ADLMC reference):

| Parameter | Value |
|---|---|
| Optimizer | AdamW |
| Beta1, Beta2 | 0.9, 0.95 |
| Learning rate | 1e-4 |
| LR schedule | constant with warmup |
| Warmup ratio | 0.01 |
| Weight decay | 0.01 |
| Max grad norm | 1.0 |
| Precision | bf16 |
| Batch size | configurable (default 16 per device) |
| Gradient accumulation | configurable (default 4) |
| Dataloader workers | 4 |

`train.py` accepts `--method bert_mlm|dlm` to switch between objectives. Everything else is identical.

## Evaluation Design

### Primary Metric: Accuracy vs Mask Rate

Evaluate on held-out data with varying mask rates (e.g., 5%, 15%, 30%, 50%, 70%, 90%).

Three-way comparison:

| Method | Model | Inference | Tests |
|---|---|---|---|
| 1. BERT + independent | BERT MLM trained | Single forward pass, predict all masks independently | Classic BERT usage |
| 2. BERT + iterative | BERT MLM trained | Iterative unmasking loop (DLM generation) | Generation method alone |
| 3. DLM + iterative | DLM trained | Iterative unmasking loop (DLM generation) | Training objective + generation method |

This disentangles:
- **Generation method contribution:** Compare 1 vs 2 (same model, different inference)
- **Training objective contribution:** Compare 2 vs 3 (same inference, different training)
- **Combined improvement:** Compare 1 vs 3

Expected result: Method 1 degrades sharply beyond 15% mask rate (OOD for BERT). Methods 2 and 3 degrade gracefully, with 3 outperforming 2.

### Secondary: Pseudo-perplexity

Mask each token one at a time, measure prediction accuracy/NLL across held-out corpus. Comparable for both models.

### Qualitative: Generation Demo (DLM only)

Iterative unmasking from all-`[MASK]` to coherent text. Demonstrates a capability BERT fundamentally lacks. Adapted from ADLMC `ablations/**/eval.py` generation loop, using BERT's `[MASK]` (id 103) as the absorbing state.

## Project Structure

```
dBERT/
├── train.py              # Main training (--method bert_mlm|dlm)
├── train_slurm.sh        # Multi-node SLURM launcher
├── train_local.sh        # Local multi-GPU via HF accelerate
├── eval.py               # Accuracy-vs-mask-rate + pseudo-perplexity
├── data.py               # BookCorpusOpen + Wikipedia loading & collators
├── _saved_models/        # Checkpoints (manageable size for 110M model)
├── _eval_results/        # Evaluation outputs (JSON/CSV)
└── notebooks/            # Training curve plots, result visualization
```

### Priority

1. **P0:** `train.py`, `data.py`, `train_local.sh`, `train_slurm.sh` — get training running
2. **P1:** `eval.py` — accuracy-vs-mask-rate three-way comparison
3. **P2:** `notebooks/` — visualization and analysis
