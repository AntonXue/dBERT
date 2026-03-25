# dBERT: Modern Diffusion LM Training on a 2018 Architecture

Modern diffusion language models (DLMs) are essentially doing masked language modeling -- what BERT was doing back in 2018. The key improvements since then are the training objective and the generation method. This project demonstrates that applying modern DLM training techniques to the original BERT architecture yields meaningfully better results than the original BERT training regime.

## Experiment

We train two models with **identical architecture and data**, differing only in the training objective:

1. **MLM** -- Classic BERT masked language modeling (15% fixed masking, 80/10/10 corruption)
2. **DLM** -- Absorbing-state diffusion with uniform noise schedule (variable mask rate, always `[MASK]`)

We then evaluate with a three-way inference comparison:

| Method | Model | Inference | What it tests |
|---|---|---|---|
| MLM + independent | MLM trained | Single forward pass, argmax | Classic BERT usage |
| MLM + iterative | MLM trained | DLM-style iterative unmasking | Generation method alone |
| DLM + iterative | DLM trained | DLM-style iterative unmasking | Training objective + generation method |

## Architecture

- **Model:** `BertForMaskedLM` from HuggingFace (`google-bert/bert-base-uncased` config)
- **Initialization:** Fresh random weights -- no pretrained weights
- **Parameters:** 109.5M (12 layers, 768 hidden, 12 heads, vocab 30,522)
- **Sequence length:** 512
- **Attention:** Bidirectional (native BERT)

## Dataset

- **BookCorpusOpen** ([lucadiliello/bookcorpusopen](https://huggingface.co/datasets/lucadiliello/bookcorpusopen)) -- ~17K books
- **English Wikipedia** ([wikimedia/wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia), 20231101.en) -- ~6.4M articles
- **Split:** 95/5 on raw documents, then tokenized and packed into 512-token sequences
- **Train:** 10,784,085 sequences (5.52B tokens)
- **Eval:** 581,637 sequences (298M tokens)

## Training Configuration

| Parameter | Value |
|---|---|
| Optimizer | AdamW |
| Adam betas | (0.9, 0.999) |
| Learning rate | 1e-4 |
| LR schedule | Constant with warmup |
| Warmup steps | 500 |
| Weight decay | 0.01 |
| Max grad norm | 1.0 |
| Precision | bf16 |
| Global batch size | 256 |
| Training steps | 100,000 |
| Tokens seen | ~13.1B |

## Training Objectives

**MLM (Classic BERT):** Select 15% of tokens. Replace 80% with `[MASK]`, 10% with random token, 10% unchanged. Cross-entropy loss on selected positions.

**DLM (Absorbing-State Diffusion):** Sample `t ~ U(0,1)`. Mask each token independently with probability `t` (always `[MASK]`). Cross-entropy loss on masked positions. Uniform time weighting.

## Results

Evaluation on 512 held-out sequences at varying mask rates. Iterative unmasking uses a cosine schedule with 64 steps, temperature=0.8, top-k=1000, remask_rate=0.1.

### Token Accuracy

| Mask Rate | MLM + independent | MLM + iterative | DLM + iterative |
|---|---|---|---|
| 5% | **66.1%** | 61.6% | 59.0% |
| 15% | **61.6%** | 56.8% | 54.5% |
| 30% | **53.0%** | 47.6% | 47.1% |
| 50% | **39.4%** | 32.5% | 33.9% |
| 70% | **22.8%** | 16.3% | 19.7% |
| 90% | **8.6%** | 4.2% | 6.4% |

### GPT-2 XL Perplexity (lower = more coherent)

| Mask Rate | Baseline | MLM + independent | MLM + iterative | DLM + iterative |
|---|---|---|---|---|
| -- | 25.3 | -- | -- | -- |
| 5% | | **25.7** | 26.5 | 27.2 |
| 15% | | **27.3** | 28.6 | 29.9 |
| 30% | | **32.9** | 33.4 | 35.5 |
| 50% | | 46.9 | 45.4 | **43.4** |
| 70% | | 53.9 | 60.0 | **47.1** |
| 90% | | 38.2* | 80.7 | **41.6** |

*\*MLM independent at 90% has deceptively low perplexity because it produces degenerate repetitive output ("the the the of of of") that scores well token-by-token but is incoherent.*

### MAUVE Score (higher = more like real text)

| Mask Rate | MLM + independent | MLM + iterative | DLM + iterative |
|---|---|---|---|
| 5% | 0.993 | 0.992 | 0.993 |
| 15% | 0.991 | 0.986 | 0.988 |
| 30% | 0.980 | 0.972 | 0.951 |
| 50% | 0.864 | **0.986** | 0.960 |
| 70% | 0.136 | 0.963 | **0.969** |
| 90% | 0.009 | 0.724 | **0.941** |

### Key Takeaways

1. **At high mask rates, DLM training + iterative generation produces dramatically more coherent text.** At 90% masking, DLM maintains MAUVE=0.941 while MLM iterative drops to 0.724 and MLM independent collapses to 0.009.

2. **Token accuracy is misleading.** MLM independent scores highest on per-token accuracy at all mask rates, but at 90% it produces "the the the of of of" -- high-frequency tokens that are individually likely but collectively incoherent.

3. **The generation method matters.** At 50%+ masking, MLM + iterative outperforms MLM + independent on MAUVE, showing that the iterative unmasking procedure alone (even with a model not trained for it) produces more coherent output than independent prediction.

4. **The training objective matters more.** At 70-90% masking, DLM + iterative beats MLM + iterative on all metrics (accuracy, perplexity, MAUVE), showing that training with variable mask rates produces a model better suited for iterative generation.

## Models

Pre-trained models available on HuggingFace:

```python
from transformers import AutoModel

model = AutoModel.from_pretrained("AntonXue/BERT-MLM")   # Classic MLM training
model = AutoModel.from_pretrained("AntonXue/BERT-DLM")   # DLM training
```

## Usage

```bash
# 1. Preprocess data (once, ~15 min)
python data.py

# 2. Train
./train_local.sh mlm    # Classic BERT MLM
./train_local.sh dlm    # Modern DLM

# 3. Generate reconstructions
python generate.py --model AntonXue/BERT-MLM --mode independent
python generate.py --model AntonXue/BERT-MLM --mode iterative
python generate.py --model AntonXue/BERT-DLM --mode iterative

# 4. Score with GPT-2 perplexity + MAUVE
python eval.py --dump _outputs/*.pt
```

## Project Structure

```
dBERT/
├── data.py               # Preprocess: download, tokenize, pack, split 95/5
├── train.py              # Train with --method mlm|dlm
├── train_local.sh        # Local multi-GPU launcher (HF Accelerate)
├── train_slurm.sh        # SLURM multi-node launcher (torchrun)
├── generate.py           # Mask + fill, save .pt dumps to _outputs/
├── eval.py               # Score dumps: GPT-2 perplexity + MAUVE
├── _data/                # Preprocessed packed sequences
├── _saved_models/        # Training checkpoints
├── _outputs/             # Generation dumps + score JSONs
└── notebooks/            # Visualization & analysis
```
