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

### Qualitative Examples (90% masking)

Two examples from each method, showing 512-token sequences with ~90% of tokens masked and reconstructed.

**MLM + independent** -- produces degenerate "the the the of of" repetitions:

> **Original:** the golden globet award for best screenplay ( ) is a prize given to the films in the main category of competition at the shanghai international film festival. award winners references lists of films by award shanghai international film festival hypancistrus contradens is a species of armored catfish endemic to venezuela where it occurs in the orinoco river...

> **Reconstructed:** the film of of.... the it is a of of the. films. the main film.. at the. of of. the award at references the of the the the the of of the hypical of the the the the the the the the the the the the the.. it is the...... oftracista.... of of of............ by by nathan........ by. bae.,,,, 2007...

> **Original:** trophy for men and the taylor trophy for women. commonwealth games organised by the commonwealth games federation. first held in 1930, the commonwealth games bowls championships for men and women are held every 4 years. women's events were added from 1982. four gold medals are competed for in the singles, pairs, triples and fours...

> **Reconstructed:** . for the.. taylor'the. the the the the the the the indoor federation. references links links the the the the...... there are hosted in.......... there. four world. the........... the the world bowls indoor indoor. the the world outdoor indoor bowls.................... the the the the the the in...

**MLM + iterative** -- grammatically coherent but topically drifts:

> **Original:** the golden globet award for best screenplay ( ) is a prize given to the films in the main category of competition at the shanghai international film festival...

> **Reconstructed:** before 2018. references external links official website 2008 establishments in venezuela iguata populated places in the united states populated places in the philippines international latin american alumni award winners national university of venezuela alumni award for latin american alumni h. g. laurmaque y maquilla ( died 1996 ), president of the university council the record is held by artist.

> **Original:** trophy for men and the taylor trophy for women. commonwealth games organised by the commonwealth games federation. first held in 1930...

> **Reconstructed:** championships in the sport are chosen except for the national events, organised by the european games held there only on the part of the european games every year ( but not by a jury every year for these championships ). there were a number of european championships and medals in the sport. the singles, singles, doubles, and singles are decided by the following events in two sports in each world.

**DLM + iterative** -- coherent, topically grounded, reads like real encyclopedia text:

> **Original:** the golden globet award for best screenplay ( ) is a prize given to the films in the main category of competition at the shanghai international film festival...

> **Reconstructed:** was nominated for the award for best screenplay, and is nominated for best original screenplay. for best main film the bride at the 1st toronto international film festival. winners external links the bride at the festival website films set in toronto toronto ambellona annuensis is a species of ray - finn dorid fish in the family cerodinonithidae. cerodinoni fish of africa fish described in 1937...

> **Original:** trophy for men and the taylor trophy for women. commonwealth games organised by the commonwealth games federation. first held in 1930...

> **Reconstructed:** people 1980 births table tennis players from sydney the pacific bowls cup competition is part of the international bowls cup. in the world bowls cup competition, it is part of the world cup series. four gold medals of eight nations were awarded to the tournament. all medals are awarded for each in singles events ; bowls in singles, fours, and doubles. championships one pacific bowls world cup cham...

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
