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

Evaluation on 5,000 held-out sequences at varying mask rates. Iterative unmasking uses a cosine schedule with 128 steps, temperature=0.8, top-k=1000, ReMDM remasking with sigma_scale=0.1.

### Token Accuracy

| Mask Rate | MLM + independent | MLM + iterative | DLM + iterative |
|---|---|---|---|
| 5% | **65.8%** | 61.4% | 59.0% |
| 15% | **61.4%** | 56.7% | 54.7% |
| 30% | **53.1%** | 47.9% | 47.2% |
| 50% | **39.0%** | 33.2% | 34.7% |
| 70% | **23.0%** | 16.9% | 20.4% |
| 90% | **8.1%** | 4.3% | 6.5% |

### GPT-2 XL Perplexity (median, lower = more coherent)

| Mask Rate | Baseline | MLM + independent | MLM + iterative | DLM + iterative |
|---|---|---|---|---|
| -- | 25.7 | -- | -- | -- |
| 5% | | **26.2** | 26.8 | 27.2 |
| 15% | | **27.9** | 29.4 | 30.5 |
| 30% | | 34.0 | 34.6 | 36.2 |
| 50% | | 50.0 | 45.7 | **44.2** |
| 70% | | 56.8 | 58.1 | **48.9** |
| 90% | | 36.2* | 66.6 | **40.8** |

*\*MLM independent at 90% has deceptively low perplexity because it produces degenerate repetitive output ("of of of references references references") that scores well token-by-token but is incoherent.*

### MAUVE Score (higher = more like real text)

| Mask Rate | MLM + independent | MLM + iterative | DLM + iterative |
|---|---|---|---|
| 5% | **0.993** | 0.990 | 0.988 |
| 15% | **0.984** | 0.979 | 0.979 |
| 30% | 0.970 | **0.976** | 0.970 |
| 50% | 0.767 | 0.954 | **0.961** |
| 70% | 0.082 | 0.932 | **0.940** |
| 90% | 0.007 | 0.523 | **0.882** |

### Qualitative Example (90% masking)

One 512-token sequence with ~90% of tokens masked, reconstructed by each method.

<details>
<summary><b>Original</b></summary>

> the golden globet award for best screenplay ( ) is a prize given to the films in the main category of competition at the shanghai international film festival. award winners references lists of films by award shanghai international film festival hypancistrus contradens is a species of armored catfish endemic to venezuela where it occurs in the orinoco river. references ancistrini fish of venezuela endemic fauna of venezuela taxa named by jonathan w. armbruster taxa named by nathan keller lujan taxa named by donald charles taphorn baechle fish described in 2007 theophanous v herald & weekly times ltd is a landmark australian judgment of the high court. the matter related to implied freedom of political communication that the high court has inferred, rests in the australian constitution. background andrew theophanous had been an australian labor party member of the australian house of representatives since 1980. in 1992, he was the chairperson of the joint parliamentary standing committee on migration. the herald and weekly times published an article by bruce ruxton, " give theophanous the shove ", which stated that theophanous " appears to want a bias shown towards greeks as migrants ". theophanous sued the herald & weekly times and ruxton for defamation. decision the judgment held that there was an implied constitutional freedom to publish material discussing government and political matters as well as the way that members of the parliament of australia conducted their duties and their suitability for office. significance just three years later, with a change in the composition of the high court, the court unanimously reversed the opinion in lange v australian broadcasting corporation. it held that no direct right to free speech could form a defence to defamation. still, the case remains important in the development of the implied freedom. references further reading. high court of australia cases 1994 in australian law 1994 in case law these are the premier international bowls events between national bowls organisations affiliated to world bowls, the pba, world bowls tour and the iibc. calendar world outdoor bowls championships the sport's premier event and organised by world bowls. first held in 1966, the world outdoor bowls championships for men and women are held every 4 years. from 2008 the men's and women's events are held together. qualifying national bowls organisations ( usually countries ) are represented by a team of 5 players, who play once as a single and a four, then again as a pair and a triple. gold, silver, and bronze medals are awarded in each of the 4 disciplines, and there is also a trophy for the best overall team — the leonard
</details>

<details>
<summary><b>MLM + independent</b> (458 of 512 tokens masked) -- degenerate repetitions of high-frequency tokens</summary>

> of of of references competition for the in of references references references references references references of of of of of the the competition at the university of of references references references references references references references references of of references international references of references references of of of references references gastropods of of of of of of of of of of of of of of of of of of of of of of of of of of external endemic fauna of venezuela fish of of of of of of of of of of of of of of lupha references, of of of tapilo of of references fish described in 2011 cephala,s & co,,, of of of of of of of of of of of of of the court of the, of the high court of the of of of of,,, of, of andrew,,,, of of of labor, of of of of of of of of of of of of of of of the court of the court of of of of of of of of,,,,,,. of buxton,....... " the of of.... the the the the has is shown as be as a.. references. the the of of,,.. rupp,, the.... that there is an implied that of the the. the the..... the the, that there of the part of the - of the the the the the the the..... was later in in the the....... of the court unanimously. the opinion. the....... is a direct. of the the there as a defence of the....................,,,,,,,, 1994. references links links of,,,,,,,,,,,,,,,,,,,, and the references of.,,,,. of of of the premier event of the of of of,,,, references references of of of of of of of of references of of of of.s from the. of of the the the. the there are the in of the national sports ( ( ( ( ). the the the the the the the the a play a the the, and the... again, the, in a triple.. the the the.. medals are the the the the the the disciplines... see also references of of references references references references of of of
</details>

<details>
<summary><b>MLM + iterative</b> (458 of 512 tokens masked) -- grammatically coherent but topically drifts</summary>

> , plait - shrimp the black - keeled bow references frogs of venezuela, chitis, proboscis, the world ripper, f. s. ( snake ) the foot brassachitle and hypococcus algae ( exact ) riparia species c. spina ) chitoyaculata the combinosus frogs of the sur sur ocucunodia endemic fauna of venezuela references s. stephens, s. a. walker, j. philip keller ( al. ) and peterson ( bighorn ) grege fish, c. smith ( 1980 ) rio de janeiro publishers ltd, 1984 external links frogs of the atlantic moths of the electoral districts of canada, chapter i : " pierre cecile hassonferere, p. ". forbes. com, june 1984 references living people missing by action politicians of french origin ( canada ) in the 1980 alberta provincial council was elected by the province of north and the alberta's. herald of the rockies, cfr, canada pp. 31 - 53. the province of north columbia was prone to the use of the " star " referring to alberta, a website that was used in the government by alberta's calgary herald, mcc and first bearers for sport college the university of alberta. newspaper was banned in the 1980s by premier of north columbia and used in the field for concern that the congress of alberta - calgary - calgary was held at the conference center of the province of north columbia which was being held in order to keep the newspaper as the venue. this was unanimously reversed in george v. inwal broadcasting corporation. it was presented no special interest in a speech or property libel in the defamation compensation act. the case was found in the quebec landmark case, george v. inwal. in 2006, issue no. xxv of the association was won its first titles in the 2006 british and scottish bowls championships. the 2016 british bowls championship part of the world tour of the british new zealand bowls association also held in the association's 2006 season is the world outdoor bowls championship event. it is the second major outdoor international championship for organized bowls and bowls competitions every year since. in 2008 the world triple bowls triple - triple bowls championships are a international series of international competitions held in six countries that are hosted by each nation. one exception, the world triple bowls world championships in triple bowls triple - triple bowls and world triple bowls triple - triple the bowls, are both levels of international competitions each year. in 2013, the triple bowls triple championship was the second major international team of the year
</details>

<details>
<summary><b>DLM + iterative</b> (465 of 512 tokens masked) -- coherent, topically grounded, reads like real encyclopedia text</summary>

> the golden nandi award for best overall director. for the film won in the best tamil feature film category for best feature film. films 2010s 2010s 2010s 2010s references external links the film page 2010s tamil language film 2012 films acanthodia percurnela is a species of moth of the family tortricidae. it was described by schiffenee in 1911 and is known from mexico. references endemic fauna of mexico beetles of texas moths described in 1911 moths of mexico taxa named by edward meyrick tribune may refer to : the tribune ( newspaper ), ceased to exist in 2007 the tribune, western australia, weekly newspaper ltd., an australian weekly newspaper the court of the court of the eastern united states, was created on 1 november 1972. on june 5, 1972, speaker of the house g. theos sits as the sitting president of the u. s. house of representatives richard w. hadmon, the president of the u. s. house committee committee, and the associate and presiding judge published an article about the braxton case g. theos authored the merriam case. federal federal courts the court of the united states in 1972 1972 under the name of g. emphal, campbell, & co. in its current government case records, the court decision was written : granting a constitutionality of the court in the united states. in constitutional proceedings, the supreme court has a sitting sitting on the supreme court for 40 years and continues to serve as the court. this decision was held by the current composition of the supreme court of the supreme court since 1978. history of the court dates back to 1965 in s. j. braxton jr. set a record in 1969. still extant, the case remains under legal jurisdiction of the supreme court. references court of the united states 1972 in music ( music ) songs " blue " is an american music video that sees events between two different clubs affiliated to the national american association of associations roman catholic association of the united states. history the video was released on april 25, 2012 on xsc4. the song is held live performances with the fourth big boy. the clip, with three versions of the song upon its release, and was released in july 2016, after wilson was invited to play a solo version of nelly in the song, as a duet. the song became the song's third single in april 2016. performance the song was released as a single in april united states, reaching the billboard hot 100 and number 7 on the hot hot 100 chart. chart certifications in the song, wilson's version
</details>

### Key Takeaways

1. **At high mask rates, DLM training + iterative generation produces dramatically more coherent text.** At 90% masking, DLM maintains MAUVE=0.882 while MLM iterative drops to 0.523 and MLM independent collapses to 0.007.

2. **Token accuracy is misleading.** MLM independent scores highest on per-token accuracy at all mask rates, but at 90% it produces degenerate repetitive output -- high-frequency tokens that are individually likely but collectively incoherent.

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
