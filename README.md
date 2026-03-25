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

### Qualitative Example (90% masking)

One 512-token sequence with ~90% of tokens masked, reconstructed by each method.

<details>
<summary><b>Original</b></summary>

> the golden globet award for best screenplay ( ) is a prize given to the films in the main category of competition at the shanghai international film festival. award winners references lists of films by award shanghai international film festival hypancistrus contradens is a species of armored catfish endemic to venezuela where it occurs in the orinoco river. references ancistrini fish of venezuela endemic fauna of venezuela taxa named by jonathan w. armbruster taxa named by nathan keller lujan taxa named by donald charles taphorn baechle fish described in 2007 theophanous v herald & weekly times ltd is a landmark australian judgment of the high court. the matter related to implied freedom of political communication that the high court has inferred, rests in the australian constitution. background andrew theophanous had been an australian labor party member of the australian house of representatives since 1980. in 1992, he was the chairperson of the joint parliamentary standing committee on migration. the herald and weekly times published an article by bruce ruxton, " give theophanous the shove ", which stated that theophanous " appears to want a bias shown towards greeks as migrants ". theophanous sued the herald & weekly times and ruxton for defamation. decision the judgment held that there was an implied constitutional freedom to publish material discussing government and political matters as well as the way that members of the parliament of australia conducted their duties and their suitability for office. significance just three years later, with a change in the composition of the high court, the court unanimously reversed the opinion in lange v australian broadcasting corporation. it held that no direct right to free speech could form a defence to defamation. still, the case remains important in the development of the implied freedom. references further reading. high court of australia cases 1994 in australian law 1994 in case law these are the premier international bowls events between national bowls organisations affiliated to world bowls, the pba, world bowls tour and the iibc. calendar world outdoor bowls championships the sport's premier event and organised by world bowls. first held in 1966, the world outdoor bowls championships for men and women are held every 4 years. from 2008 the men's and women's events are held together. qualifying national bowls organisations ( usually countries ) are represented by a team of 5 players, who play once as a single and a four, then again as a pair and a triple. gold, silver, and bronze medals are awarded in each of the 4 disciplines, and there is also a trophy for the best overall team — the leonard
</details>

<details>
<summary><b>MLM + independent</b> (446 of 512 tokens masked) -- degenerate "the the the of of" repetitions</summary>

> the film of of.... the it is a of of the. films. the main film.. at the. of of. the award at references the of the the the the of of the hypical of the the the the the the the the the the the the the.. it is the...... oftracista.... of of of............ by by nathan........ by. bae.,,,, 2007.. the the herald & herald,....... the the the the the the of the implied use of the communication. the the of the the the the the the the the the the the the the the the it had been the by the party. in the australian house of representatives. the of of of the. of the australian of the the the the the the the the the the the the. is an article by the............ the. that theophatic is the the the the the the as. as the by by theopha... in the the times in the.,.. in the same of the the the the the the. to publish the the of the political, as well as the way that the the the the the the the the to in their own. for the... three years later, with the. in the., the court of the supreme court court reversed.. in the.... the the the the the the the the the the the the the the the the the.. is of an important in in the the. the the the the the of of the list of the. of of of law of the of of of of the premier international. of the the the of the the the the the the the the the world tour of the the of of the the the the the the the the the the the the the the the the of. women in the. the world. of references the of of of of of of the the the the the the the the the the women'the the the... the. the the the the the there are the by the the the the the the the the the the the, and the four.. the the the the the the the the the in., and. medals are awarded by the.... the the the the the the the the the the the team in the world
</details>

<details>
<summary><b>MLM + iterative</b> (460 of 512 tokens masked) -- grammatically coherent but topically drifts</summary>

> before 2018. references external links official website 2008 establishments in venezuela iguata populated places in the united states populated places in the philippines international latin american alumni award winners national university of venezuela alumni award for latin american alumni h. g. laurmaque y maquilla ( died 1996 ), president of the university council the record is held by artist. references educated margarita margarita margarita 1955 births living people architects from venezuela 1991 births 2021 deaths accidental deaths in caracas a jury selection at the prix peter pojan, hosted by donald charles tapia baech, richard beaumont and guy shephard 1983. reprint 1998. the notes of united states supreme court and supreme court and justice and related pages. from william a. sk, chicago / strangler publishing / publishing corp., ed. by andrew loew. 1995 : fifteenth - born editions, berkeley.. law publishing 2nd edition 1977 : barbara fernandez, he wrote a book of international law. in : law, war, democracy and society. reprint 2000, by bruce j. klberrie. from the women of the united states : ann barry, the caribbean and...., greeks ( eds. ) theegnous press, 2011 : i. anthony ruffney.. " the part of an american freedom party 2 : 126. " from women to politics '. mildred wellman accused the aclu of firing any other professional if she gets off of her contract with aclu which is a matter of action with a change of duty in australia. legal review bulletin no. 1 - 8. 2011, an australian law review with the understanding that the decision has to be made in australia, and there are legal consequences for the victims of their work in development, the oil industry and to the royal standards of society. the university of maryland press. publications the football club news the aaa ria weekly references truro law books australian brands professional association, the pba association of the pba professional association is the national association for the player's sport of basketball in the professional basketball division of the pba professional association in the pba association. the highest continental association in association with bat is the pbaa grand conference federation. association of players and premier's of the pba was recognised by all the organisations of the pbaa that represented the various groups of pba players in the 1958 pba and the year there were four representatives. the the players who were playing won the gold medal. in march 1963 all the players becoming members of the pbaa in malaysia there won the prg gold and gold commissioner's trophies.
</details>

<details>
<summary><b>DLM + iterative</b> (458 of 512 tokens masked) -- coherent, topically grounded, reads like real encyclopedia text</summary>

> was nominated for the award for best screenplay, and is nominated for best original screenplay. for best main film the bride at the 1st toronto international film festival. winners external links the bride at the festival website films set in toronto toronto ambellona annuensis is a species of ray - finn dorid fish in the family cerodinonithidae. cerodinoni fish of africa fish described in 1937 taxa named by bernard kiefembee inc. v ( 2005, 355 a. c. ) case inc. v ( 2005 v. s. res.,, september, 2009 ) and that of a landmark supreme court ruling which extends all kinds human rights related to civil rights and political rights to privacy. plaintiffs'infernal discretion rests on the defendant's discretion in the court court speaking as an exception to the jurisprudence of the act of 1949. the third rule applies the courts on matters of compulsory citizenship, immigration, and immigration, migration, privacy, and constitutional freedoms. the court also applies the state authority to give theologia libellaps, which applies only to the court's decision that people may have from criticism of the right of defamation, on one particular basis that is not connected to its own defamation. in broad or general interest a majority in the court had made clear that the amendment of united states as well as the way and means of defamation was a violation of criminal duties and constitutional freedom at the time. the courts also held that the third rule rule was " to be an almost unreasonable " rule, except in the opinion of richardson v. cheney in which the court ruled that damages instead of free speech or any other form of defamation were still upheld in the u. s.. references 2006 in case law case law united states supreme court cases appellate courts case law in united states case case law the arehia division of bowls organised 11 national championships. they include the national rugby confederation of bowls, and the world war ii invitational tournament in asia. events the national rugby union championship for bowls is organised by professional bowls. world championship bowls ( created ) determine the broken minimum for all bowls to be played every year from the rest of the world. since 2019, the event will be held annually. the national bowls world championship is contested annually every consecutive years. it was founded in 1999 for the most. super slam bowls while bowls world slam bowls the premier bowls are contested by three pools of teams, with more than 22 teams winning each of the national bowls world cup. england had a record for the most overall event. the champions
</details>

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
