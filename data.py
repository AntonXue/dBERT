"""Dataset loading for dBERT: BookCorpusOpen + English Wikipedia.

Splits raw text 95/5, then tokenizes and packs each split into
fixed-length chunks independently (no padding waste, no token leakage).
"""

import torch
from datasets import load_dataset, concatenate_datasets, Dataset
from transformers import BertTokenizer

TOKENIZER_NAME = "google-bert/bert-base-uncased"
EVAL_FRACTION = 0.05
MAX_LENGTH = 512


def load_tokenizer():
    """Load BERT tokenizer. Provides [MASK] at id 103, [PAD] at id 0."""
    return BertTokenizer.from_pretrained(TOKENIZER_NAME)


def _tokenize_and_pack(dataset, tokenizer, max_length):
    """Tokenize all texts and pack into fixed-length chunks.

    Concatenates all tokens into one flat stream, then splits into
    non-overlapping sequences of exactly max_length. Drops the remainder.
    """
    all_ids = []
    for i, ex in enumerate(dataset):
        ids = tokenizer.encode(ex["text"], add_special_tokens=False)
        all_ids.extend(ids)
        if (i + 1) % 500_000 == 0:
            print(f"  tokenized {i+1}/{len(dataset)} docs, {len(all_ids)/1e6:.1f}M tokens so far")

    n_chunks = len(all_ids) // max_length
    if n_chunks == 0:
        return Dataset.from_dict({"input_ids": []})

    all_ids = all_ids[:n_chunks * max_length]
    chunks = [all_ids[i * max_length:(i + 1) * max_length] for i in range(n_chunks)]
    return Dataset.from_dict({"input_ids": chunks})


def load_data(max_length=MAX_LENGTH, tokenizer=None):
    """Load BookCorpusOpen + Wikipedia, split 95/5, tokenize and pack.

    Returns (train_dataset, eval_dataset), where each has a single
    'input_ids' column with packed token sequences of exactly max_length.

    Split happens on raw documents BEFORE tokenization, so there is
    zero token leakage between train and eval.
    """
    if tokenizer is None:
        tokenizer = load_tokenizer()

    # Load raw text
    bookcorpus = load_dataset("lucadiliello/bookcorpusopen", split="train")
    bookcorpus = bookcorpus.select_columns(["text"])

    wiki = load_dataset("wikimedia/wikipedia", "20231101.en", split="train")
    wiki = wiki.select_columns(["text"])

    raw = concatenate_datasets([bookcorpus, wiki])
    print(f"Total documents: {len(raw):,}")

    # Split raw documents 95/5
    splits = raw.train_test_split(test_size=EVAL_FRACTION, seed=42)
    raw_train = splits["train"]
    raw_eval = splits["test"]
    print(f"Split: {len(raw_train):,} train docs, {len(raw_eval):,} eval docs")

    # Tokenize and pack each split independently
    print("Packing train split...")
    train_dataset = _tokenize_and_pack(raw_train, tokenizer, max_length)
    print(f"  -> {len(train_dataset):,} sequences of {max_length} tokens")

    print("Packing eval split...")
    eval_dataset = _tokenize_and_pack(raw_eval, tokenizer, max_length)
    print(f"  -> {len(eval_dataset):,} sequences of {max_length} tokens")

    return train_dataset, eval_dataset


def make_collator():
    """Collator for pre-packed sequences. Just stacks input_ids into a tensor."""
    def collate_fn(batch):
        input_ids = torch.tensor([ex["input_ids"] for ex in batch], dtype=torch.long)
        return {"input_ids": input_ids}
    return collate_fn
