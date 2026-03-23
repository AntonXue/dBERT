"""Dataset loading for dBERT: BookCorpusOpen + English Wikipedia."""

from datasets import load_dataset, concatenate_datasets
from transformers import BertTokenizer

TOKENIZER_NAME = "google-bert/bert-base-uncased"


def load_tokenizer():
    """Load BERT tokenizer. Provides [MASK] at id 103, [PAD] at id 0."""
    return BertTokenizer.from_pretrained(TOKENIZER_NAME)


def load_train_data():
    """Load BookCorpusOpen + Wikipedia, downloaded to disk.

    BookCorpusOpen: ~74M sentences from ~17K books.
    Wikipedia: English Wikipedia articles (20231101 snapshot).
    Both have a 'text' column.
    """
    bookcorpus = load_dataset(
        "lucadiliello/bookcorpusopen",
        split="train",
    )
    wikipedia = load_dataset(
        "wikimedia/wikipedia",
        "20231101.en",
        split="train",
    )
    # Keep only 'text' column from both
    bookcorpus = bookcorpus.select_columns(["text"])
    wikipedia = wikipedia.select_columns(["text"])

    dataset = concatenate_datasets([bookcorpus, wikipedia])
    dataset = dataset.shuffle(seed=42)
    return dataset


def load_eval_data(num_samples=10000):
    """Load a small eval set from Wikipedia for held-out evaluation.

    Takes the last num_samples from the Wikipedia dataset to avoid
    overlap with early training data.
    """
    wiki = load_dataset(
        "wikimedia/wikipedia",
        "20231101.en",
        split="train",
    )
    wiki = wiki.select_columns(["text"])
    # Take from the end to avoid overlap with training
    total = len(wiki)
    eval_set = wiki.select(range(total - num_samples, total))
    return eval_set


def make_collator(tokenizer, max_length=512):
    """Collator that tokenizes and pads text to fixed length.

    Returns dict with 'input_ids' tensor of shape (B, max_length).
    """
    def collate_fn(batch):
        texts = [ex["text"] for ex in batch]
        encodings = tokenizer(
            texts,
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {"input_ids": encodings["input_ids"]}
    return collate_fn
