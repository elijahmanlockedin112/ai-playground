"""Corpus loading and batching."""

import numpy as np

from .tokenizer import CharTokenizer


class Dataset:
    def __init__(self, text, tokenizer=None, val_frac=0.1):
        self.text = text
        self.tok = tokenizer or CharTokenizer.from_text(text)
        ids = self.tok.encode(text)
        n = int(len(ids) * (1.0 - val_frac))
        self.train = ids[:n]
        self.val = ids[n:]

    @classmethod
    def from_file(cls, path, val_frac=0.1):
        with open(path, "r", encoding="utf-8") as f:
            return cls(f.read(), val_frac=val_frac)

    def batch(self, split, batch_size, block_size, rng):
        """Sample `batch_size` random windows. Each target is the input
        shifted one step left -- every position predicts its next character,
        so one window of length T gives T training signals."""
        data = self.train if split == "train" else self.val
        hi = len(data) - block_size - 1
        starts = rng.integers(0, hi, size=batch_size)
        x = np.stack([data[s:s + block_size] for s in starts])
        y = np.stack([data[s + 1:s + 1 + block_size] for s in starts])
        return x, y

    def summary(self):
        return (f"{len(self.text):,} chars | vocab {self.tok.vocab_size} | "
                f"train {len(self.train):,} / val {len(self.val):,} tokens")
