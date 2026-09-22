"""Character-level tokenizer.

Deliberately the dumbest possible choice: one token per character. It makes
the vocabulary tiny (~70 symbols instead of 50,000), so a small model can
actually learn something, and it means the model has to learn spelling from
scratch -- which is most of the fun early on.
"""

import numpy as np


class CharTokenizer:
    def __init__(self, vocab):
        self.itos = list(vocab)
        self.stoi = {ch: i for i, ch in enumerate(self.itos)}

    @classmethod
    def from_text(cls, text):
        return cls(sorted(set(text)))

    @property
    def vocab_size(self):
        return len(self.itos)

    def encode(self, s, unknown="?"):
        fallback = self.stoi.get(unknown, 0)
        return np.array([self.stoi.get(ch, fallback) for ch in s],
                        dtype=np.int64)

    def decode(self, ids):
        return "".join(self.itos[int(i)] for i in np.asarray(ids).reshape(-1))

    # Stored in the checkpoint as one string; np.savez handles unicode fine.
    def to_meta(self):
        return "".join(self.itos)

    @classmethod
    def from_meta(cls, blob):
        return cls(list(str(blob)))

    def __repr__(self):
        shown = "".join(self.itos).replace("\n", "\\n")
        return f"CharTokenizer({self.vocab_size} symbols: {shown!r})"
