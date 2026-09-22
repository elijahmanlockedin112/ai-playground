"""AdamW, hand-rolled, plus the learning-rate schedule."""

import math

import numpy as np


class AdamW:
    """Decoupled weight decay: decay is applied to the weights directly,
    not folded into the gradient. Biases, LayerNorm gains and the position
    table are excluded -- decaying those just hurts."""

    NO_DECAY_SUFFIXES = (".b", ".g", ".b1", ".b2", ".pb", "wpe")

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.95), eps=1e-8,
                 weight_decay=0.1):
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.wd = weight_decay
        self.t = 0
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.decayed = {k: not k.endswith(self.NO_DECAY_SUFFIXES)
                        for k in params}

    def step(self, params, grads, lr=None):
        lr = self.lr if lr is None else lr
        self.t += 1
        bc1 = 1.0 - self.b1 ** self.t
        bc2 = 1.0 - self.b2 ** self.t
        for k, p in params.items():
            gk = grads[k]
            m, v = self.m[k], self.v[k]
            m *= self.b1
            m += (1.0 - self.b1) * gk
            v *= self.b2
            v += (1.0 - self.b2) * (gk * gk)
            step = (m / bc1) / (np.sqrt(v / bc2) + self.eps)
            if self.wd and self.decayed[k]:
                p -= lr * self.wd * p
            p -= lr * step

    def state(self):
        out = {f"opt_m::{k}": v for k, v in self.m.items()}
        out.update({f"opt_v::{k}": v for k, v in self.v.items()})
        out["opt_t"] = np.array(self.t)
        return out

    def load_state(self, blob):
        for k in list(self.m):
            if f"opt_m::{k}" in blob:
                self.m[k] = np.asarray(blob[f"opt_m::{k}"], dtype=np.float32)
                self.v[k] = np.asarray(blob[f"opt_v::{k}"], dtype=np.float32)
        if "opt_t" in blob:
            self.t = int(blob["opt_t"])


def clip_grad_norm(grads, max_norm):
    """Global L2 clipping. Returns the pre-clip norm so you can watch it."""
    total = math.sqrt(sum(float((g * g).sum()) for g in grads.values()))
    if max_norm and total > max_norm:
        scale = max_norm / (total + 1e-6)
        for g in grads.values():
            g *= scale
    return total


def lr_at(step, total_steps, base_lr, warmup=100, min_ratio=0.1):
    """Linear warmup, then cosine decay down to min_ratio * base_lr."""
    if step < warmup:
        return base_lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, total_steps - warmup)
    progress = min(1.0, max(0.0, progress))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return base_lr * (min_ratio + (1.0 - min_ratio) * cosine)
