"""The convnet itself, plus saving and loading.

Shape of the thing: four conv stages, each halving the image and doubling
the detail it can represent, then a small classifier head.

    96x96x3  -> conv 16 -> pool -> 48x48
             -> conv 32 -> pool -> 24x24
             -> conv 64 -> pool -> 12x12
             -> conv 64 -> pool ->  6x6
             -> flatten (2304) -> 96 -> 8 classes

The early layers can only see a 3x3 patch, so they learn edges and colour
boundaries. By the last stage one unit is looking at most of the image and
can represent something like "a grid of coloured squares with dark lines
between them, on a face at an angle" -- which is the thing we are asking
about.
"""

import numpy as np

from .layers import (Conv2D, Dropout, Flatten, Linear, MaxPool2D, ReLU,
                     Sequential)

from .render import CLASSES as PUZZLE_CLASSES

# The original task was binary. It is kept because those checkpoints still
# load, and a model says how many classes it has by the shape of its head.
BINARY_CLASSES = ["not a cube", "Rubik's cube"]
CLASSES = PUZZLE_CLASSES


def classes_for(n):
    """Name the outputs of a model with n of them."""
    if n == len(PUZZLE_CLASSES):
        return PUZZLE_CLASSES
    if n == len(BINARY_CLASSES):
        return BINARY_CLASSES
    return [f"class {i}" for i in range(n)]


def build(size=96, channels=(16, 32, 64, 64), hidden=96,
          n_classes=len(PUZZLE_CLASSES),
          dropout=0.0, seed=0):
    rng = np.random.default_rng(seed)
    layers = []
    c_in, side = 3, size
    for c_out in channels:
        layers += [Conv2D(c_in, c_out, 3, 1, 1, rng), ReLU(), MaxPool2D(2)]
        c_in, side = c_out, side // 2
    layers += [Flatten(), Linear(c_in * side * side, hidden, rng), ReLU()]
    if dropout:
        layers.append(Dropout(dropout))
    layers.append(Linear(hidden, n_classes, rng, scale=0.01))

    model = Sequential(layers)
    model.spec = dict(size=size, channels=list(channels), hidden=hidden,
                      n_classes=n_classes, dropout=dropout)
    return model


def preprocess(imgs):
    """uint8 (N,H,W,3) -> float32 (N,3,H,W), centred roughly on zero."""
    x = np.asarray(imgs, dtype=np.float32) / 255.0
    if x.ndim == 3:
        x = x[None]
    return ((x - 0.45) / 0.25).transpose(0, 3, 1, 2).astype(np.float32)


def save(model, path, meta=None):
    blob = {f"p::{k}": v for k, v in model.params.items()}
    s = model.spec
    blob["__spec__"] = np.array(
        [s["size"], len(s["channels"]), *s["channels"], s["hidden"],
         s["n_classes"]], dtype=np.int64)
    for k, v in (meta or {}).items():
        blob[f"m::{k}"] = np.asarray(v)
    np.savez(path, **blob)


def load(path):
    z = np.load(path, allow_pickle=False)
    spec = [int(v) for v in z["__spec__"]]
    size, n_ch = spec[0], spec[1]
    channels = spec[2:2 + n_ch]
    hidden, n_classes = spec[2 + n_ch], spec[3 + n_ch]
    model = build(size, tuple(channels), hidden, n_classes)
    for key in z.files:
        if key.startswith("p::"):
            name = key[3:]
            model.params[name][...] = z[key]      # in place, so layers see it
    meta = {k[3:]: z[k] for k in z.files if k.startswith("m::")}
    return model, meta


def predict(model, imgs):
    """Returns (labels, probabilities)."""
    logits = model.forward(preprocess(imgs), train=False)
    z = logits - logits.max(-1, keepdims=True)
    e = np.exp(z)
    p = e / e.sum(-1, keepdims=True)
    return p.argmax(-1), p
