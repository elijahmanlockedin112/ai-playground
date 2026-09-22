"""Proof that the convnet's hand-derived gradients are the real gradients.

Same method as gradcheck.py: nudge one weight, see how much the loss
actually moved, compare with what backward() claimed. Runs in float64,
because a finite difference in float32 is mostly noise.

Convolution backward is the one most likely to be subtly wrong -- col2im
has to scatter-*add* overlapping patches, and if it overwrites instead, the
gradient is quietly too small and training merely trains badly rather than
visibly failing.

    python gradcheck_cnn.py
"""

import sys

import numpy as np

from vision import layers as L

L.DTYPE = np.float64          # must be set before any layer is constructed

from vision.cnn import build                       # noqa: E402
from vision.layers import softmax_cross_entropy    # noqa: E402


def loss_of(model, x, y):
    return softmax_cross_entropy(model.forward(x, train=False), y)[0]


def main(samples=4, seed=3):
    rng = np.random.default_rng(seed)
    # Tiny, so a numerical check is quick, but every layer type is present.
    model = build(size=16, channels=(4, 6), hidden=8, n_classes=3, seed=seed)
    for k in model.params:
        model.params[k] = model.params[k].astype(np.float64)
    for i, layer in enumerate(model.layers):
        for name in list(layer.params):
            layer.params[name] = model.params[f"{i}.{name}"]

    x = rng.normal(0, 1, (3, 3, 16, 16))
    y = rng.integers(0, 3, 3)

    logits = model.forward(x)
    loss, dlogits = softmax_cross_entropy(logits, y)
    grads = model.backward(dlogits)

    print(f"{model.num_params():,} parameters, loss at init {loss:.6f} "
          f"(uniform baseline {np.log(3):.6f})\n")
    print(f"{'layer':<22}{'parameter':>10}{'shape':>16}"
          f"{'max abs err':>13}   verdict")
    print("-" * 68)

    worst, failures = 0.0, []
    for i, layer in enumerate(model.layers):
        for name in layer.params:
            key = f"{i}.{name}"
            p = model.params[key]
            flat_ana = grads[key].reshape(-1)
            flat_p = p.reshape(-1)
            # the biggest-gradient entries, plus a few at random
            picks = set(np.argsort(-np.abs(flat_ana))[:samples].tolist())
            picks |= set(rng.choice(p.size, min(samples, p.size),
                                    replace=False).tolist())
            err = 0.0
            for idx in picks:
                orig = flat_p[idx]
                eps = 1e-5
                flat_p[idx] = orig + eps
                lp = loss_of(model, x, y)
                flat_p[idx] = orig - eps
                lm = loss_of(model, x, y)
                flat_p[idx] = orig
                num = (lp - lm) / (2 * eps)
                err = max(err, abs(num - flat_ana[idx]))
            worst = max(worst, err)
            ok = err < 1e-7
            if not ok:
                failures.append(key)
            print(f"{type(layer).__name__ + f' #{i}':<22}{name:>10}"
                  f"{str(p.shape):>16}{err:>13.2e}   "
                  f"{'ok' if ok else 'MISMATCH'}")

    print("-" * 68)
    if failures:
        print(f"FAILED on: {', '.join(failures)}")
        return 1
    print(f"every gradient matches. worst absolute error: {worst:.2e}")
    print("convolution, pooling and the classifier head all backprop "
          "correctly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
