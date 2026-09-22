"""Proof that the hand-derived gradients are actually the gradients.

For a sample of individual weights we nudge the weight by +/- eps, measure
how the loss actually changes, and compare that to what backward() claimed.
Runs in float64 so the numerical derivative is trustworthy.

    python gradcheck.py
"""

import sys

import numpy as np

from playground import model as M

M.DTYPE = np.float64            # must be set before the model is built

from playground.model import GPT, Config       # noqa: E402


def numerical_grad(model, x, y, key, flat_idx, eps=1e-5):
    p = model.params[key]
    flat = p.reshape(-1)
    orig = flat[flat_idx].copy()

    flat[flat_idx] = orig + eps
    _, loss_pos, _ = model.forward(x, y)
    flat[flat_idx] = orig - eps
    _, loss_neg, _ = model.forward(x, y)
    flat[flat_idx] = orig
    return (loss_pos - loss_neg) / (2 * eps)


def main(samples_per_param=4, seed=7):
    rng = np.random.default_rng(seed)
    cfg = Config(vocab_size=11, block_size=8, n_layer=2, n_head=2, n_embd=16)
    model = GPT(cfg, seed=seed)

    B, T = 3, cfg.block_size
    x = rng.integers(0, cfg.vocab_size, size=(B, T))
    y = rng.integers(0, cfg.vocab_size, size=(B, T))

    _, loss, cache = model.forward(x, y)
    grads = model.backward(cache)

    print(f"model: {cfg}")
    print(f"params: {model.num_params():,}   loss at init: {loss:.6f} "
          f"(uniform baseline {np.log(cfg.vocab_size):.6f})")
    print()
    # Judged on absolute error. Relative error is meaningless for the many
    # weights whose true gradient is ~1e-9: the finite difference itself is
    # only good to ~1e-11 absolute, so a correct gradient can still show a
    # huge relative "error" there.
    print(f"{'parameter':<16} {'shape':<12} {'max abs err':>12} "
          f"{'rel err':>10}   verdict")
    print("-" * 66)

    worst = 0.0
    failures = []
    for key in model.params:
        p = model.params[key]
        # Bias the sample towards weights that actually matter, plus a few
        # at random, so a bug anywhere still gets caught.
        flat_ana = grads[key].reshape(-1)
        big = np.argsort(-np.abs(flat_ana))[:samples_per_param]
        rand = rng.choice(p.size, size=min(samples_per_param, p.size),
                          replace=False)
        abs_err = rel_err = 0.0
        for i in set(list(big) + list(rand)):
            num = numerical_grad(model, x, y, key, int(i))
            ana = float(flat_ana[int(i)])
            abs_err = max(abs_err, abs(num - ana))
            if max(abs(num), abs(ana)) > 1e-6:
                rel_err = max(rel_err, abs(num - ana) / max(abs(num), abs(ana)))
        worst = max(worst, abs_err)
        ok = abs_err < 1e-7
        if not ok:
            failures.append(key)
        print(f"{key:<16} {str(p.shape):<12} {abs_err:>12.2e} "
              f"{rel_err:>10.2e}   {'ok' if ok else 'MISMATCH'}")

    print("-" * 66)
    if failures:
        print(f"FAILED on {len(failures)} parameter(s): {', '.join(failures)}")
        return 1
    print(f"every gradient matches. worst absolute error: {worst:.2e}")
    print("the backward pass is correct. we can train this thing.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
