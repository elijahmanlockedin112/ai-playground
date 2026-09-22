"""Check that the KV cache changes the speed and nothing else.

A cache is a correctness hazard: it is easy to write one that is subtly
wrong -- off-by-one on positions, a stale entry, a mask that lets a token
see the future -- and still produces text that looks fine. So we check it
against the slow path it replaces, which we already trust.

    python test_cache.py
"""

import sys
import time

import numpy as np

from playground.model import GPT, Config


def main():
    rng = np.random.default_rng(0)
    cfg = Config(vocab_size=23, block_size=32, n_layer=3, n_head=4, n_embd=32)
    model = GPT(cfg, seed=5)
    ids = rng.integers(0, cfg.vocab_size, size=20)

    print(f"{cfg}\n")
    ok = True

    # 1. Prefill the whole sequence in one go.
    full, _, _ = model.forward(ids[None, :])
    state = model.new_cache(1)
    pre = model.forward_incremental(ids[None, :], state)
    err = np.abs(full - pre).max()
    ok &= err < 1e-5
    print(f"prefill all {len(ids)} tokens at once   max diff {err:.2e}  "
          f"{'ok' if err < 1e-5 else 'WRONG'}")

    # 2. One token at a time -- the path generation actually uses.
    state = model.new_cache(1)
    step_logits = [model.forward_incremental(ids[None, i:i + 1], state)[0, 0]
                   for i in range(len(ids))]
    err = np.abs(full[0] - np.stack(step_logits)).max()
    ok &= err < 1e-5
    print(f"one token at a time                 max diff {err:.2e}  "
          f"{'ok' if err < 1e-5 else 'WRONG'}")

    # 3. A token must never see the future. Change the last token and the
    #    logits for every earlier position must not move at all.
    poisoned = ids.copy()
    poisoned[-1] = (poisoned[-1] + 7) % cfg.vocab_size
    other, _, _ = model.forward(poisoned[None, :])
    err = np.abs(full[0, :-1] - other[0, :-1]).max()
    ok &= err == 0.0
    print(f"causality (no peeking ahead)        max diff {err:.2e}  "
          f"{'ok' if err == 0.0 else 'WRONG'}")

    # 4. Cross the window boundary, where the cache has to be rebuilt.
    long_prompt = list(rng.integers(0, cfg.vocab_size, size=8))
    a = list(model.generate_stream(long_prompt, 60, temperature=0.0))
    b = list(model.generate_stream(long_prompt, 60, temperature=0.0))
    same = a == b
    ok &= same
    print(f"deterministic across a window rebuild "
          f"({cfg.block_size} wide, 60 generated)  "
          f"{'ok' if same else 'WRONG'}")

    # 5. And the speedup, which is the entire point.
    prompt = list(ids[:8])
    t0 = time.perf_counter()
    cached = list(model.generate_stream(prompt, 120, temperature=0.0))
    t_cached = time.perf_counter() - t0

    t0 = time.perf_counter()
    naive = []
    seq = list(prompt)
    for _ in range(120):
        logits, _, _ = model.forward(
            np.array([seq[-cfg.block_size:]], dtype=np.int64))
        nxt = int(logits[0, -1].argmax())
        naive.append(nxt)
        seq.append(nxt)
    t_naive = time.perf_counter() - t0

    match = cached == naive
    ok &= match
    print(f"\nsame 120 tokens as the uncached path  "
          f"{'ok' if match else 'WRONG'}")
    print(f"  uncached {t_naive * 1000:7.0f} ms   "
          f"({120 / t_naive:6.0f} tokens/sec)")
    print(f"  cached   {t_cached * 1000:7.0f} ms   "
          f"({120 / t_cached:6.0f} tokens/sec)   "
          f"{t_naive / t_cached:.1f}x faster")

    print("\n" + ("all cache checks passed" if ok else "CACHE IS BROKEN"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
