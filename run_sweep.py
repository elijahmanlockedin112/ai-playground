"""Run the context-window experiment: same model, same data, same number of
tokens per step -- only the size of the context window changes.

Day headers in the corpus sit ~361 characters apart, so the width of the
window decides how often the model can even see the previous day number
while predicting the next one. That ranges from never (96) to almost always
(768). If attention is doing what we think it is doing, the ability to count
should appear as the window crosses that distance.

Each run gets one BLAS thread. The matmuls here are small enough that extra
threads buy almost nothing (2001ms -> 1671ms going from 1 thread to 16), so
N independent processes beat one process with N threads by a wide margin.

    python run_sweep.py              # the whole sweep
    python run_sweep.py --steps 1000 # a quicker look
"""

import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))

# (name, block_size, batch) -- batch chosen so every run sees 6,144
# tokens per step, otherwise we would be comparing token budgets and not
# context windows.
SWEEP = [
    ("count-96",  96,  64),
    ("count-192", 192, 32),
    ("count-256", 256, 24),
    ("count-384", 384, 16),
    ("count-512", 512, 12),
    ("count-768", 768,  8),
]

# A wider, deeper model to actually use in the web playground.
FLAGSHIP = ("playground", 512, 12, dict(layers=4, heads=4, embd=160))


def launch(name, block, batch, steps, extra=None):
    arch = dict(layers=3, heads=4, embd=128)
    arch.update(extra or {})
    log = os.path.join(HERE, "logs", f"{name}.log")
    # Pinned to the single-genre corpus on purpose: the whole measurement
    # depends on how far apart DAY headers are, and mixing in recipes and
    # letters would change that distance and quietly invalidate the result.
    cmd = [sys.executable, "-u", os.path.join(HERE, "train.py"),
           "--data", os.path.join(HERE, "corpus", "chronicles.txt"),
           "--block", str(block), "--batch", str(batch),
           "--layers", str(arch["layers"]), "--heads", str(arch["heads"]),
           "--embd", str(arch["embd"]),
           "--steps", str(steps), "--lr", "2.5e-3", "--warmup", "200",
           "--eval-every", "250", "--save-every", "250", "--sample-every", "0",
           "--out", os.path.join(HERE, "checkpoints", f"{name}.npz")]
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    return subprocess.Popen(cmd, stdout=open(log, "w"),
                            stderr=subprocess.STDOUT, env=env, cwd=HERE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--no-flagship", action="store_true")
    args = ap.parse_args()

    os.makedirs(os.path.join(HERE, "logs"), exist_ok=True)
    os.makedirs(os.path.join(HERE, "checkpoints"), exist_ok=True)

    jobs = []
    for name, block, batch in SWEEP:
        jobs.append((name, launch(name, block, batch, args.steps)))
        print(f"launched {name:<12} window {block:>4}  batch {batch:>2}  "
              f"{block * batch:,} tokens/step")
    if not args.no_flagship:
        n, b, bs, arch = FLAGSHIP
        jobs.append((n, launch(n, b, bs, args.steps, arch)))
        print(f"launched {n:<12} window {b:>4}  batch {bs:>2}  "
              f"({arch['layers']} layers, width {arch['embd']})")

    print(f"\n{len(jobs)} runs going. logs in logs/, checkpoints every 250 "
          f"steps.\nwatch one with:  tail -f logs/count-512.log\n")

    t0 = time.time()
    for name, proc in jobs:
        code = proc.wait()
        print(f"[{(time.time() - t0) / 60:5.1f} min] {name} finished "
              f"(exit {code})")
    print("\nsweep complete. now run:  python eval_counting.py")


if __name__ == "__main__":
    main()
