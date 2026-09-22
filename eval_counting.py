"""Does the model know that days go forward?

The corpus never says so, but it is always true: each `DAY n` header is
1 to 3 days after the one before it. Learning that requires carrying a
number across ~360 characters of unrelated text, so it is a clean test of
whether the context window is wide enough to matter.

We generate a pile of text from each checkpoint, pull out the day numbers
in order, and ask two questions of every consecutive pair:

  going forward  -- is the next day later than the one before?
                    (a model emitting random numbers scores ~50%)
  legal step     -- is it exactly 1, 2 or 3 days later, as the corpus has it?
                    (random numbers score ~0%)

    python eval_counting.py checkpoints/*.npz
"""

import argparse
import glob
import os
import re
import sys

import numpy as np

from playground.model import GPT
from playground.tokenizer import CharTokenizer

DAY = re.compile(r"^DAY (\d+)", re.M)


def score(text, seed=0):
    """Two questions, plus a control.

    'legal step' is all-or-nothing: is the next day exactly 1-3 later, as
    the corpus always has it. But a model can be completely wrong about the
    arithmetic and still have learned something real -- that the next number
    lives *near* the last one. So we also measure how far off it lands, and
    compare that against the same numbers in a shuffled order. The shuffle
    keeps the model's range and distribution and destroys only the
    ordering, which is exactly the thing we are testing for.
    """
    days = [int(m.group(1)) for m in DAY.finditer(text)]
    pairs = list(zip(days, days[1:]))
    if not pairs:
        return None
    deltas = [b - a for a, b in pairs]

    control = list(days)
    np.random.default_rng(seed).shuffle(control)
    c_deltas = [b - a for a, b in zip(control, control[1:])]

    def near(ds, limit):
        return sum(0 < d <= limit for d in ds) / len(ds)

    return {
        "headers": len(days),
        "pairs": len(pairs),
        "forward": sum(d > 0 for d in deltas) / len(deltas),
        "legal": sum(1 <= d <= 3 for d in deltas) / len(deltas),
        "near10": near(deltas, 10),
        "median_abs": float(np.median(np.abs(deltas))),
        "ctl_near10": near(c_deltas, 10),
        "ctl_median_abs": float(np.median(np.abs(c_deltas))),
    }


def generate_text(path, samples, chars, temperature, top_k, seed):
    model, meta = GPT.load(path)
    tok = CharTokenizer.from_meta(meta["vocab"])
    rng = np.random.default_rng(seed)
    chunks = []
    for i in range(samples):
        start = f"DAY {rng.integers(50, 500)} -- "
        ids = tok.encode(start)
        got = model.generate_stream(ids, chars, temperature=temperature,
                                    top_k=top_k or None, rng=rng)
        chunks.append(start + tok.decode(list(got)))
    info = {
        "block": model.cfg.block_size,
        "params": model.num_params(),
        "step": int(meta.get("step", 0)),
        "val": float(meta.get("val_loss", float("nan"))),
    }
    return "\n".join(chunks), info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoints", nargs="*")
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--chars", type=int, default=1800)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--seed", type=int, default=99)
    ap.add_argument("--corpus", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "corpus",
        "playground.txt"))
    args = ap.parse_args()

    paths = args.checkpoints or sorted(glob.glob("checkpoints/*.npz"))
    paths = [p for p in paths if not p.endswith(".opt.npz")]
    if not paths:
        sys.exit("no checkpoints given")

    print(f"{args.samples} samples x {args.chars} chars per model, "
          f"temperature {args.temp}, top-k {args.top_k}\n")
    head = (f"{'model':<14} {'window':>7} {'bits/ch':>8} {'days':>6} "
            f"{'forward':>8} {'legal':>7} {'within 10':>10} "
            f"{'|gap|':>7} {'shuffled':>9}")
    print(head)
    print("-" * len(head))

    def row(name, window, bits, s):
        print(f"{name:<14} {window:>7} {bits:>8} {s['pairs']:>6} "
              f"{s['forward']:>7.0%} {s['legal']:>7.0%} {s['near10']:>9.0%} "
              f"{s['median_abs']:>7.0f} "
              f"{s['ctl_near10']:>4.0%}/{s['ctl_median_abs']:>4.0f}")

    # The corpus itself, as the ceiling.
    if os.path.exists(args.corpus):
        with open(args.corpus, encoding="utf-8") as f:
            row("(the corpus)", "--", "--", score(f.read()))

    for path in paths:
        name = os.path.splitext(os.path.basename(path))[0]
        text, info = generate_text(path, args.samples, args.chars,
                                   args.temp, args.top_k, args.seed)
        s = score(text)
        if not s or s["pairs"] < 2:
            print(f"{name:<14} {info['block']:>7} "
                  f"(never produced two day headers)")
            continue
        row(name, info["block"], f"{info['val'] / np.log(2):.2f}", s)

    print("\n'within 10' is how often the next day lands 1-10 later.")
    print("'shuffled' is the same numbers in random order -- the score you")
    print("get from the model's range and distribution alone, with the")
    print("ordering destroyed. Beating it means the model is tracking the")
    print("previous number; matching it means it is not.")


if __name__ == "__main__":
    main()
