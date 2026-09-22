"""Talk to the model.

    python sample.py                              # one sample
    python sample.py --prompt "DAY 12 -- "
    python sample.py --interactive                # type prompts at it
    python sample.py --compare                    # same prompt, 5 temperatures
"""

import argparse
import os
import sys

import numpy as np

from playground.model import GPT, softmax
from playground.tokenizer import CharTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CKPT = os.path.join(HERE, "checkpoints", "tiny.npz")


def stream(model, tok, prompt, n, temperature, top_k, rng, out=sys.stdout):
    """Generate one character at a time and print as we go. There is no KV
    cache, so every character costs a full forward pass over the context --
    which is exactly why real implementations bother with one."""
    ids = list(tok.encode(prompt)) or [tok.stoi.get("\n", 0)]
    produced = []
    for _ in range(n):
        ctx = np.array(ids[-model.cfg.block_size:], dtype=np.int64)[None, :]
        logits, _, _ = model.forward(ctx)
        logits = logits[0, -1].astype(np.float64)
        if temperature <= 1e-6:
            nxt = int(logits.argmax())
        else:
            logits = logits / temperature
            if top_k and 0 < top_k < logits.size:
                logits[logits < np.partition(logits, -top_k)[-top_k]] = -np.inf
            probs = softmax(logits)
            nxt = int(rng.choice(len(probs), p=probs))
        ids.append(nxt)
        ch = tok.itos[nxt]
        produced.append(ch)
        out.write(ch)
        out.flush()
    return "".join(produced)


def load(ckpt):
    if not os.path.exists(ckpt):
        sys.exit(f"no checkpoint at {ckpt} -- run train.py first")
    model, meta = GPT.load(ckpt)
    tok = CharTokenizer.from_meta(meta["vocab"])
    step = int(meta.get("step", 0))
    val = float(meta.get("val_loss", float("nan")))
    print(f"[{os.path.basename(ckpt)}] {model.num_params():,} params, "
          f"trained {step:,} steps, val loss {val:.3f} "
          f"({val / np.log(2):.2f} bits/char)\n")
    return model, tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--prompt", default="DAY 7 -- ")
    ap.add_argument("-n", "--tokens", type=int, default=400)
    ap.add_argument("-t", "--temp", type=float, default=0.8)
    ap.add_argument("-k", "--top-k", type=int, default=20)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--interactive", action="store_true")
    ap.add_argument("--compare", action="store_true")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    model, tok = load(args.ckpt)

    if args.compare:
        for t in (0.2, 0.5, 0.8, 1.0, 1.3):
            print(f"--- temperature {t} " + "-" * 50)
            print(args.prompt, end="")
            stream(model, tok, args.prompt, args.tokens, t, args.top_k, rng)
            print("\n")
        return

    if args.interactive:
        print("type a prompt and press enter. blank line = free-run. "
              "ctrl-c to quit.")
        print(f"(temperature {args.temp}, top-k {args.top_k}, "
              f"{args.tokens} chars per reply)\n")
        while True:
            try:
                p = input(">>> ")
            except (KeyboardInterrupt, EOFError):
                print("\nbye")
                return
            p = p if p else "\n"
            print(p, end="")
            stream(model, tok, p, args.tokens, args.temp, args.top_k, rng)
            print("\n")
        return

    print(args.prompt, end="")
    stream(model, tok, args.prompt, args.tokens, args.temp, args.top_k, rng)
    print()


if __name__ == "__main__":
    main()
