"""Train the playground model.

    python train.py --preset tiny --steps 2000
    python train.py --resume checkpoints/tiny.npz --steps 2000   # keep going

Ctrl-C is safe: the checkpoint is written before it exits.
"""

import argparse
import os
import time

import numpy as np

from playground.data import Dataset
from playground.model import GPT, Config, PRESETS
from playground.optim import AdamW, clip_grad_norm, lr_at
from playground.tokenizer import CharTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA = os.path.join(HERE, "corpus", "playground.txt")


def estimate_loss(model, data, split, batch_size, block_size, rng, batches=12):
    losses = []
    for _ in range(batches):
        x, y = data.batch(split, batch_size, block_size, rng)
        _, loss, _ = model.forward(x, y)
        losses.append(loss)
    return float(np.mean(losses))


def sample_text(model, tok, prompt, n=220, temperature=0.8, top_k=20,
                rng=None):
    ids = tok.encode(prompt if prompt else "\n")
    out = model.generate(ids, n, temperature=temperature, top_k=top_k, rng=rng)
    return tok.decode(out[0])


def opt_path(ckpt):
    return os.path.splitext(ckpt)[0] + ".opt.npz"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="tiny", choices=sorted(PRESETS))
    # Any of these override the preset, so you can sweep one dimension.
    ap.add_argument("--block", type=int, default=None)
    ap.add_argument("--layers", type=int, default=None)
    ap.add_argument("--heads", type=int, default=None)
    ap.add_argument("--embd", type=int, default=None)
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--sample-every", type=int, default=500)
    ap.add_argument("--save-every", type=int, default=500,
                    help="checkpoint every N steps so you can watch it learn")
    ap.add_argument("--prompt", default="DAY 7 -- ")
    ap.add_argument("--out", default=None, help="checkpoint path")
    ap.add_argument("--resume", default=None, help="checkpoint to continue")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    # ---------------------------------------------------------------- data
    data = Dataset.from_file(args.data)
    print(f"corpus: {os.path.relpath(args.data, HERE)}")
    print(f"        {data.summary()}")

    # --------------------------------------------------------------- model
    step0 = 0
    if args.resume:
        model, meta = GPT.load(args.resume)
        tok = CharTokenizer.from_meta(meta["vocab"])
        step0 = int(meta.get("step", 0))
        data = Dataset(data.text, tokenizer=tok)
        out_path = args.out or args.resume
        print(f"resumed {args.resume} at step {step0}")
    else:
        tok = data.tok
        block, layers, heads, embd = PRESETS[args.preset]
        block = args.block or block
        layers = args.layers or layers
        heads = args.heads or heads
        embd = args.embd or embd
        cfg = Config(tok.vocab_size, block, layers, heads, embd)
        model = GPT(cfg, seed=args.seed)
        out_path = args.out or os.path.join(HERE, "checkpoints",
                                            f"{args.preset}.npz")

    cfg = model.cfg
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    print(f"model:  {cfg}")
    print(f"        {model.num_params():,} parameters")

    opt = AdamW(model.params, lr=args.lr, weight_decay=args.weight_decay)
    if args.resume and os.path.exists(opt_path(args.resume)):
        opt.load_state(np.load(opt_path(args.resume)))
        print(f"        optimizer state restored")

    total = step0 + args.steps
    uniform = float(np.log(cfg.vocab_size))
    print(f"train:  {args.steps} steps, batch {args.batch} x {cfg.block_size} "
          f"= {args.batch * cfg.block_size:,} tokens/step")
    print(f"        a model that knows nothing scores {uniform:.3f}\n")

    best = float("inf")
    t_start = time.time()
    t_win = time.time()

    def save(tag=""):
        model.save(out_path, meta={"vocab": tok.to_meta(), "step": step + 1,
                                   "val_loss": last_val,
                                   "data": os.path.basename(args.data)})
        np.savez(opt_path(out_path), **opt.state())
        if tag:
            print(f"  -> saved {os.path.basename(out_path)} {tag}")

    last_val = float("nan")
    step = step0
    try:
        for step in range(step0, total):
            lr = lr_at(step - step0, args.steps, args.lr, warmup=args.warmup)
            x, y = data.batch("train", args.batch, cfg.block_size, rng)
            _, loss, cache = model.forward(x, y)
            grads = model.backward(cache)
            gnorm = clip_grad_norm(grads, args.clip)
            opt.step(model.params, grads, lr=lr)

            if (step + 1) % args.eval_every == 0 or step == step0:
                last_val = estimate_loss(model, data, "val", args.batch,
                                         cfg.block_size, rng)
                dt = time.time() - t_win
                t_win = time.time()
                per = dt / (args.eval_every if step != step0 else 1)
                bpc = last_val / np.log(2)
                flag = ""
                if last_val < best:
                    best = last_val
                    flag = " *"
                print(f"step {step + 1:>6}/{total}  train {loss:.3f}  "
                      f"val {last_val:.3f}  ({bpc:.2f} bits/char)  "
                      f"lr {lr:.2e}  |g| {gnorm:4.1f}  {per * 1000:5.0f} ms/step{flag}")

            if args.save_every and (step + 1) % args.save_every == 0:
                save()

            if args.sample_every and (step + 1) % args.sample_every == 0:
                print("\n  ---- sample " + "-" * 58)
                text = sample_text(model, tok, args.prompt, rng=rng)
                for line in text.split("\n"):
                    print("  | " + line)
                print("  " + "-" * 70 + "\n")
                save()
    except KeyboardInterrupt:
        print("\ninterrupted -- saving before exit")

    save(f"(step {step + 1})")
    mins = (time.time() - t_start) / 60
    print(f"\ndone in {mins:.1f} min. best val loss {best:.4f} "
          f"({best / np.log(2):.2f} bits/char, down from {uniform / np.log(2):.2f})")
    print(f"talk to it:  python sample.py --ckpt {os.path.relpath(out_path, HERE)}")


if __name__ == "__main__":
    main()
