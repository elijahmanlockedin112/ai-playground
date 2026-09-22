"""Train the twisty-puzzle classifier.

    python train_cnn.py --steps 1200
    python train_cnn.py --resume checkpoints/cube.npz --steps 600

By default every training batch is rendered fresh, so the model never sees
the same image twice. That is the luxury of generating your own data: there
is no training set to overfit, and the gap between train and validation loss
means something other than memorisation. Pass `--pool N` to train on a fixed
set of N images instead, if you want to watch it overfit on purpose.

The validation set is fixed (seed 999) so the numbers are comparable across
runs, and it reports accuracy per distractor type -- the average hides which
things actually fool it, which is the only interesting part.
"""

import argparse
import os
import time

import numpy as np

from playground.optim import AdamW, clip_grad_norm, lr_at
from vision.cnn import CLASSES, build, load, preprocess, save
from vision.layers import softmax_cross_entropy
from vision.render import KINDS, PUZZLE_KINDS, dataset

HERE = os.path.dirname(os.path.abspath(__file__))


def evaluate(model, x, y, kinds, batch=200):
    """Returns (loss, accuracy, per-kind accuracy)."""
    losses, correct = [], np.zeros(len(y), dtype=bool)
    for i in range(0, len(y), batch):
        xb, yb = x[i:i + batch], y[i:i + batch]
        logits = model.forward(xb, train=False)
        losses.append(softmax_cross_entropy(logits, yb)[0] * len(yb))
        correct[i:i + batch] = logits.argmax(-1) == yb
    per_kind = {}
    for k, name in enumerate(KINDS):
        m = kinds == k
        if m.any():
            per_kind[name] = float(correct[m].mean())
    return sum(losses) / len(y), float(correct.mean()), per_kind


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--clip", type=float, default=2.0)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--size", type=int, default=96)
    ap.add_argument("--channels", default="16,32,64,64")
    ap.add_argument("--hidden", type=int, default=96)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--pool", type=int, default=0,
                    help="train on a fixed set of N images instead of fresh ones")
    ap.add_argument("--val", type=int, default=1200)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--save-every", type=int, default=100)
    ap.add_argument("--out", default=os.path.join(HERE, "checkpoints", "cube.npz"))
    ap.add_argument("--resume", default=None)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    # ---- validation set, fixed forever
    t0 = time.time()
    vx_raw, vy, vkinds = dataset(args.val, seed=999, size=args.size)
    vx = preprocess(vx_raw)
    print(f"validation: {args.val} images, fixed seed  "
          f"({time.time() - t0:.0f}s to render)")

    pool = None
    if args.pool:
        t0 = time.time()
        px, py, _ = dataset(args.pool, seed=args.seed + 7, size=args.size)
        pool = (preprocess(px), py)
        print(f"training pool: {args.pool} fixed images "
              f"({time.time() - t0:.0f}s)")
    else:
        print("training data: rendered fresh every step (infinite, "
              "no training set to overfit)")

    # ---- model
    step0 = 0
    if args.resume:
        model, meta = load(args.resume)
        step0 = int(meta.get("step", 0))
        out_path = args.out or args.resume
        print(f"resumed {args.resume} at step {step0}")
    else:
        channels = tuple(int(c) for c in args.channels.split(","))
        model = build(args.size, channels, args.hidden, len(CLASSES),
                      args.dropout, seed=args.seed)
        out_path = args.out
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    print(f"model: {model.num_params():,} parameters, "
          f"{args.size}x{args.size} RGB in, {len(CLASSES)} classes out")
    print(f"train: {args.steps} steps, batch {args.batch}, lr {args.lr}")
    print(f"       guessing scores {np.log(len(CLASSES)):.3f} loss, "
          f"{1 / len(CLASSES):.0%} accuracy\n")

    opt = AdamW(model.params, lr=args.lr, weight_decay=args.weight_decay)
    total = step0 + args.steps
    best = 0.0
    last = (float("nan"), 0.0, {})
    t_start = t_win = time.time()
    step = step0

    def checkpoint():
        save(model, out_path, meta={"step": step + 1, "val_loss": last[0],
                                    "val_acc": last[1], "size": args.size,
                                    "classes": "|".join(CLASSES)})

    try:
        for step in range(step0, total):
            if pool is not None:
                pick = rng.integers(0, len(pool[1]), args.batch)
                xb, yb = pool[0][pick], pool[1][pick]
            else:
                imgs, yb, _ = dataset(args.batch, seed=int(rng.integers(1 << 62)),
                                      size=args.size)
                xb = preprocess(imgs)

            logits = model.forward(xb, train=True)
            loss, dlogits = softmax_cross_entropy(logits, yb)
            grads = model.backward(dlogits)
            gnorm = clip_grad_norm(grads, args.clip)
            opt.step(model.params, grads,
                     lr=lr_at(step - step0, args.steps, args.lr,
                              warmup=args.warmup))

            if (step + 1) % args.eval_every == 0 or step == step0:
                last = evaluate(model, vx, vy, vkinds)
                dt = time.time() - t_win
                t_win = time.time()
                per = dt / (args.eval_every if step != step0 else 1)
                flag = " *" if last[1] > best else ""
                best = max(best, last[1])
                print(f"step {step + 1:>5}/{total}  train {loss:.4f}  "
                      f"val {last[0]:.4f}  acc {last[1]:6.2%}  "
                      f"|g| {gnorm:5.2f}  {per * 1000:5.0f} ms/step{flag}")

            if args.save_every and (step + 1) % args.save_every == 0:
                checkpoint()
    except KeyboardInterrupt:
        print("\ninterrupted -- saving before exit")

    checkpoint()
    loss_v, acc, per_kind = evaluate(model, vx, vy, vkinds)
    print(f"\ndone in {(time.time() - t_start) / 60:.1f} min. "
          f"best accuracy {best:.2%}\n")
    print("accuracy by what the image actually contained:")
    for name, a in sorted(per_kind.items(), key=lambda kv: kv[1]):
        bar = "#" * int(a * 40)
        star = "  <- a puzzle" if name in PUZZLE_KINDS else ""
        print(f"  {name:<10}{a:7.1%}  {bar}{star}")
    print(f"\ntry it:  python eval_cnn.py --ckpt "
          f"{os.path.relpath(out_path, HERE)}")


if __name__ == "__main__":
    main()
