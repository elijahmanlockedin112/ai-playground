"""Look at what the puzzle classifier actually gets wrong.

An accuracy number tells you almost nothing. This renders a fresh batch the
model has never seen, scores it, breaks the result down by what was really
in the picture, and writes a contact sheet with every prediction on it so
you can go and look at the failures.

    python eval_cnn.py
    python eval_cnn.py --n 2000 --sheet docs/cube_eval.png
"""

import argparse
import os

import numpy as np

from vision.cnn import classes_for, load, predict
from vision.render import KINDS, PUZZLE_KINDS, SIZE, dataset

HERE = os.path.dirname(os.path.abspath(__file__))


def contact_sheet(imgs, pred, prob, truth, path, cols=12, scale=2):
    """Green border = right, red = wrong. Brightness of the border is the
    model's confidence, so a confidently wrong image glows."""
    from PIL import Image
    n = len(imgs)
    rows = (n + cols - 1) // cols
    s, pad = imgs.shape[1], 6
    sheet = np.full((rows * (s + pad) + pad, cols * (s + pad) + pad, 3),
                    24, np.uint8)
    for i in range(n):
        r, c = divmod(i, cols)
        y, x = pad + r * (s + pad), pad + c * (s + pad)
        conf = float(prob[i].max())
        ok = pred[i] == truth[i]
        shade = int(90 + 165 * (conf - 0.5) * 2)
        border = (0, shade, 0) if ok else (shade, 0, 0)
        sheet[y - 2:y + s + 2, x - 2:x + s + 2] = border
        sheet[y:y + s, x:x + s] = imgs[i]
    img = Image.fromarray(sheet)
    img = img.resize((img.width * scale, img.height * scale), Image.NEAREST)
    img.save(path)
    return path


def load_photos(folder, size):
    """Read real photographs: centre-crop to square, resize to the model's
    input. This is the only honest test of whether the thing learned cubes
    or merely learned our renderer."""
    from PIL import Image
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    paths = sorted(p for p in os.listdir(folder) if p.lower().endswith(exts))
    if not paths:
        raise SystemExit(f"no images found in {folder}")
    imgs, names = [], []
    for p in paths:
        im = Image.open(os.path.join(folder, p)).convert("RGB")
        s = min(im.size)
        left, top = (im.width - s) // 2, (im.height - s) // 2
        im = im.crop((left, top, left + s, top + s)).resize(
            (size, size), Image.LANCZOS)
        imgs.append(np.asarray(im, dtype=np.uint8))
        names.append(p)
    return np.stack(imgs), names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(HERE, "checkpoints",
                                                   "puzzles.npz"))
    ap.add_argument("--photos", default=None,
                    help="a folder of real photos to classify instead")
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--sheet", default=os.path.join(HERE, "docs",
                                                    "puzzle_eval.png"))
    ap.add_argument("--show", type=int, default=48,
                    help="how many images to put on the contact sheet")
    args = ap.parse_args()

    if not os.path.exists(args.ckpt):
        raise SystemExit(f"no checkpoint at {args.ckpt} -- run train_cnn.py")
    model, meta = load(args.ckpt)
    size = int(meta.get("size", SIZE))
    step = int(meta.get("step", 0))
    print(f"[{os.path.basename(args.ckpt)}] {model.num_params():,} params, "
          f"trained {step:,} steps\n")

    # --- real photographs, if we were given any
    if args.photos:
        imgs, names = load_photos(args.photos, size)
        pred, prob = predict(model, imgs)
        classes = classes_for(int(prob.shape[1]))
        print(f"{len(names)} photo(s) from {args.photos}\n")
        width = max(len(n) for n in names) + 2
        for name, p, pr in zip(names, pred, prob):
            bar = "#" * int(pr[p] * 30)
            runner = int(np.argsort(-pr)[1])
            print(f"  {name:<{width}}{classes[p]:<18}{pr[p]:6.1%}  {bar}"
                  f"   (else {classes[runner]}, {pr[runner]:.0%})")
        say = int((pred != 0).sum())
        print(f"\nsaw a puzzle in {say} of {len(names)}")
        if args.sheet:
            os.makedirs(os.path.dirname(args.sheet), exist_ok=True)
            contact_sheet(imgs, pred, prob, pred, args.sheet)
            print(f"contact sheet -> {os.path.relpath(args.sheet, HERE)}")
        return

    imgs, truth, kinds = dataset(args.n, seed=args.seed, size=size)
    pred, prob = predict(model, imgs)
    right = pred == truth

    print(f"overall accuracy on {args.n} unseen images: {right.mean():.2%}\n")

    names = classes_for(int(prob.shape[1]))

    # "Is there a puzzle at all" is a different question from "which one",
    # and they fail differently: mistaking a 4x4 for a 5x5 is a miscount,
    # mistaking a checkerboard for a puzzle is a hallucination.
    if len(names) > 2:
        any_pred, any_true = pred != 0, truth != 0
        tp = int((any_pred & any_true).sum())
        fp = int((any_pred & ~any_true).sum())
        fn = int((~any_pred & any_true).sum())
        detected = right[any_true].mean() if any_true.any() else 0
        print(f"  spotted a puzzle in {tp} of {tp + fn}   "
              f"recall {tp / max(tp + fn, 1):.2%}")
        print(f"  saw one in {fp} that had none    "
              f"precision {tp / max(tp + fp, 1):.2%}")
        print(f"  and named it correctly {detected:.2%} of the time\n")

        print("confusion — rows are the truth, columns what it said:")
        short = [n.replace(" cube", "").replace("3x3 Rubik's", "3x3")
                 .replace("not a puzzle", "none")[:8] for n in names]
        print(" " * 12 + "".join(f"{s:>9}" for s in short))
        for t in range(len(names)):
            m = truth == t
            if not m.any():
                continue
            row = [int(((pred == p) & m).sum()) for p in range(len(names))]
            cells = "".join(f"{v:>9}" if v else f"{'.':>9}" for v in row)
            print(f"{short[t]:>11} {cells}")
        print()

    print(f"{'what it really was':<20}{'n':>5}{'correct':>9}   ")
    print("-" * 62)
    for k, name in enumerate(KINDS):
        m = kinds == k
        if not m.any():
            continue
        acc = right[m].mean()
        bar = "#" * int(acc * 30)
        tag = "  <- a puzzle" if name in PUZZLE_KINDS else ""
        print(f"{name:<20}{m.sum():>5}{acc:>8.1%}   {bar}{tag}")

    # The confidently wrong ones are where the model's idea of a cube is
    # visibly not ours.
    wrong = np.where(~right)[0]
    if len(wrong):
        conf = prob[wrong].max(-1)
        order = wrong[np.argsort(-conf)]
        print(f"\nmost confident mistakes ({len(wrong)} wrong in total):")
        for i in order[:8]:
            print(f"  a {KINDS[kinds[i]]:<10} it called "
                  f"{names[pred[i]]!r:<18} with "
                  f"{prob[i].max():.1%} confidence")

    if args.sheet:
        os.makedirs(os.path.dirname(args.sheet), exist_ok=True)
        # Show the mistakes first, then fill up with correct ones.
        order = list(np.where(~right)[0]) + list(np.where(right)[0])
        pick = np.array(order[:args.show])
        contact_sheet(imgs[pick], pred[pick], prob[pick], truth[pick],
                      args.sheet)
        print(f"\ncontact sheet -> {os.path.relpath(args.sheet, HERE)} "
              f"(mistakes first; red border = wrong, brighter = more sure)")


if __name__ == "__main__":
    main()
