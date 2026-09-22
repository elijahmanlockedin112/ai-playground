"""A local web playground for the model.

    python server.py            then open http://localhost:8123

Standard library only -- no Flask, no dependencies. Tokens are streamed to
the browser as they are generated, so you watch the model think in real
time rather than waiting for a paragraph.

Checkpoints are rescanned on every request and reloaded when their file
changes on disk, which means you can leave this running while train.py is
working and new models simply appear in the dropdown.
"""

import argparse
import base64
import io
import json
import os
import socket
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

from playground.model import GPT
from playground.tokenizer import CharTokenizer
from vision import cnn as vision_cnn
from vision.render import KINDS, dataset as render_dataset

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR = os.path.join(HERE, "checkpoints")
UI_FILE = os.path.join(HERE, "ui", "index.html")

_loaded = {}                       # name -> (mtime, model, tokenizer)
_lock = threading.Lock()


# ------------------------------------------------------------------ models

def checkpoint_paths():
    if not os.path.isdir(CKPT_DIR):
        return []
    return sorted(
        os.path.join(CKPT_DIR, f) for f in os.listdir(CKPT_DIR)
        if f.endswith(".npz") and not f.endswith(".opt.npz"))


def describe(path):
    """Read a checkpoint's metadata without materializing its weights."""
    try:
        z = np.load(path, allow_pickle=False)
        vocab, block, layers, heads, embd = (int(x) for x in z["__config__"])
        n_params = sum(z[k].size for k in z.files if k.startswith("p::"))
        step = int(z["m::step"]) if "m::step" in z.files else 0
        val = float(z["m::val_loss"]) if "m::val_loss" in z.files else None
        data = str(z["m::data"]) if "m::data" in z.files else None
    except Exception:
        return None
    return {
        "name": os.path.splitext(os.path.basename(path))[0],
        "params": n_params, "block": block, "layers": layers,
        "heads": heads, "embd": embd, "vocab": vocab, "step": step,
        "val_loss": val, "data": data,
        "bits_per_char": (val / np.log(2)) if val else None,
    }


def get_model(name):
    path = os.path.join(CKPT_DIR, name + ".npz")
    if not os.path.exists(path):
        raise FileNotFoundError(name)
    mtime = os.path.getmtime(path)
    with _lock:
        cached = _loaded.get(name)
        if cached and cached[0] == mtime:
            return cached[1], cached[2]
    model, meta = GPT.load(path)            # slow part, done outside the lock
    tok = CharTokenizer.from_meta(meta["vocab"])
    with _lock:
        _loaded[name] = (mtime, model, tok)
    return model, tok


# ------------------------------------------------------------------ vision

_vision = {}                       # name -> (mtime, model)


def describe_vision(path):
    """Convnet checkpoints carry __spec__; language models carry __config__.
    That is how the two kinds are told apart."""
    try:
        z = np.load(path, allow_pickle=False)
        if "__spec__" not in z.files:
            return None
        spec = [int(v) for v in z["__spec__"]]
        size, n_ch = spec[0], spec[1]
        return {
            "name": os.path.splitext(os.path.basename(path))[0],
            "params": sum(z[k].size for k in z.files if k.startswith("p::")),
            "size": size,
            "channels": spec[2:2 + n_ch],
            "step": int(z["m::step"]) if "m::step" in z.files else 0,
            "val_acc": float(z["m::val_acc"]) if "m::val_acc" in z.files
            else None,
            "classes": vision_cnn.classes_for(spec[3 + n_ch]),
        }
    except Exception:
        return None


def get_vision(name):
    path = os.path.join(CKPT_DIR, name + ".npz")
    if not os.path.exists(path):
        raise FileNotFoundError(name)
    mtime = os.path.getmtime(path)
    with _lock:
        hit = _vision.get(name)
        if hit and hit[0] == mtime:
            return hit[1]
    model, _ = vision_cnn.load(path)
    with _lock:
        _vision[name] = (mtime, model)
    return model


def png_data_url(img, scale=3):
    from PIL import Image
    im = Image.fromarray(np.asarray(img, dtype=np.uint8))
    im = im.resize((im.width * scale, im.height * scale), Image.NEAREST)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


# ------------------------------------------------------------------ server

class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "playground"

    def log_message(self, fmt, *a):         # quieter than the default
        if "/api/generate" in (self.path or ""):
            return
        print(f"  {self.command} {self.path}")

    # -- helpers ----------------------------------------------------------

    def _send(self, code, body, ctype="application/json"):
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode()
        elif isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _open_stream(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Transfer-Encoding", "chunked")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

    def _chunk(self, obj):
        data = (json.dumps(obj) + "\n").encode("utf-8")
        self.wfile.write(f"{len(data):X}\r\n".encode() + data + b"\r\n")
        self.wfile.flush()

    def _end_stream(self):
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()

    # -- routes -----------------------------------------------------------

    def do_GET(self):
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            try:
                with open(UI_FILE, "r", encoding="utf-8") as f:
                    return self._send(200, f.read(), "text/html; charset=utf-8")
            except FileNotFoundError:
                return self._send(500, "ui/index.html is missing", "text/plain")
        if path == "/api/models":
            models = [d for d in (describe(p) for p in checkpoint_paths())
                      if d]
            return self._send(200, {"models": models})
        if path == "/api/vision/models":
            models = [d for d in (describe_vision(p)
                                  for p in checkpoint_paths()) if d]
            return self._send(200, {"models": models})
        return self._send(404, {"error": "not found"})

    def _vision_sample(self, req):
        """Render a fresh batch the model has never seen and classify it."""
        name = req.get("model") or ""
        n = max(1, min(48, int(req.get("n", 12))))
        hard = bool(req.get("hard", True))
        try:
            model = get_vision(name)
        except FileNotFoundError:
            return self._send(404, {"error": f"no convnet named {name!r}"})

        info = describe_vision(os.path.join(CKPT_DIR, name + ".npz"))
        size = info["size"] if info else 64
        # default_rng().integers, not np.random.randint: the latter defaults
        # to int32 on Windows and overflows on a 62-bit seed
        seed = int(req.get("seed") or
                   np.random.default_rng().integers(1 << 62))
        # The batch has to be labelled for the task this model was trained
        # on. Rendering 8-class labels for a 2-class model indexes off the
        # end of its class list.
        names = (info["classes"] if info else vision_cnn.CLASSES)
        task = "binary" if len(names) == 2 else "puzzle"
        t0 = time.perf_counter()
        imgs, truth, kinds = render_dataset(n, seed=seed, size=size,
                                            hard=hard, task=task)
        pred, prob = vision_cnn.predict(model, imgs)
        dt = time.perf_counter() - t0

        items = [{
            "png": png_data_url(imgs[i]),
            "pred": names[int(pred[i])],
            "is_cube": bool(pred[i] != 0),      # anything but "not a puzzle"
            "confidence": float(prob[i].max()),
            "cube_prob": float(1.0 - prob[i][0]),
            "truth": names[int(truth[i])],
            "kind": KINDS[int(kinds[i])],
            "correct": bool(pred[i] == truth[i]),
        } for i in range(n)]
        return self._send(200, {
            "items": items,
            "accuracy": float((pred == truth).mean()),
            "seconds": dt,
        })

    def _vision_classify(self, req):
        """Classify images the user supplied, as data URLs.

        The thumbnail we send back is what the model actually saw -- centre
        cropped and squashed to 64x64 -- not the original. Seeing how little
        detail it gets explains most of its mistakes.
        """
        from PIL import Image

        name = req.get("model") or ""
        try:
            model = get_vision(name)
        except FileNotFoundError:
            return self._send(404, {"error": f"no convnet named {name!r}"})
        info = describe_vision(os.path.join(CKPT_DIR, name + ".npz"))
        size = info["size"] if info else 96
        names = info["classes"] if info else vision_cnn.CLASSES

        items, arrays = [], []
        for entry in (req.get("images") or [])[:24]:
            url = entry.get("data", "")
            label = entry.get("name", "image")
            if "," in url:
                url = url.split(",", 1)[1]
            try:
                im = Image.open(io.BytesIO(base64.b64decode(url)))
                im = im.convert("RGB")
            except Exception:
                items.append({"name": label, "error": "could not read"})
                continue
            s = min(im.size)                       # centre crop to a square
            left, top = (im.width - s) // 2, (im.height - s) // 2
            im = im.crop((left, top, left + s, top + s)).resize(
                (size, size), Image.LANCZOS)
            arrays.append(np.asarray(im, dtype=np.uint8))
            items.append({"name": label})

        if not arrays:
            return self._send(400, {"error": "no readable images"})

        imgs = np.stack(arrays)
        pred, prob = vision_cnn.predict(model, imgs)
        out, i = [], 0
        for it in items:
            if "error" in it:
                out.append(it)
                continue
            out.append({
                "name": it["name"],
                "png": png_data_url(imgs[i]),      # what the model saw
                "pred": names[int(pred[i])],
                "is_cube": bool(pred[i] != 0),
                "cube_prob": float(1.0 - prob[i][0]),
                "runner_up": names[int(np.argsort(-prob[i])[1])],
                "confidence": float(prob[i].max()),
                "truth": "your image",
                "correct": True,                   # unknown; not scored
            })
            i += 1
        return self._send(200, {"items": out, "accuracy": None,
                                "seconds": 0.0})

    def do_POST(self):
        route = self.path.split("?")[0]
        if route == "/api/vision/classify":
            try:
                ln = int(self.headers.get("Content-Length", 0))
                if ln > 40 * 1024 * 1024:
                    return self._send(413, {"error": "too much data"})
                return self._vision_classify(
                    json.loads(self.rfile.read(ln) or b"{}"))
            except Exception:
                traceback.print_exc()
                return self._send(400, {"error": "bad request"})
        if route == "/api/vision/sample":
            try:
                ln = int(self.headers.get("Content-Length", 0))
                return self._vision_sample(json.loads(self.rfile.read(ln)
                                                      or b"{}"))
            except Exception:
                traceback.print_exc()
                return self._send(400, {"error": "bad request"})
        if route != "/api/generate":
            return self._send(404, {"error": "not found"})
        try:
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n) or b"{}")
        except Exception:
            return self._send(400, {"error": "bad json"})

        name = req.get("model") or ""
        prompt = req.get("prompt", "")
        try:
            max_new = max(1, min(4000, int(req.get("tokens", 400))))
            temp = max(0.0, min(2.0, float(req.get("temperature", 0.8))))
            top_k = int(req.get("top_k", 20))
            seed = req.get("seed")
        except (TypeError, ValueError):
            return self._send(400, {"error": "bad parameters"})

        try:
            model, tok = get_model(name)
        except FileNotFoundError:
            return self._send(404, {"error": f"no checkpoint named {name!r}"})

        rng = np.random.default_rng(None if seed in (None, "") else int(seed))
        ids = tok.encode(prompt) if prompt else tok.encode("\n")

        self._open_stream()
        started = time.perf_counter()
        count = 0
        try:
            self._chunk({"type": "start", "block": model.cfg.block_size,
                         "prompt_tokens": len(ids)})
            for tid in model.generate_stream(ids, max_new, temperature=temp,
                                             top_k=top_k or None, rng=rng):
                count += 1
                self._chunk({"type": "token", "text": tok.itos[tid]})
            dt = time.perf_counter() - started
            self._chunk({"type": "done", "tokens": count, "seconds": dt,
                         "tokens_per_sec": count / dt if dt else 0})
            self._end_stream()
        except (BrokenPipeError, ConnectionResetError):
            pass                            # the browser hit Stop; fine
        except Exception:
            traceback.print_exc()
            try:
                self._chunk({"type": "error", "message": "generation failed"})
                self._end_stream()
            except OSError:
                pass


def free_port(preferred):
    for port in range(preferred, preferred + 20):
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise SystemExit("no free port found")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8123)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()

    port = free_port(args.port)
    found = [d for d in (describe(p) for p in checkpoint_paths()) if d]
    print(f"playground server")
    print(f"  {len(found)} checkpoint(s) in checkpoints/")
    for d in found:
        bits = f"{d['bits_per_char']:.2f} bits/char" if d["bits_per_char"] \
            else "untrained"
        print(f"    {d['name']:<14} {d['params']:>9,} params  "
              f"block {d['block']:<4} {bits}")
    print(f"\n  http://localhost:{port}\n")
    ThreadingHTTPServer((args.host, port), Handler).serve_forever()


if __name__ == "__main__":
    main()
