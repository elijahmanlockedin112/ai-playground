"""A transformer language model, written from scratch in NumPy.

There is no autograd here. Every derivative in this file was worked out by
hand, which means two things:
  1. You can read the entire model, end to end, and know exactly what it does.
  2. gradcheck.py exists, because hand-derived gradients are usually wrong
     the first time.

Architecture is decoder-only GPT: learned token + position embeddings, N
pre-norm blocks of (causal multi-head self-attention -> MLP), a final
LayerNorm, and an output head whose weights are tied to the token embedding.
"""

import numpy as np

DTYPE = np.float32

# Swap this to np.float64 before building a model when you want exact
# numerics (gradcheck.py does exactly that). Training runs in float32.

# ---------------------------------------------------------------- activations

SQRT_2_OVER_PI = np.float32(np.sqrt(2.0 / np.pi))


def gelu(x):
    """Tanh approximation of GELU, the same one GPT-2 used."""
    inner = SQRT_2_OVER_PI * (x + np.float32(0.044715) * x ** 3)
    return (np.float32(0.5) * x * (1.0 + np.tanh(inner))).astype(DTYPE)


def gelu_backward(x, dout):
    inner = SQRT_2_OVER_PI * (x + np.float32(0.044715) * x ** 3)
    t = np.tanh(inner)
    dinner = SQRT_2_OVER_PI * (1.0 + np.float32(3 * 0.044715) * x ** 2)
    dx = 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * dinner
    return (dout * dx).astype(DTYPE)


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def sample_next(logits, temperature, top_k, rng):
    """Pick one token id from a row of logits.

    Temperature divides the logits: low sharpens toward whatever the model
    is most sure of, high flattens toward chaos. Top-k throws away
    everything outside the k most likely, so one unlucky draw from the long
    tail cannot derail a whole paragraph.
    """
    logits = np.asarray(logits, dtype=np.float64).reshape(-1)
    if temperature <= 1e-6:
        return int(logits.argmax())
    logits = logits / temperature
    if top_k and 0 < top_k < logits.size:
        logits = np.where(logits < np.partition(logits, -top_k)[-top_k],
                          -np.inf, logits)
    probs = softmax(logits)
    return int(rng.choice(probs.size, p=probs))


# ----------------------------------------------------------------- layer norm

def layernorm(x, g, b, eps=1e-5):
    mu = x.mean(axis=-1, keepdims=True)
    xc = x - mu
    var = (xc * xc).mean(axis=-1, keepdims=True)
    rstd = 1.0 / np.sqrt(var + eps)
    xhat = xc * rstd
    out = g * xhat + b
    return out.astype(DTYPE), (xhat, rstd, g)


def layernorm_backward(dout, cache):
    xhat, rstd, g = cache
    C = xhat.shape[-1]
    dg = (dout * xhat).reshape(-1, C).sum(0)
    db = dout.reshape(-1, C).sum(0)
    dxhat = dout * g
    mean_dxhat = dxhat.mean(axis=-1, keepdims=True)
    mean_dxhat_xhat = (dxhat * xhat).mean(axis=-1, keepdims=True)
    dx = (dxhat - mean_dxhat - xhat * mean_dxhat_xhat) * rstd
    return dx.astype(DTYPE), dg.astype(DTYPE), db.astype(DTYPE)


# --------------------------------------------------------------------- config

class Config:
    def __init__(self, vocab_size, block_size=128, n_layer=4, n_head=4,
                 n_embd=128):
        self.vocab_size = int(vocab_size)
        self.block_size = int(block_size)
        self.n_layer = int(n_layer)
        self.n_head = int(n_head)
        self.n_embd = int(n_embd)
        assert self.n_embd % self.n_head == 0, "n_embd must divide by n_head"

    def to_dict(self):
        return {k: getattr(self, k) for k in
                ("vocab_size", "block_size", "n_layer", "n_head", "n_embd")}

    @staticmethod
    def from_dict(d):
        return Config(**{k: int(v) for k, v in d.items()})

    def __repr__(self):
        return (f"Config(vocab={self.vocab_size}, block={self.block_size}, "
                f"layers={self.n_layer}, heads={self.n_head}, "
                f"embd={self.n_embd})")


# name -> (block_size, n_layer, n_head, n_embd)
PRESETS = {
    "nano":   (64,  2, 2,  64),
    "tiny":   (96,  3, 3,  96),
    "small":  (128, 4, 4, 128),
    "medium": (192, 6, 6, 192),
}


# ---------------------------------------------------------------------- model

class GPT:
    def __init__(self, cfg, seed=1337):
        self.cfg = cfg
        rng = np.random.default_rng(seed)
        C, L = cfg.n_embd, cfg.n_layer
        std = 0.02
        # Residual projections start smaller so the residual stream does not
        # blow up as depth grows (the GPT-2 trick).
        rstd = std / np.sqrt(2 * L)

        def normal(*shape, s=std):
            return rng.normal(0.0, s, size=shape).astype(DTYPE)

        def zeros(*shape):
            return np.zeros(shape, dtype=DTYPE)

        def ones(*shape):
            return np.ones(shape, dtype=DTYPE)

        p = {}
        p["wte"] = normal(cfg.vocab_size, C)
        p["wpe"] = normal(cfg.block_size, C, s=0.01)
        for i in range(L):
            p[f"b{i}.ln1.g"] = ones(C)
            p[f"b{i}.ln1.b"] = zeros(C)
            p[f"b{i}.attn.w"] = normal(C, 3 * C)
            p[f"b{i}.attn.b"] = zeros(3 * C)
            p[f"b{i}.attn.pw"] = normal(C, C, s=rstd)
            p[f"b{i}.attn.pb"] = zeros(C)
            p[f"b{i}.ln2.g"] = ones(C)
            p[f"b{i}.ln2.b"] = zeros(C)
            p[f"b{i}.mlp.w1"] = normal(C, 4 * C)
            p[f"b{i}.mlp.b1"] = zeros(4 * C)
            p[f"b{i}.mlp.w2"] = normal(4 * C, C, s=rstd)
            p[f"b{i}.mlp.b2"] = zeros(C)
        p["lnf.g"] = ones(C)
        p["lnf.b"] = zeros(C)
        self.params = p

        # Causal mask: position t may only attend to positions <= t.
        T = cfg.block_size
        self.mask = np.triu(np.full((T, T), -1e9, dtype=DTYPE), k=1)

    def num_params(self):
        return sum(v.size for v in self.params.values())

    # ------------------------------------------------------------- forward

    def forward(self, idx, targets=None):
        """idx: (B, T) int token ids. targets: (B, T) or None.

        Returns (logits, loss, cache); loss is None when targets is None.
        """
        p, cfg = self.params, self.cfg
        B, T = idx.shape
        assert T <= cfg.block_size, f"sequence {T} > block_size {cfg.block_size}"
        C, nh = cfg.n_embd, cfg.n_head
        hs = C // nh
        scale = np.float32(1.0 / np.sqrt(hs))

        x = (p["wte"][idx] + p["wpe"][:T][None, :, :]).astype(DTYPE)

        blocks = []
        for i in range(cfg.n_layer):
            bc = {}

            # --- causal self-attention
            y, ln1c = layernorm(x, p[f"b{i}.ln1.g"], p[f"b{i}.ln1.b"])
            bc["ln1"] = ln1c
            y2d = y.reshape(B * T, C)
            bc["y2d"] = y2d
            qkv = y2d @ p[f"b{i}.attn.w"] + p[f"b{i}.attn.b"]
            qkv = qkv.reshape(B, T, 3, nh, hs).transpose(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]            # each (B, nh, T, hs)

            scores = (q @ k.transpose(0, 1, 3, 2)) * scale
            scores = scores + self.mask[:T, :T]
            att = softmax(scores, axis=-1).astype(DTYPE)
            o = att @ v                                  # (B, nh, T, hs)
            o2d = o.transpose(0, 2, 1, 3).reshape(B * T, C)
            bc.update(q=q, k=k, v=v, att=att, o2d=o2d)

            a = (o2d @ p[f"b{i}.attn.pw"] + p[f"b{i}.attn.pb"]).reshape(B, T, C)
            x = x + a

            # --- position-wise MLP
            y2, ln2c = layernorm(x, p[f"b{i}.ln2.g"], p[f"b{i}.ln2.b"])
            bc["ln2"] = ln2c
            h2d = y2.reshape(B * T, C)
            bc["h2d"] = h2d
            fc = h2d @ p[f"b{i}.mlp.w1"] + p[f"b{i}.mlp.b1"]
            act = gelu(fc)
            bc.update(fc=fc, act=act)
            m = (act @ p[f"b{i}.mlp.w2"] + p[f"b{i}.mlp.b2"]).reshape(B, T, C)
            x = x + m

            blocks.append(bc)

        xf, lnfc = layernorm(x, p["lnf.g"], p["lnf.b"])
        xf2d = xf.reshape(B * T, C)
        logits = (xf2d @ p["wte"].T).reshape(B, T, cfg.vocab_size)

        cache = {"idx": idx, "blocks": blocks, "lnf": lnfc, "xf2d": xf2d,
                 "B": B, "T": T, "scale": scale}

        loss = None
        if targets is not None:
            probs = softmax(logits, axis=-1)
            flat = probs.reshape(B * T, -1)
            picked = flat[np.arange(B * T), targets.reshape(-1)]
            loss = float(-np.log(np.maximum(picked, 1e-12)).mean())
            cache["probs"] = probs
            cache["targets"] = targets
        return logits, loss, cache

    # ------------------------------------------------------------ backward

    def backward(self, cache):
        """Returns a dict of gradients with the same keys as self.params."""
        p, cfg = self.params, self.cfg
        B, T = cache["B"], cache["T"]
        C, nh = cfg.n_embd, cfg.n_head
        hs = C // nh
        scale = cache["scale"]
        N = B * T

        g = {k: np.zeros_like(v) for k, v in p.items()}

        # --- cross entropy -> logits
        probs = cache["probs"].reshape(N, -1).copy()
        probs[np.arange(N), cache["targets"].reshape(-1)] -= 1.0
        dlogits = (probs / N).astype(DTYPE)          # (N, V)

        # --- tied output head: logits = xf @ wte.T
        xf2d = cache["xf2d"]
        g["wte"] += dlogits.T @ xf2d
        dxf = (dlogits @ p["wte"]).reshape(B, T, C)

        dx, dgf, dbf = layernorm_backward(dxf, cache["lnf"])
        g["lnf.g"] += dgf
        g["lnf.b"] += dbf

        for i in reversed(range(cfg.n_layer)):
            bc = cache["blocks"][i]

            # --- MLP backward
            dm2d = dx.reshape(N, C)
            g[f"b{i}.mlp.w2"] += bc["act"].T @ dm2d
            g[f"b{i}.mlp.b2"] += dm2d.sum(0)
            dact = dm2d @ p[f"b{i}.mlp.w2"].T
            dfc = gelu_backward(bc["fc"], dact)
            g[f"b{i}.mlp.w1"] += bc["h2d"].T @ dfc
            g[f"b{i}.mlp.b1"] += dfc.sum(0)
            dh = (dfc @ p[f"b{i}.mlp.w1"].T).reshape(B, T, C)

            dln2, dg2, db2 = layernorm_backward(dh, bc["ln2"])
            g[f"b{i}.ln2.g"] += dg2
            g[f"b{i}.ln2.b"] += db2
            dx = dx + dln2                     # residual around the MLP

            # --- attention backward
            da2d = dx.reshape(N, C)
            g[f"b{i}.attn.pw"] += bc["o2d"].T @ da2d
            g[f"b{i}.attn.pb"] += da2d.sum(0)
            do2d = da2d @ p[f"b{i}.attn.pw"].T
            do = do2d.reshape(B, T, nh, hs).transpose(0, 2, 1, 3)

            att, q, k, v = bc["att"], bc["q"], bc["k"], bc["v"]
            datt = do @ v.transpose(0, 1, 3, 2)
            dv = att.transpose(0, 1, 3, 2) @ do
            # row-wise softmax jacobian; masked entries are 0 in att already
            dscores = att * (datt - (datt * att).sum(-1, keepdims=True))
            dq = (dscores @ k) * scale
            dk = (dscores.transpose(0, 1, 3, 2) @ q) * scale

            dqkv = np.stack([dq, dk, dv], axis=0)         # (3, B, nh, T, hs)
            dqkv = dqkv.transpose(1, 3, 0, 2, 4).reshape(N, 3 * C)
            g[f"b{i}.attn.w"] += bc["y2d"].T @ dqkv
            g[f"b{i}.attn.b"] += dqkv.sum(0)
            dy = (dqkv @ p[f"b{i}.attn.w"].T).reshape(B, T, C)

            dln1, dg1, db1 = layernorm_backward(dy, bc["ln1"])
            g[f"b{i}.ln1.g"] += dg1
            g[f"b{i}.ln1.b"] += db1
            dx = dx + dln1                     # residual around attention

        # --- embeddings
        np.add.at(g["wte"], cache["idx"].reshape(-1), dx.reshape(N, C))
        g["wpe"][:T] += dx.sum(axis=0)
        return g

    # -------------------------------------------------------- the KV cache

    def new_cache(self, batch=1):
        """Somewhere to keep the keys and values we have already computed.

        Without this, generating character 400 re-runs attention over all
        400 -- the same keys and values recomputed from scratch every time.
        They never change, so we compute each one once and keep it."""
        C, nh = self.cfg.n_embd, self.cfg.n_head
        hs, L, T = C // nh, self.cfg.n_layer, self.cfg.block_size
        shape = (batch, nh, T, hs)
        return {"n": 0,
                "k": [np.zeros(shape, dtype=DTYPE) for _ in range(L)],
                "v": [np.zeros(shape, dtype=DTYPE) for _ in range(L)]}

    def forward_incremental(self, idx_new, state):
        """Push T new tokens through, attending to everything cached before
        them. T can be the whole prompt (prefill) or a single token (decode).
        Returns logits for the new tokens only.
        """
        p, cfg = self.params, self.cfg
        n = state["n"]
        B, T = idx_new.shape
        assert n + T <= cfg.block_size, "context window is full"
        C, nh = cfg.n_embd, cfg.n_head
        hs = C // nh
        scale = np.float32(1.0 / np.sqrt(hs))

        # Query at global position n+i may see keys at 0 .. n+i.
        rows = np.arange(T)[:, None] + n
        cols = np.arange(n + T)[None, :]
        mask = np.where(cols <= rows, 0.0, -1e9).astype(DTYPE)

        x = (p["wte"][idx_new] + p["wpe"][n:n + T][None, :, :]).astype(DTYPE)

        for i in range(cfg.n_layer):
            y, _ = layernorm(x, p[f"b{i}.ln1.g"], p[f"b{i}.ln1.b"])
            qkv = y.reshape(B * T, C) @ p[f"b{i}.attn.w"] + p[f"b{i}.attn.b"]
            qkv = qkv.reshape(B, T, 3, nh, hs).transpose(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]

            state["k"][i][:, :, n:n + T] = k        # remember, do not recompute
            state["v"][i][:, :, n:n + T] = v
            K = state["k"][i][:, :, :n + T]
            V = state["v"][i][:, :, :n + T]

            scores = (q @ K.transpose(0, 1, 3, 2)) * scale + mask
            att = softmax(scores, axis=-1).astype(DTYPE)
            o2d = (att @ V).transpose(0, 2, 1, 3).reshape(B * T, C)
            x = x + (o2d @ p[f"b{i}.attn.pw"]
                     + p[f"b{i}.attn.pb"]).reshape(B, T, C)

            y2, _ = layernorm(x, p[f"b{i}.ln2.g"], p[f"b{i}.ln2.b"])
            act = gelu(y2.reshape(B * T, C) @ p[f"b{i}.mlp.w1"]
                       + p[f"b{i}.mlp.b1"])
            x = x + (act @ p[f"b{i}.mlp.w2"]
                     + p[f"b{i}.mlp.b2"]).reshape(B, T, C)

        xf, _ = layernorm(x, p["lnf.g"], p["lnf.b"])
        state["n"] = n + T
        return (xf.reshape(B * T, C) @ p["wte"].T).reshape(B, T, cfg.vocab_size)

    # ------------------------------------------------------------ sampling

    def generate_stream(self, ids, max_new_tokens, temperature=1.0,
                        top_k=None, rng=None, stop_ids=None):
        """Yields token ids one at a time, KV-cached.

        When the window fills up we cannot simply drop the oldest entry --
        position embeddings are absolute, so every cached key would be at
        the wrong position. Instead we rebuild the cache from the most
        recent 3/4 of the window, which costs one prefill every
        block_size/4 tokens instead of a full pass every single token.
        """
        rng = rng or np.random.default_rng()
        bs = self.cfg.block_size
        ids = list(ids) or [0]

        state = self.new_cache(1)
        ctx = ids[-bs:]
        logits = self.forward_incremental(np.array([ctx], dtype=np.int64),
                                          state)[0, -1]

        for _ in range(max_new_tokens):
            nxt = sample_next(logits, temperature, top_k, rng)
            yield nxt
            ids.append(nxt)
            if stop_ids and nxt in stop_ids:
                return
            if state["n"] >= bs:
                state = self.new_cache(1)
                ctx = ids[-(bs - bs // 4):]
                logits = self.forward_incremental(
                    np.array([ctx], dtype=np.int64), state)[0, -1]
            else:
                logits = self.forward_incremental(
                    np.array([[nxt]], dtype=np.int64), state)[0, -1]

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None,
                 rng=None, stop_ids=None):
        """Convenience wrapper: returns prompt + continuation as (1, N)."""
        ids = list(np.asarray(idx).reshape(-1))
        out = list(self.generate_stream(ids, max_new_tokens, temperature,
                                        top_k, rng, stop_ids))
        return np.array([ids + out], dtype=np.int64)

    # --------------------------------------------------------- persistence

    def save(self, path, meta=None):
        blob = {f"p::{k}": v for k, v in self.params.items()}
        blob["__config__"] = np.array(
            [self.cfg.vocab_size, self.cfg.block_size, self.cfg.n_layer,
             self.cfg.n_head, self.cfg.n_embd], dtype=np.int64)
        for k, v in (meta or {}).items():
            blob[f"m::{k}"] = np.asarray(v)
        np.savez(path, **blob)

    @staticmethod
    def load(path):
        z = np.load(path, allow_pickle=False)
        v, bs, nl, nh, ne = (int(x) for x in z["__config__"])
        model = GPT(Config(v, bs, nl, nh, ne))
        for key in z.files:
            if key.startswith("p::"):
                model.params[key[3:]] = z[key].astype(DTYPE)
        meta = {k[3:]: z[k] for k in z.files if k.startswith("m::")}
        return model, meta
