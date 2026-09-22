"""Convolution and friends, in NumPy, with the backward pass done by hand.

Same deal as the language model: no autograd, every derivative worked out
and then checked numerically (`gradcheck_cnn.py`).

The one idea worth understanding here is **im2col**. A convolution is a
small filter slid over an image, which sounds nothing like a matrix
multiply -- but if you copy out every patch the filter will ever see and
stack them as rows, it becomes exactly one matrix multiply, and BLAS does
the rest. That single trick is why a convnet runs at a sensible speed here
instead of grinding through a quadruple-nested Python loop.

Everything is NCHW: (batch, channels, height, width).
"""

import numpy as np

DTYPE = np.float32


# --------------------------------------------------------------- im2col

def _indices(C, H, W, kh, kw, stride, pad):
    out_h = (H + 2 * pad - kh) // stride + 1
    out_w = (W + 2 * pad - kw) // stride + 1
    i0 = np.tile(np.repeat(np.arange(kh), kw), C)
    i1 = stride * np.repeat(np.arange(out_h), out_w)
    j0 = np.tile(np.arange(kw), kh * C)
    j1 = stride * np.tile(np.arange(out_w), out_h)
    k = np.repeat(np.arange(C), kh * kw).reshape(-1, 1)
    return k, i0.reshape(-1, 1) + i1, j0.reshape(-1, 1) + j1, out_h, out_w


def im2col(x, kh, kw, stride, pad):
    """(N,C,H,W) -> (N, C*kh*kw, out_h*out_w), one column per output pixel."""
    N, C, H, W = x.shape
    if pad:
        x = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    k, i, j, oh, ow = _indices(C, H, W, kh, kw, stride, pad)
    return x[:, k, i, j], oh, ow


def col2im(cols, shape, kh, kw, stride, pad):
    """The transpose of im2col: scatter each patch back and add the overlaps.

    The adding is the whole point -- a pixel used by nine different filter
    positions receives gradient from all nine."""
    N, C, H, W = shape
    out = np.zeros((N, C, H + 2 * pad, W + 2 * pad), dtype=cols.dtype)
    k, i, j, _, _ = _indices(C, H, W, kh, kw, stride, pad)
    np.add.at(out, (slice(None), k, i, j), cols)
    return out[:, :, pad:H + pad, pad:W + pad] if pad else out


# ---------------------------------------------------------------- layers

class Layer:
    """Layers own their parameters and their gradients, both as plain dicts
    keyed by name, so a model can hand the whole lot to AdamW unchanged."""
    def __init__(self):
        self.params, self.grads = {}, {}

    def forward(self, x, train=True):
        raise NotImplementedError

    def backward(self, dout):
        raise NotImplementedError


class Conv2D(Layer):
    def __init__(self, c_in, c_out, k=3, stride=1, pad=1, rng=None):
        super().__init__()
        rng = rng or np.random.default_rng()
        fan_in = c_in * k * k
        # He initialisation: variance 2/fan_in keeps ReLU activations from
        # shrinking to nothing as depth grows.
        self.params["w"] = (rng.normal(0, np.sqrt(2.0 / fan_in),
                                       (c_out, fan_in))).astype(DTYPE)
        self.params["b"] = np.zeros(c_out, dtype=DTYPE)
        self.k, self.stride, self.pad = k, stride, pad
        self.c_in, self.c_out = c_in, c_out

    def forward(self, x, train=True):
        cols, oh, ow = im2col(x, self.k, self.k, self.stride, self.pad)
        N = x.shape[0]
        # (N, L, fan_in) @ (fan_in, c_out) -> one matmul for the whole batch
        flat = cols.transpose(0, 2, 1).reshape(-1, cols.shape[1])
        out = flat @ self.params["w"].T + self.params["b"]
        self.cache = (x.shape, flat, oh, ow)
        return out.reshape(N, oh * ow, self.c_out) \
                  .transpose(0, 2, 1).reshape(N, self.c_out, oh, ow)

    def backward(self, dout):
        xshape, flat, oh, ow = self.cache
        N = xshape[0]
        d = dout.reshape(N, self.c_out, -1).transpose(0, 2, 1) \
                .reshape(-1, self.c_out)
        self.grads["w"] = (d.T @ flat).astype(DTYPE)
        self.grads["b"] = d.sum(0).astype(DTYPE)
        dcols = (d @ self.params["w"]).reshape(N, oh * ow, -1) \
                                      .transpose(0, 2, 1)
        return col2im(dcols, xshape, self.k, self.k,
                      self.stride, self.pad).astype(DTYPE)


class ReLU(Layer):
    def forward(self, x, train=True):
        self.mask = x > 0
        return x * self.mask

    def backward(self, dout):
        return dout * self.mask


class MaxPool2D(Layer):
    """Keeps the largest value in each window; gradient flows only to the
    pixel that won."""
    def __init__(self, k=2):
        super().__init__()
        self.k = k

    def forward(self, x, train=True):
        N, C, H, W = x.shape
        k = self.k
        H2, W2 = H // k, W // k
        xr = x[:, :, :H2 * k, :W2 * k].reshape(N, C, H2, k, W2, k)
        xr = xr.transpose(0, 1, 2, 4, 3, 5).reshape(N, C, H2, W2, k * k)
        self.arg = xr.argmax(-1)
        self.shape = x.shape
        return xr.max(-1)

    def backward(self, dout):
        N, C, H, W = self.shape
        k = self.k
        H2, W2 = H // k, W // k
        flat = np.zeros((N, C, H2, W2, k * k), dtype=dout.dtype)
        idx = np.ogrid[:N, :C, :H2, :W2]
        flat[idx[0], idx[1], idx[2], idx[3], self.arg] = dout
        out = flat.reshape(N, C, H2, W2, k, k).transpose(0, 1, 2, 4, 3, 5)
        dx = np.zeros(self.shape, dtype=dout.dtype)
        dx[:, :, :H2 * k, :W2 * k] = out.reshape(N, C, H2 * k, W2 * k)
        return dx


class Flatten(Layer):
    def forward(self, x, train=True):
        self.shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, dout):
        return dout.reshape(self.shape)


class Linear(Layer):
    def __init__(self, n_in, n_out, rng=None, scale=1.0):
        """`scale` shrinks the usual He initialisation. The classifier head
        wants scale ~0.01: max-pooling grows activations on the way up (each
        pool takes the largest of four), so with a standard init the logits
        start around +/-2 and the model begins confidently wrong. Starting
        the head near zero makes it begin undecided, which is the honest
        place for an untrained model to be."""
        super().__init__()
        rng = rng or np.random.default_rng()
        self.params["w"] = (scale * rng.normal(0, np.sqrt(2.0 / n_in),
                                               (n_in, n_out))).astype(DTYPE)
        self.params["b"] = np.zeros(n_out, dtype=DTYPE)

    def forward(self, x, train=True):
        self.x = x
        return x @ self.params["w"] + self.params["b"]

    def backward(self, dout):
        self.grads["w"] = (self.x.T @ dout).astype(DTYPE)
        self.grads["b"] = dout.sum(0).astype(DTYPE)
        return dout @ self.params["w"].T


class Dropout(Layer):
    """Off at eval time. With procedurally generated data we can simply make
    more images instead, so this defaults to zero."""
    def __init__(self, p=0.0):
        super().__init__()
        self.p = p

    def forward(self, x, train=True):
        if not train or self.p <= 0:
            self.mask = None
            return x
        self.mask = (np.random.random(x.shape) >= self.p) / (1 - self.p)
        return x * self.mask

    def backward(self, dout):
        return dout if self.mask is None else dout * self.mask


# ------------------------------------------------------------------- loss

def softmax_cross_entropy(logits, targets):
    """Returns (loss, dlogits). Targets are integer class indices."""
    z = logits - logits.max(-1, keepdims=True)
    e = np.exp(z)
    p = e / e.sum(-1, keepdims=True)
    n = len(targets)
    loss = float(-np.log(np.maximum(p[np.arange(n), targets], 1e-12)).mean())
    d = p.copy()
    d[np.arange(n), targets] -= 1.0
    return loss, (d / n).astype(DTYPE)


# ------------------------------------------------------------------ model

class Sequential:
    """A list of layers, plus one flat parameter dict so the AdamW we
    already wrote for the language model works here untouched."""

    def __init__(self, layers):
        self.layers = layers
        self.params = {}
        for i, layer in enumerate(layers):
            for name, arr in layer.params.items():
                self.params[f"{i}.{name}"] = arr

    def forward(self, x, train=True):
        for layer in self.layers:
            x = layer.forward(x, train)
        return x

    def backward(self, dout):
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        grads = {}
        for i, layer in enumerate(self.layers):
            for name, g in layer.grads.items():
                grads[f"{i}.{name}"] = g
        return grads

    def num_params(self):
        return sum(v.size for v in self.params.values())

    def describe(self):
        rows = []
        for i, layer in enumerate(self.layers):
            n = sum(v.size for v in layer.params.values())
            rows.append((i, type(layer).__name__, n))
        return rows
