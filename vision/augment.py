"""Making the training images messy on purpose.

The danger with generated data is that the model learns the *renderer*
rather than the subject. A clean synthetic cube on a flat background is a
much easier thing to recognise than a cube on someone's desk, and a model
trained only on the former will happily reach 99% while having learned
"crisp polygon edges on smooth background" -- a feature that does not exist
in a real photograph.

So everything here exists to destroy the tells of the renderer:

  backgrounds    textured, cluttered, photo-ish, never just flat colour
  optics         defocus, motion blur, vignetting, chromatic fringing
  sensor         shot noise, colour casts, wrong white balance, JPEG blocks
  the world      other objects in shot, things partly covering the cube

None of it makes the cube harder to identify *in principle*. All of it
makes the shortcuts stop working.
"""

import io

import numpy as np

# ------------------------------------------------------------- backgrounds


def _value_noise(rng, n, cells):
    """Smooth random blobs: a coarse grid of random values, bilinearly
    upsampled. A cheap stand-in for natural texture."""
    g = rng.random((cells + 1, cells + 1))
    ys = np.linspace(0, cells, n)
    xs = np.linspace(0, cells, n)
    # clip: linspace includes the endpoint, so the last sample would index
    # one past the grid when we look up the neighbour at +1
    y0 = np.clip(np.floor(ys).astype(int), 0, cells - 1)[:, None]
    x0 = np.clip(np.floor(xs).astype(int), 0, cells - 1)[None, :]
    fy = (ys[:, None] - y0)
    fx = (xs[None, :] - x0)
    fy = fy * fy * (3 - 2 * fy)          # smoothstep, so there are no creases
    fx = fx * fx * (3 - 2 * fx)
    top = g[y0, x0] * (1 - fx) + g[y0, x0 + 1] * fx
    bot = g[y0 + 1, x0] * (1 - fx) + g[y0 + 1, x0 + 1] * fx
    return top * (1 - fy) + bot * fy


def fractal_noise(rng, n, octaves=4):
    out = np.zeros((n, n))
    amp, cells, total = 1.0, 2, 0.0
    for _ in range(octaves):
        out += amp * _value_noise(rng, n, cells)
        total += amp
        amp *= 0.55
        cells *= 2
    return out / total


def textured_background(canvas, rng):
    """Never a flat colour. Wood-ish, cloth-ish, blotchy, or striped --
    whatever it is, it has structure at several scales, like a real scene."""
    n = canvas.shape[0]
    kind = rng.random()
    a = rng.uniform(15, 235, size=3)
    b = rng.uniform(15, 235, size=3)

    if kind < 0.30:                                  # blotchy texture
        t = fractal_noise(rng, n)[:, :, None]
    elif kind < 0.50:                                # directional grain
        t = fractal_noise(rng, n, octaves=3)
        t = (t + np.linspace(0, 1, n)[:, None] * rng.uniform(-0.6, 0.6))
        t = ((t - t.min()) / (np.ptp(t) + 1e-6))[:, :, None]
    elif kind < 0.68:                                # soft stripes
        f = rng.uniform(2, 14)
        ang = rng.uniform(0, np.pi)
        yy, xx = np.mgrid[0:n, 0:n] / n
        t = (0.5 + 0.5 * np.sin(2 * np.pi * f *
                                (xx * np.cos(ang) + yy * np.sin(ang))))
        t = (t * 0.6 + 0.4 * fractal_noise(rng, n))[:, :, None]
    elif kind < 0.84:                                # large soft gradient
        yy, xx = np.mgrid[0:n, 0:n] / n
        t = (xx * rng.uniform(-1, 1) + yy * rng.uniform(-1, 1))
        t = ((t - t.min()) / (np.ptp(t) + 1e-6))[:, :, None]
        t = t * 0.75 + 0.25 * fractal_noise(rng, n)[:, :, None]
    else:                                            # busy clutter texture
        t = fractal_noise(rng, n, octaves=5)[:, :, None]
        t = np.clip(t * rng.uniform(1.2, 2.2) - 0.2, 0, 1)

    canvas[:] = a * (1 - t) + b * t


def clutter(canvas, rng, fill_convex, n_max=5):
    """Other things in the picture. Real photographs are not of one object
    on an empty table."""
    n = canvas.shape[0]
    for _ in range(rng.integers(0, n_max + 1)):
        col = rng.uniform(20, 235, size=3)
        cx, cy = rng.uniform(0, n, 2)
        r = rng.uniform(0.04, 0.22) * n
        k = rng.integers(3, 7)
        th = np.sort(rng.uniform(0, 2 * np.pi, k))
        pts = np.stack([cx + r * np.cos(th) * rng.uniform(0.6, 1.4),
                        cy + r * np.sin(th) * rng.uniform(0.6, 1.4)], axis=1)
        try:
            fill_convex(canvas, pts, col)
        except Exception:
            pass


def occlude(canvas, rng, fill_convex, p=0.30):
    """Something in front of the subject: a hand, a mug, the edge of a
    table. Teaches the model that a cube is still a cube with a corner
    missing.

    Kept deliberately modest: an occluder big enough to hide the whole
    subject would turn a labelled cube into an image with no cube in it,
    which is not augmentation, it is mislabelled data."""
    if rng.random() > p:
        return
    n = canvas.shape[0]
    col = rng.uniform(10, 240, size=3)
    if rng.random() < 0.5:                       # a bar across the frame
        ang = rng.uniform(0, np.pi)
        w = rng.uniform(0.05, 0.16) * n
        cx, cy = rng.uniform(0.25, 0.75, 2) * n
        dx, dy = np.cos(ang) * n, np.sin(ang) * n
        px, py = -np.sin(ang) * w, np.cos(ang) * w
        pts = np.array([[cx - dx + px, cy - dy + py],
                        [cx + dx + px, cy + dy + py],
                        [cx + dx - px, cy + dy - py],
                        [cx - dx - px, cy - dy - py]])
    else:                                        # a blob over one corner
        cx, cy = rng.uniform(0.2, 0.8, 2) * n
        r = rng.uniform(0.08, 0.20) * n
        th = np.sort(rng.uniform(0, 2 * np.pi, 6))
        pts = np.stack([cx + r * np.cos(th), cy + r * np.sin(th)], axis=1)
    fill_convex(canvas, pts, col)


# ------------------------------------------------------------------ optics

def blur(img, sigma):
    rad = int(max(1, round(2 * sigma)))
    xs = np.arange(-rad, rad + 1)
    k = np.exp(-xs ** 2 / (2 * sigma ** 2))
    k /= k.sum()
    out = np.zeros_like(img)
    pad = np.pad(img, ((rad, rad), (0, 0), (0, 0)), mode="edge")
    for i, w in enumerate(k):
        out += w * pad[i:i + img.shape[0]]
    img2, out = out, np.zeros_like(img)
    pad = np.pad(img2, ((0, 0), (rad, rad), (0, 0)), mode="edge")
    for i, w in enumerate(k):
        out += w * pad[:, i:i + img.shape[1]]
    return out


def motion_blur(img, rng):
    length = int(rng.integers(2, 5))
    ang = rng.uniform(0, np.pi)
    dx, dy = np.cos(ang), np.sin(ang)
    out = np.zeros_like(img)
    for i in range(length):
        sy, sx = int(round(dy * i)), int(round(dx * i))
        out += np.roll(np.roll(img, sy, axis=0), sx, axis=1)
    return out / length


def vignette(img, rng):
    n, m = img.shape[:2]
    yy, xx = np.mgrid[0:n, 0:m]
    d = np.sqrt(((xx - m / 2) / (m / 2)) ** 2 + ((yy - n / 2) / (n / 2)) ** 2)
    strength = rng.uniform(0.15, 0.55)
    return img * np.clip(1 - strength * (d ** 2) / 2, 0, 1)[:, :, None]


def chromatic(img, rng):
    """Cheap lens fringing: shift the red and blue channels apart by a pixel."""
    s = int(rng.integers(1, 3))
    out = img.copy()
    out[:, :, 0] = np.roll(img[:, :, 0], s, axis=1)
    out[:, :, 2] = np.roll(img[:, :, 2], -s, axis=1)
    return out


# ------------------------------------------------------------------ sensor

def jpeg(img, rng):
    """A real JPEG round-trip, so the artefacts are the genuine article."""
    from PIL import Image
    q = int(rng.integers(25, 80))
    buf = io.BytesIO()
    Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).save(
        buf, format="JPEG", quality=q)
    buf.seek(0)
    return np.asarray(Image.open(buf).convert("RGB"), dtype=np.float64)


def photometric(img, rng):
    """Everything between the lens and the file: exposure, white balance,
    contrast, and noise that scales with brightness like a real sensor."""
    img = img * rng.uniform(0.55, 1.45)                    # exposure
    img = img * rng.uniform(0.82, 1.18, size=3)            # white balance
    mean = img.mean()
    img = (img - mean) * rng.uniform(0.7, 1.35) + mean     # contrast
    g = rng.uniform(0.75, 1.35)                            # gamma
    img = 255.0 * np.clip(img / 255.0, 0, 1) ** g
    if rng.random() < 0.35:                                # desaturate
        grey = img.mean(-1, keepdims=True)
        img = img * (1 - 0.5) + grey * 0.5 if rng.random() < 0.3 else img
    shot = rng.uniform(1.5, 9.0)                           # noise
    img = img + rng.normal(0, shot, img.shape) * (0.4 + 0.6 * img / 255.0)
    return img


def camera(img, rng):
    """Run an image through a plausible camera. Applied after downsampling,
    at the resolution the model actually sees."""
    if rng.random() < 0.45:
        img = blur(img, rng.uniform(0.4, 1.1))
    if rng.random() < 0.15:
        img = motion_blur(img, rng)
    if rng.random() < 0.20:
        img = chromatic(img, rng)
    if rng.random() < 0.45:
        img = vignette(img, rng)
    img = photometric(img, rng)
    if rng.random() < 0.45:
        img = jpeg(np.clip(img, 0, 255), rng)
    return np.clip(img, 0, 255)
