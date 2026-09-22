"""Draws the training images. Every pixel the CNN ever sees comes from here.

Same rule as the text corpus: we make the data, so it is ours, it is free of
anyone's copyright, it is perfectly labelled, and there is an unlimited
supply of it.

A cube is built in 3D -- eight corners, six faces -- rotated to a random
orientation, back-face culled, projected, and rasterised. Each visible face
is subdivided into a 3x3 grid of stickers with a gap between them showing
the dark plastic underneath. That gap, and the 3x3-ness, is the entire
visual signature of a Rubik's cube, and it is what the model has to find.

The negatives matter more than the positives. If "not a cube" only meant
spheres and noise, a model could score 99% by learning "is it boxy" and we
would have proved nothing. So the distractors include plain boxes (cube
shape, no stickers) and checkerboard boxes (a grid, but the wrong grid),
which can only be separated by actually looking at the sticker layout.
"""

import numpy as np

from .augment import camera, clutter, occlude, textured_background
from .shapes import (cube, dodecahedron, grid_quads, grid_triangles,
                     megaminx_face, skewb_face, tetrahedron)

SIZE = 96          # output images are SIZE x SIZE x 3
# 96, not 64: a 5x5 face at 64px is about three pixels per sticker, which is
# not enough to count them. Telling a 4x4 from a 5x5 is the hardest thing we
# ask, and it is pointless to ask it at a resolution where the answer is not
# present.
SUPERSAMPLE = 2    # render at 2x and box-filter down, for cheap anti-aliasing

# The six standard Rubik's colours.
STICKERS = np.array([
    [255, 255, 255],     # white
    [255, 213,  0],      # yellow
    [200,  30,  40],     # red
    [255, 110,  20],     # orange
    [0,   170,  85],     # green
    [0,    90, 200],     # blue
], dtype=np.float64)

PLASTIC = np.array([22, 22, 26], dtype=np.float64)

# Unit cube corners and the four corners of each face, wound consistently.
CORNERS = np.array([
    [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
    [-1, -1,  1], [1, -1,  1], [1, 1,  1], [-1, 1,  1],
], dtype=np.float64)

FACES = [
    ([0, 1, 2, 3], [0, 0, -1]),
    ([5, 4, 7, 6], [0, 0, 1]),
    ([4, 0, 3, 7], [-1, 0, 0]),
    ([1, 5, 6, 2], [1, 0, 0]),
    ([4, 5, 1, 0], [0, -1, 0]),
    ([3, 2, 6, 7], [0, 1, 0]),
]


# ------------------------------------------------------------------ geometry

def rotation(yaw, pitch, roll):
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    return rz @ ry @ rx


def project(pts3d, scale, cx, cy, dist=7.0):
    """Weak perspective: things further away get slightly smaller."""
    z = pts3d[:, 2] + dist
    f = dist / np.maximum(z, 0.1)
    return np.stack([cx + pts3d[:, 0] * scale * f,
                     cy - pts3d[:, 1] * scale * f], axis=1)


def fill_convex(canvas, pts, color):
    """Rasterise a convex polygon by testing every pixel in its bounding box
    against all edges at once. Small enough polygons that this is fast."""
    h, w = canvas.shape[:2]
    x0 = max(int(np.floor(pts[:, 0].min())), 0)
    x1 = min(int(np.ceil(pts[:, 0].max())) + 1, w)
    y0 = max(int(np.floor(pts[:, 1].min())), 0)
    y1 = min(int(np.ceil(pts[:, 1].max())) + 1, h)
    if x1 <= x0 or y1 <= y0:
        return

    # Make the winding consistent so "inside" is one sign for every edge.
    area = 0.0
    for i in range(len(pts)):
        a, b = pts[i], pts[(i + 1) % len(pts)]
        area += a[0] * b[1] - b[0] * a[1]
    if area < 0:
        pts = pts[::-1]

    ys, xs = np.mgrid[y0:y1, x0:x1]
    inside = np.ones(xs.shape, dtype=bool)
    for i in range(len(pts)):
        a, b = pts[i], pts[(i + 1) % len(pts)]
        cross = (b[0] - a[0]) * (ys - a[1]) - (b[1] - a[1]) * (xs - a[0])
        inside &= cross >= 0
        if not inside.any():
            return
    canvas[y0:y1, x0:x1][inside] = color


def shade(color, normal, light=(0.4, 0.7, 0.6)):
    n = np.asarray(normal, dtype=np.float64)
    n = n / (np.linalg.norm(n) + 1e-9)
    l = np.asarray(light) / np.linalg.norm(light)
    return np.clip(color * (0.55 + 0.45 * max(0.0, float(n @ l))), 0, 255)


def bilinear(quad, u0, u1, v0, v1):
    """The sub-quad of `quad` covering [u0,u1] x [v0,v1] in face coordinates."""
    a, b, c, d = quad
    def at(u, v):
        top = (1 - u) * a + u * b
        bot = (1 - u) * d + u * c
        return (1 - v) * top + v * bot
    return np.array([at(u0, v0), at(u1, v0), at(u1, v1), at(u0, v1)])


# ------------------------------------------------------------- the puzzles

# kind -> (solid, how the faces are cut, grid size)
PUZZLE_SPECS = {
    "cube2":    ("cube", "grid", 2),
    "cube3":    ("cube", "grid", 3),
    "cube4":    ("cube", "grid", 4),
    "cube5":    ("cube", "grid", 5),
    "pyraminx": ("tetra", "tri", 3),
    "skewb":    ("cube", "skewb", 0),
    "megaminx": ("dodeca", "mega", 0),
}

_SOLID_CACHE = {}


def solid(name):
    """Vertices scaled to a common circumradius, so a tetrahedron does not
    end up looking tiny beside a dodecahedron and the model cannot tell them
    apart on size alone. Normalising by the largest *coordinate* is the
    tempting mistake -- it leaves rounder solids visibly smaller."""
    if name not in _SOLID_CACHE:
        verts, faces = {"cube": cube, "tetra": tetrahedron,
                        "dodeca": dodecahedron}[name]()
        verts = verts / np.linalg.norm(verts, axis=1).max()
        _SOLID_CACHE[name] = (verts, faces)
    return _SOLID_CACHE[name]


def draw_puzzle(canvas, rng, kind, scale, cx, cy, rot):
    """Any of the twisty puzzles: rotate, cull the faces pointing away,
    paint the body, then lay the stickers on with gaps between them."""
    solid_name, cut, n = PUZZLE_SPECS[kind]
    verts, faces = solid(solid_name)
    if solid_name == "dodeca":
        scale *= 1.18        # reads smaller than a cube at equal circumradius
    v = verts @ rot.T
    proj = project(v, scale, cx, cy)
    margin = rng.uniform(0.07, 0.15)
    body = PLASTIC * rng.uniform(0.7, 1.8)

    visible = []
    for f in faces:
        pts = v[f]
        centre = pts.mean(0)
        nrm = np.cross(pts[1] - pts[0], pts[2] - pts[0])
        nrm = nrm / (np.linalg.norm(nrm) + 1e-9)
        if nrm @ centre < 0:                 # point it away from the middle
            nrm = -nrm
        if nrm[2] >= -0.02:                  # facing away from the camera
            continue
        visible.append((centre[2], f, nrm))
    visible.sort(key=lambda t: -t[0])        # furthest painted first

    for _, f, nrm in visible:
        poly = proj[f]
        fill_convex(canvas, poly, shade(body, nrm))
        if cut == "grid":
            pieces = grid_quads(poly, n, margin)
        elif cut == "tri":
            pieces = grid_triangles(poly, n, margin)
        elif cut == "skewb":
            pieces = skewb_face(poly, margin)
        else:
            pieces = megaminx_face(poly, margin)
        for piece in pieces:
            col = STICKERS[rng.integers(len(STICKERS))]
            fill_convex(canvas, piece, shade(col, nrm))


# -------------------------------------------------------------- distractors

def draw_box(canvas, rng, style, scale, cx, cy, rot):
    """style: 'rubik' | 'plain' | 'checker' | 'stickerless'."""
    pts = CORNERS @ rot.T
    proj = project(pts, scale, cx, cy)

    flat = rng.uniform(70, 230, size=3) if style in ("plain", "stickerless") \
        else None
    checker_cols = STICKERS[rng.choice(len(STICKERS), 2, replace=False)] \
        if style == "checker" else None

    order = []
    for idx, normal in FACES:
        n = rot @ np.asarray(normal, dtype=np.float64)
        if n[2] >= -0.02:          # back-face cull; +z is away from camera
            continue
        order.append((float(n[2]), idx, n))
    order.sort()                    # nearest last, so it paints over

    for _, idx, n in order:
        quad = proj[idx]
        if style == "plain":
            fill_convex(canvas, quad, shade(flat, n))
            continue
        if style == "stickerless":
            fill_convex(canvas, quad, shade(flat, n))
            continue

        fill_convex(canvas, quad, shade(PLASTIC, n))   # the body first
        if style == "checker":
            k = rng.choice([4, 6, 8])
            for i in range(k):
                for j in range(k):
                    col = checker_cols[(i + j) % 2]
                    sub = bilinear(quad, i / k, (i + 1) / k,
                                   j / k, (j + 1) / k)
                    fill_convex(canvas, sub, shade(col, n))
        else:                                           # rubik
            m = rng.uniform(0.06, 0.12)                 # sticker gap
            for i in range(3):
                for j in range(3):
                    col = STICKERS[rng.integers(len(STICKERS))]
                    sub = bilinear(quad, (i + m) / 3, (i + 1 - m) / 3,
                                   (j + m) / 3, (j + 1 - m) / 3)
                    fill_convex(canvas, sub, shade(col, n))


def draw_sphere(canvas, rng, scale, cx, cy):
    h, w = canvas.shape[:2]
    col = rng.uniform(60, 230, size=3)
    ys, xs = np.mgrid[0:h, 0:w]
    d = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
    m = d <= scale
    lit = np.clip(1.15 - d / (scale + 1e-6) * 0.55, 0.35, 1.2)[m][:, None]
    canvas[m] = np.clip(col * lit, 0, 255)


def draw_cylinder(canvas, rng, scale, cx, cy, rot):
    col = rng.uniform(60, 230, size=3)
    n = 20
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    r, hh = 0.9, rng.uniform(0.8, 1.4)
    top = np.stack([r * np.cos(th), np.full(n, hh), r * np.sin(th)], 1) @ rot.T
    bot = np.stack([r * np.cos(th), np.full(n, -hh), r * np.sin(th)], 1) @ rot.T
    pt, pb = project(top, scale, cx, cy), project(bot, scale, cx, cy)
    for i in range(n):
        j = (i + 1) % n
        quad = np.array([pt[i], pt[j], pb[j], pb[i]])
        fill_convex(canvas, quad, np.clip(col * (0.6 + 0.5 * np.cos(th[i])), 0, 255))
    fill_convex(canvas, pt, np.clip(col * 1.15, 0, 255))


def draw_stripes(canvas, rng, scale, cx, cy, rot):
    """Flat coloured bands -- lots of colour, no cube structure."""
    k = rng.integers(3, 7)
    cols = STICKERS[rng.choice(len(STICKERS), min(k, 6), replace=False)]
    quad = project(np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]],
                            dtype=np.float64) @ rot.T, scale, cx, cy)
    for i in range(k):
        fill_convex(canvas, bilinear(quad, 0, 1, i / k, (i + 1) / k),
                    cols[i % len(cols)])


def draw_grid_flat(canvas, rng, scale, cx, cy, rot):
    """A flat 3x3 patch of colours with no gaps and no 3D body -- the
    nastiest negative, because it has the right grid and nothing else."""
    quad = project(np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]],
                            dtype=np.float64) @ rot.T, scale, cx, cy)
    for i in range(3):
        for j in range(3):
            col = STICKERS[rng.integers(len(STICKERS))]
            fill_convex(canvas, bilinear(quad, i / 3, (i + 1) / 3,
                                         j / 3, (j + 1) / 3), col)


# ------------------------------------------------------------- backgrounds

def background(canvas, rng):
    """Textured, never flat. See vision/augment.py for why that matters."""
    textured_background(canvas, rng)


# ------------------------------------------------------------------- sample

PUZZLE_KINDS = ["cube2", "cube3", "cube4", "cube5", "pyraminx", "skewb",
                "megaminx"]
DISTRACTORS = ["plain", "checker", "sphere", "cylinder", "stripes",
               "gridflat", "empty"]
KINDS = PUZZLE_KINDS + DISTRACTORS

# What the model is asked to output. Class 0 is everything that is not a
# twisty puzzle; after that, one class per puzzle.
CLASSES = ["not a puzzle", "2x2 cube", "3x3 Rubik's cube", "4x4 cube",
           "5x5 cube", "Pyraminx", "Skewb", "Megaminx"]
LABEL = {k: i + 1 for i, k in enumerate(PUZZLE_KINDS)}
LABEL.update({k: 0 for k in DISTRACTORS})

# The legacy binary task: is it a 3x3 Rubik's cube or not.
POSITIVE = {"cube3"}


def render(rng, kind, size=SIZE, ss=SUPERSAMPLE, hard=True,
           scale_range=(0.20, 0.42)):
    """One image of the requested kind. Returns uint8 (size, size, 3).

    `hard` turns on the full mess -- textured backgrounds, clutter,
    occlusion, and a simulated camera. Off, you get the clean renders, which
    are useful for looking at but much too easy to learn from.
    """
    n = size * ss
    canvas = np.zeros((n, n, 3), dtype=np.float64)
    background(canvas, rng)
    if hard and rng.random() < 0.55:
        clutter(canvas, rng, fill_convex)

    scale = rng.uniform(*scale_range) * n
    cx = n / 2 + rng.uniform(-0.14, 0.14) * n
    cy = n / 2 + rng.uniform(-0.14, 0.14) * n
    rot = rotation(rng.uniform(0, 2 * np.pi),
                   rng.uniform(-0.9, 0.9),
                   rng.uniform(-0.5, 0.5))

    if kind in PUZZLE_SPECS:
        draw_puzzle(canvas, rng, kind, scale, cx, cy, rot)
    elif kind == "plain":
        draw_box(canvas, rng, "plain", scale, cx, cy, rot)
    elif kind == "checker":
        draw_box(canvas, rng, "checker", scale, cx, cy, rot)
    elif kind == "sphere":
        draw_sphere(canvas, rng, scale, cx, cy)
    elif kind == "cylinder":
        draw_cylinder(canvas, rng, scale, cx, cy, rot)
    elif kind == "stripes":
        draw_stripes(canvas, rng, scale, cx, cy, rot)
    elif kind == "gridflat":
        draw_grid_flat(canvas, rng, scale, cx, cy, rot)
    elif kind == "empty":
        pass
    else:
        raise ValueError(kind)

    if hard:
        occlude(canvas, rng, fill_convex)

    # box-filter down for anti-aliasing
    img = canvas.reshape(size, ss, size, ss, 3).mean(axis=(1, 3))

    if hard:
        img = camera(img, rng)       # optics, sensor, compression
    else:
        img = np.clip(img + rng.normal(0, 3, img.shape), 0, 255)
    return np.clip(img, 0, 255).astype(np.uint8)


def dataset(n, seed=0, size=SIZE, p_positive=0.62, hard=True,
            task="puzzle", scale_range=(0.20, 0.42)):
    """n images. Returns (images uint8, labels int, kinds int).

    task="puzzle" -> 8 classes, one per puzzle plus "not a puzzle"
    task="binary" -> the original: is this a 3x3 Rubik's cube or not

    `p_positive` is how much of the batch is a puzzle at all. The rest are
    distractors, weighted towards the ones that are genuinely hard: a plain
    box has the right shape and no stickers, a checkerboard box has a grid
    but the wrong one, and a flat 3x3 colour patch has the stickers but no
    solid underneath.
    """
    rng = np.random.default_rng(seed)
    if task == "binary":
        positives = ["cube3"]
    else:
        positives = PUZZLE_KINDS
    weights = np.array([3.0 if k in ("plain", "checker", "gridflat") else 1.0
                        for k in DISTRACTORS])
    weights /= weights.sum()

    imgs = np.zeros((n, size, size, 3), dtype=np.uint8)
    labels = np.zeros(n, dtype=np.int64)
    kinds = np.zeros(n, dtype=np.int64)
    for i in range(n):
        if rng.random() < p_positive:
            kind = positives[rng.integers(len(positives))]
        else:
            kind = DISTRACTORS[rng.choice(len(DISTRACTORS), p=weights)]
        imgs[i] = render(rng, kind, size, hard=hard,
                         scale_range=scale_range)
        if task == "binary":
            labels[i] = 1 if kind in POSITIVE else 0
        else:
            labels[i] = LABEL[kind]
        kinds[i] = KINDS.index(kind)
    return imgs, labels, kinds
