"""
Creates simple synthetic test images so every topic can be tried without a dataset.
For real results use real datasets (e.g. BCCD for blood cells, Kaggle fruit-ripeness sets).

    python samples/generate_samples.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core import utils  # noqa: E402

OUT = Path(__file__).resolve().parent


def _finish(img: np.ndarray, rng: np.random.Generator, noise: float = 4.0) -> np.ndarray:
    img = cv2.GaussianBlur(img, (3, 3), 0).astype(np.float32)
    img += rng.normal(0, noise, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def blood_smear(n_rbc: int = 70, n_wbc: int = 2, n_platelets: int = 10, seed: int = 7):
    """Pink RBC discs with pale centres, purple multi-lobed WBC nuclei, tiny purple platelets."""
    rng = np.random.default_rng(seed)
    h, w = 480, 640
    img = np.full((h, w, 3), (228, 214, 232), np.uint8)

    wbc_centres = [(int(rng.integers(80, w - 80)), int(rng.integers(80, h - 80))) for _ in range(n_wbc)]
    rbcs: list[tuple[int, int, int]] = []
    while len(rbcs) < n_rbc:
        r = int(rng.integers(17, 22))
        x, y = int(rng.integers(r, w - r)), int(rng.integers(r, h - r))
        too_close = any(np.hypot(x - a, y - b) < 1.35 * r for a, b, _ in rbcs)
        on_wbc = any(np.hypot(x - a, y - b) < 55 for a, b in wbc_centres)
        if not too_close and not on_wbc:
            rbcs.append((x, y, r))
    for x, y, r in rbcs:
        cv2.circle(img, (x, y), r, (150, 118, 205), -1, cv2.LINE_AA)
        cv2.circle(img, (x, y), r, (128, 96, 185), 2, cv2.LINE_AA)
        cv2.circle(img, (x, y), int(r * 0.45), (178, 160, 222), -1, cv2.LINE_AA)

    for x, y in wbc_centres:
        cv2.circle(img, (x, y), 34, (215, 186, 222), -1, cv2.LINE_AA)          # cytoplasm
        for dx, dy in ((-10, -6), (8, -8), (2, 10)):                              # 3 nucleus lobes
            cv2.ellipse(img, (x + dx, y + dy), (13, 10), int(rng.integers(0, 180)), 0, 360,
                        (150, 45, 110), -1, cv2.LINE_AA)

    platelets = 0
    while platelets < n_platelets:
        x, y = int(rng.integers(5, w - 5)), int(rng.integers(5, h - 5))
        if all(np.hypot(x - a, y - b) > r + 6 for a, b, r in rbcs) and \
           all(np.hypot(x - a, y - b) > 45 for a, b in wbc_centres):
            cv2.circle(img, (x, y), int(rng.integers(3, 5)), (160, 60, 120), -1, cv2.LINE_AA)
            platelets += 1
    return _finish(img, rng), {"RBC": n_rbc, "WBC": n_wbc, "Platelets": n_platelets}


def _shaded(mask: np.ndarray, color: tuple[int, int, int], rng: np.random.Generator) -> np.ndarray:
    """Fill a mask with a colour that gets darker towards the bottom (fake 3-D lighting)."""
    h, w = mask.shape
    shade = np.linspace(1.12, 0.78, h)[:, None, None]
    layer = np.clip(np.array(color, np.float32) * shade * rng.uniform(0.95, 1.05), 0, 255)
    return np.broadcast_to(layer, (h, w, 3))


def fruit(kind: str, state: str, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    h, w = 420, 560
    img = np.full((h, w, 3), 236, np.float32)
    shadow = np.zeros((h, w), np.uint8)
    mask = np.zeros((h, w), np.uint8)

    if kind == "banana":
        cv2.ellipse(shadow, (285, 140), (215, 150), 0, 25, 155, 255, 60)
        cv2.ellipse(mask, (280, 120), (215, 150), 0, 25, 155, 255, 64)
        colours = {"raw": (60, 170, 90), "ripe": (40, 200, 235), "rotten": (30, 75, 115)}
    else:
        cv2.ellipse(shadow, (290, 225), (150, 105), -20, 0, 360, 255, -1)
        cv2.ellipse(mask, (280, 210), (150, 105), -20, 0, 360, 255, -1)
        colours = {"raw": (50, 160, 80), "ripe": (40, 185, 240), "rotten": (35, 80, 120)}

    shadow = cv2.GaussianBlur(shadow, (41, 41), 0).astype(np.float32) / 255
    img *= (1 - 0.18 * shadow)[:, :, None]
    body = _shaded(mask, colours[state], rng)
    m = (mask > 0)[:, :, None]
    img = np.where(m, body, img)

    if kind == "mango" and state == "ripe":       # red-orange blush on one side
        blush = np.zeros((h, w), np.float32)
        cv2.circle(blush, (220, 160), 90, 1.0, -1)
        blush = cv2.GaussianBlur(blush, (0, 0), 35)[:, :, None] * m
        img = img * (1 - 0.6 * blush) + np.array((40, 90, 225), np.float32) * 0.6 * blush

    spots = {"raw": 0, "ripe": 6, "rotten": 45}[state]
    ys, xs = np.nonzero(mask)
    for _ in range(spots):
        i = rng.integers(len(xs))
        r = int(rng.integers(4, 9 if state == "ripe" else 22))
        cv2.circle(img, (int(xs[i]), int(ys[i])), r, (20, 35, 55), -1, cv2.LINE_AA)
    return _finish(img.astype(np.uint8), rng, noise=3.0)


def fractal_noise(h: int, w: int, rng: np.random.Generator, octaves: int = 5,
                  persistence: float = 0.55) -> np.ndarray:
    """Sum of up-scaled random noise at several scales (cheap Perlin-like texture), 0..1."""
    out = np.zeros((h, w), np.float32)
    amp, total = 1.0, 0.0
    for o in range(octaves):
        cell = 2 ** (octaves - o + 1)                      # coarse -> fine
        small = rng.random((h // cell + 2, w // cell + 2)).astype(np.float32)
        out += amp * cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
        total += amp
        amp *= persistence
    return cv2.normalize(out / total, None, 0, 1, cv2.NORM_MINMAX)


# ----- Topic 24: drone view of a field -------------------------------------------------
def _rosette(img, centre, radius, colour, rng, leaves=7):
    for i in range(leaves):
        a = 2 * np.pi * i / leaves + rng.uniform(-0.2, 0.2)
        c = (int(centre[0] + 0.55 * radius * np.cos(a)), int(centre[1] + 0.55 * radius * np.sin(a)))
        shade = np.clip(np.array(colour) * rng.uniform(0.85, 1.15), 0, 255).tolist()
        cv2.ellipse(img, c, (int(radius * 0.6), max(2, int(radius * 0.28))), np.degrees(a), 0, 360,
                    shade, -1, cv2.LINE_AA)


def drone_field(seed: int = 3, rows: int = 6, angle_deg: float = 12, n_weeds: int = 28):
    """Brown soil, tilted crop rows of leafy rosettes, small weeds scattered between rows."""
    rng = np.random.default_rng(seed)
    h, w = 600, 800
    soil = np.array((55, 85, 125), np.float32) * (0.75 + 0.5 * fractal_noise(h, w, rng))[:, :, None]
    img = np.clip(soil + rng.normal(0, 6, (h, w, 3)), 0, 255).astype(np.uint8)

    t = np.radians(angle_deg)
    along, normal = np.array([np.sin(t), np.cos(t)]), np.array([np.cos(t), -np.sin(t)])
    centre, spacing = np.array([w / 2, h / 2]), w / rows
    offsets = [(k - (rows - 1) / 2) * spacing for k in range(rows)]
    crops = 0
    for d in offsets:
        for s in np.arange(-500, 500, 48):
            p = centre + d * normal + (s + rng.uniform(-6, 6)) * along
            if 18 < p[0] < w - 18 and 18 < p[1] < h - 18:
                _rosette(img, p, rng.uniform(15, 19), (40, 150, 60), rng)
                crops += 1

    weeds = 0
    while weeds < n_weeds:
        p = np.array([rng.uniform(10, w - 10), rng.uniform(10, h - 10)])
        if min(abs((p - centre) @ normal - d) for d in offsets) > 0.3 * spacing:
            for _ in range(int(rng.integers(2, 5))):
                q = p + rng.normal(0, 4, 2)
                cv2.ellipse(img, (int(q[0]), int(q[1])), (int(rng.integers(3, 8)), int(rng.integers(2, 5))),
                            float(rng.uniform(0, 180)), 0, 360, (50, 165, 115), -1, cv2.LINE_AA)
            weeds += 1
    return _finish(img, rng, noise=2.0), {"crop plants": crops, "weeds": n_weeds, "rows": rows}


# ----- Topic 12: small image database with 8 classes -----------------------------------
CBIR_CLASSES = ("sunset", "forest", "sea", "bricks", "grass", "night", "desert", "tiles")


def _vgradient(h, w, top, bottom):
    t = np.linspace(0, 1, h, dtype=np.float32)[:, None, None]
    return np.broadcast_to((1 - t) * np.array(top, np.float32) + t * np.array(bottom, np.float32), (h, w, 3)).copy()


def cbir_image(cls: str, rng: np.random.Generator, size: int = 160) -> np.ndarray:
    s = size
    jitter = lambda c: np.clip(np.array(c) * rng.uniform(0.85, 1.15, 3), 0, 255).tolist()
    if cls == "sunset":
        img = _vgradient(s, s, jitter((60, 120, 250)), jitter((110, 50, 140)))
        cv2.circle(img, (int(rng.integers(30, s - 30)), int(s * rng.uniform(0.35, 0.55))),
                   int(rng.integers(12, 22)), (90, 230, 255), -1, cv2.LINE_AA)
        hills = (s * (0.65 + 0.15 * fractal_noise(1, s, rng, octaves=3)[0])).astype(np.int32)
        pts = np.array([[0, s]] + [[x, hills[x]] for x in range(s)] + [[s - 1, s]], np.int32)
        cv2.fillPoly(img, [pts], (25, 20, 35))
    elif cls == "forest":
        img = _vgradient(s, s, jitter((70, 110, 60)), jitter((30, 70, 30)))
        for _ in range(int(rng.integers(25, 45))):
            x, y, r = int(rng.integers(0, s)), int(rng.integers(10, s)), int(rng.integers(8, 20))
            cv2.fillPoly(img, [np.array([[x, y - 2 * r], [x - r, y], [x + r, y]], np.int32)], jitter((30, 90, 30)))
    elif cls == "sea":
        img = _vgradient(s, s, jitter((190, 130, 50)), jitter((110, 55, 10)))
        freq, phase = rng.uniform(0.05, 0.12), rng.uniform(0, 6)
        for y0 in range(8, s, int(rng.integers(10, 16))):
            xs = np.arange(s)
            ys = (y0 + 3 * np.sin(freq * xs + phase + y0)).astype(np.int32)
            cv2.polylines(img, [np.column_stack([xs, ys]).astype(np.int32)], False, (235, 225, 210), 1, cv2.LINE_AA)
    elif cls == "bricks":
        img = np.full((s, s, 3), jitter((165, 170, 175)), np.float32)
        bw, bh = int(rng.integers(28, 40)), int(rng.integers(11, 16))
        for row, y in enumerate(range(0, s, bh + 3)):
            for x in range(-bw if row % 2 else -bw // 2, s, bw + 3):
                cv2.rectangle(img, (x, y), (x + bw, y + bh), jitter((45, 65, 150)), -1)
    elif cls == "grass":
        img = np.array(jitter((35, 135, 60)), np.float32) * (0.7 + 0.5 * fractal_noise(s, s, rng))[:, :, None]
        for _ in range(250):
            x, y = int(rng.integers(0, s)), int(rng.integers(0, s))
            cv2.line(img, (x, y), (x + int(rng.integers(-3, 4)), y - int(rng.integers(6, 14))),
                     jitter((40, 170, 80)), 1, cv2.LINE_AA)
    elif cls == "night":
        img = _vgradient(s, s, jitter((60, 25, 15)), jitter((25, 10, 5)))
        for _ in range(int(rng.integers(60, 150))):
            img[int(rng.integers(0, s)), int(rng.integers(0, s))] = (255, 255, 255)
        cv2.circle(img, (int(rng.integers(20, s - 20)), int(rng.integers(20, s // 2))),
                   int(rng.integers(8, 15)), (215, 225, 235), -1, cv2.LINE_AA)
    elif cls == "desert":
        img = _vgradient(s, s, jitter((110, 185, 225)), jitter((70, 150, 200)))
        sky = int(s * rng.uniform(0.2, 0.35))
        img[:sky] = jitter((235, 205, 160))
        xs = np.arange(s)
        for k in range(3):
            ys = (sky + 20 + 30 * k + 8 * np.sin(rng.uniform(0.02, 0.06) * xs + rng.uniform(0, 6))).astype(np.int32)
            cv2.polylines(img, [np.column_stack([xs, ys]).astype(np.int32)], False, jitter((60, 120, 170)), 2, cv2.LINE_AA)
    elif cls == "tiles":
        tile = int(rng.integers(14, 30))
        a, b = jitter((200, 200, 205)), jitter((80, 85, 95))
        yy, xx = np.mgrid[0:s, 0:s]
        board = ((yy // tile + xx // tile) % 2).astype(bool)
        img = np.where(board[:, :, None], np.array(a, np.float32), np.array(b, np.float32))
    else:
        raise ValueError(cls)
    return _finish(np.clip(img, 0, 255).astype(np.uint8), rng, noise=3.0)


# ----- Topic 18: a textured scene and a copy-move forged version ------------------------
def copy_move_scene(seed: int = 11):
    """Returns (authentic, forged, ground-truth mask with BOTH copied and pasted regions)."""
    rng = np.random.default_rng(seed)
    h, w = 420, 600
    sky_h = int(h * 0.38)
    img = np.zeros((h, w, 3), np.float32)
    img[:sky_h] = _vgradient(sky_h, w, (235, 190, 140), (250, 225, 200))
    img[:sky_h] *= (0.92 + 0.12 * fractal_noise(sky_h, w, rng))[:, :, None]
    ground = fractal_noise(h - sky_h, w, rng, octaves=6)
    img[sky_h:] = np.array((50, 120, 90)) * (0.55 + 0.9 * ground)[:, :, None]

    def tree(canvas, x, y, r, layer=None):
        cv2.rectangle(canvas, (x - 3, y), (x + 3, y + r), (30, 60, 90), -1)
        if layer is not None:
            cv2.rectangle(layer, (x - 3, y), (x + 3, y + r), 255, -1)
        for _ in range(12):
            c = (x + int(rng.normal(0, r * 0.5)), y - int(abs(rng.normal(0, r * 0.4))))
            rad = int(r * rng.uniform(0.3, 0.6))
            cv2.circle(canvas, c, rad, np.clip(np.array((35, 100, 45)) * rng.uniform(0.6, 1.3), 0, 255).tolist(), -1, cv2.LINE_AA)
            if layer is not None:
                cv2.circle(layer, c, rad, 255, -1)

    for _ in range(8):                                     # bushes / trees with texture
        tree(img, int(rng.integers(30, w - 30)), int(rng.integers(sky_h + 20, h - 20)), int(rng.integers(14, 30)))
    for _ in range(5):                                     # rocks
        x, y = int(rng.integers(20, w - 20)), int(rng.integers(sky_h + 10, h - 10))
        cv2.ellipse(img, (x, y), (int(rng.integers(10, 22)), int(rng.integers(6, 12))), 0, 0, 360,
                    (120, 125, 130), -1, cv2.LINE_AA)
    obj = np.zeros((h, w), np.uint8)                       # the tree that will be copied
    tree(img, 120, sky_h + 120, 30, layer=obj)
    authentic = _finish(np.clip(img, 0, 255).astype(np.uint8), rng, noise=3.0)

    # copy the tree (+ a thin ring of ground around it) to another place
    src = cv2.dilate(obj, np.ones((9, 9), np.uint8)) > 0
    dx, dy = 340, -25
    dst = np.roll(np.roll(src, dy, axis=0), dx, axis=1)
    forged = authentic.copy()
    forged[dst] = authentic[src]
    gt = ((src | dst) * 255).astype(np.uint8)
    return authentic, forged, gt


#  ----- Topic 13: road with potholes and cracks ------------------------------------------
def road_scene(seed: int = 21, n_potholes: int = 4, n_cracks: int = 3):
    rng = np.random.default_rng(seed)
    h, w = 540, 720
    base = 100 * (0.85 + 0.3 * fractal_noise(h, w, rng))
    grain = cv2.GaussianBlur(rng.normal(0, 14, (h, w)).astype(np.float32), (3, 3), 0)
    img = np.repeat(np.clip(base + grain, 0, 255)[:, :, None], 3, axis=2)
    img *= np.array([1.0, 1.0, 1.04])                     # asphalt is slightly warm gray
    for y in range(-20, h, 110):                          # dashed lane marking
        cv2.rectangle(img, (w // 2 - 6, y), (w // 2 + 6, y + 60), (215, 220, 225), -1)

    centres = []
    while len(centres) < n_potholes:
        c = np.array([rng.uniform(70, w - 70), rng.uniform(70, h - 70)])
        if abs(c[0] - w / 2) > 70 and all(np.hypot(*(c - o)) > 150 for o in centres):
            centres.append(c)
    for c in centres:
        r = rng.uniform(28, 60)
        angles = np.linspace(0, 2 * np.pi, 40, endpoint=False)
        radii = r * (1 + 0.25 * np.convolve(rng.normal(0, 1, 40), np.ones(5) / 5, mode="same"))
        pts = np.column_stack([c[0] + radii * np.cos(angles), c[1] + radii * 0.75 * np.sin(angles)]).astype(np.int32)
        cv2.polylines(img, [pts], True, (125, 125, 130), 5, cv2.LINE_AA)          # broken lighter rim
        hole = np.zeros((h, w), np.uint8)
        cv2.fillPoly(hole, [pts], 255)
        inside = hole > 0
        img[inside] = (40 + 30 * fractal_noise(h, w, rng)[inside] + rng.normal(0, 6, inside.sum()))[:, None]

    for _ in range(n_cracks):                             # random-walk cracks with a branch
        while True:
            p = np.array([rng.uniform(40, w - 40), rng.uniform(40, h - 40)])
            if abs(p[0] - w / 2) > 40 and all(np.hypot(*(p - o)) > 110 for o in centres):
                break
        heading = rng.uniform(0, 2 * np.pi)
        for branch in range(2):
            q, pts = p.copy(), [p.copy()]
            for _ in range(int(rng.integers(30, 55)) // (branch + 1)):
                heading += rng.normal(0, 0.35)
                q = q + 5 * np.array([np.cos(heading), np.sin(heading)])
                pts.append(q.copy())
            cv2.polylines(img, [np.array(pts, np.int32)], False, (38, 38, 40), int(rng.integers(1, 3)), cv2.LINE_AA)
            heading += rng.uniform(1.0, 2.0)
    return _finish(np.clip(img, 0, 255).astype(np.uint8), rng, noise=2.0), \
        {"potholes": n_potholes, "cracks": n_cracks}


# ----- Topic 16: an old, damaged black & white photo ------------------------------------
def old_photo(seed: int = 31):
    """Returns (damaged old photo, clean colour original, colour reference for colorization)."""
    original = copy_move_scene(seed=seed)[0]
    reference = copy_move_scene(seed=seed + 1)[0]
    return age_photo(original, seed), original, reference


def age_photo(original: np.ndarray, seed: int = 31) -> np.ndarray:
    """Make any photo look like an old damaged print: faded B&W, grain, scratches, dust, a crease, sepia."""
    rng = np.random.default_rng(seed)
    h, w = original.shape[:2]
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY).astype(np.float32)
    faded = gray * 0.55 + 70                               # low contrast, lifted blacks
    faded += rng.normal(0, 7, faded.shape)                 # film grain
    for _ in range(14):                                    # bright scratches (thin, long, wavy)
        x, y0 = rng.uniform(0, w), rng.uniform(-50, h * 0.5)
        ys = np.arange(int(y0), int(y0 + rng.uniform(80, 300)))
        xs = x + np.cumsum(rng.normal(0, 0.6, len(ys)))
        pts = np.column_stack([xs, ys]).astype(np.int32)
        cv2.polylines(faded, [pts], False, float(rng.uniform(215, 250)), 1, cv2.LINE_AA)
    for _ in range(4):                                     # dark scratches
        p = rng.uniform((0, 0), (w, h))
        q = p + rng.normal(0, 90, 2)
        cv2.line(faded, tuple(p.astype(int)), tuple(q.astype(int)), 25, 1, cv2.LINE_AA)
    for _ in range(120):                                   # dust specks
        cv2.circle(faded, (int(rng.uniform(0, w)), int(rng.uniform(0, h))), int(rng.integers(1, 3)),
                   float(rng.uniform(200, 250)), -1, cv2.LINE_AA)
    cv2.line(faded, (0, int(h * 0.7)), (w, int(h * 0.62)), 240, 2, cv2.LINE_AA)   # a fold crease

    g = np.clip(faded, 0, 255) / 255.0                     # sepia / yellowed paper tint
    damaged = np.stack([g * 0.78, g * 0.92, g * 1.0], axis=2) * 255
    return np.clip(damaged, 0, 255).astype(np.uint8)


# ----- Topic 17: a subject in front of a room wall / a green screen ------------------------
def subject_scene(green_screen: bool = False, seed: int = 41):
    """Returns (image, ground-truth foreground mask). The subject touches the bottom edge."""
    rng = np.random.default_rng(seed)
    h, w = 480, 640
    if green_screen:
        img = _vgradient(h, w, (70, 205, 80), (45, 170, 55))
    else:
        img = _vgradient(h, w, (175, 205, 225), (140, 170, 195))
        img *= (0.9 + 0.15 * fractal_noise(h, w, rng))[:, :, None]
        img[int(h * 0.78):] = np.array((70, 95, 130)) * (0.85 + 0.3 * fractal_noise(h - int(h * 0.78), w, rng))[:, :, None]
        cv2.rectangle(img, (40, 60), (170, 190), (60, 60, 70), 6)                 # picture frame
        cv2.rectangle(img, (46, 66), (164, 184), (150, 120, 90), -1)
        cv2.circle(img, (560, 330), 35, (60, 140, 70), -1)                         # pot plant
        cv2.rectangle(img, (540, 360), (580, 420), (50, 80, 150), -1)

    mask = np.zeros((h, w), np.uint8)
    parts = [  # (draw function, colour)
        (lambda c, col: cv2.ellipse(c, (320, 380), (120, 150), 0, 180, 360, col, -1), (170, 90, 40)),   # shirt
        (lambda c, col: cv2.rectangle(c, (200, 380), (440, h), col, -1), (170, 90, 40)),
        (lambda c, col: cv2.rectangle(c, (295, 215), (345, 260), col, -1), (125, 160, 210)),            # neck
        (lambda c, col: cv2.ellipse(c, (320, 170), (62, 75), 0, 0, 360, col, -1), (135, 170, 220)),     # face
        (lambda c, col: cv2.ellipse(c, (320, 130), (66, 48), 0, 180, 360, col, -1), (40, 40, 55)),      # hair
    ]
    for draw, colour in parts:
        draw(img, colour)
        draw(mask, 255)
    for x in (295, 345):                                                           # eyes
        cv2.circle(img, (x, 170), 6, (40, 30, 30), -1)
    cv2.ellipse(img, (320, 205), (18, 8), 0, 0, 180, (60, 60, 160), 2)            # smile
    cv2.line(img, (320, 290), (320, h), (140, 70, 30), 2)                          # shirt seam
    return _finish(np.clip(img, 0, 255).astype(np.uint8), rng, noise=3.0), mask


# ----- Topic 1: aerial view of an orchard + a dense patch ----------------------------------
def _tree_crown(img, centre, r, rng):
    x, y = int(centre[0]), int(centre[1])
    shadow = np.zeros(img.shape[:2], np.float32)
    cv2.circle(shadow, (x + int(r * 0.35), y + int(r * 0.3)), int(r), 1.0, -1)
    shadow = cv2.GaussianBlur(shadow, (0, 0), r * 0.25)[:, :, None]
    img *= 1 - 0.45 * shadow
    cv2.circle(img, (x, y), int(r), (35, 100, 45), -1, cv2.LINE_AA)
    for _ in range(int(r)):                                # leaf clumps give texture
        a, d = rng.uniform(0, 2 * np.pi), r * np.sqrt(rng.uniform(0, 0.8))
        light = 1.25 if np.cos(a + 2.4) > 0 else 0.85       # lit from the top-left
        colour = np.clip(np.array((40, 115, 50)) * rng.uniform(0.75, 1.2) * light, 0, 255).tolist()
        cv2.circle(img, (int(x + d * np.cos(a)), int(y + d * np.sin(a))), int(r * rng.uniform(0.15, 0.3)),
                   colour, -1, cv2.LINE_AA)


def aerial_orchard(seed: int = 51):
    rng = np.random.default_rng(seed)
    h, w = 600, 800
    img = np.array((85, 145, 160), np.float32) * (0.8 + 0.35 * fractal_noise(h, w, rng))[:, :, None]
    centres = []
    for row in range(5):                                   # orchard grid with a few gaps
        for col in range(6):
            if rng.random() < 0.1:
                continue
            centres.append((70 + col * 95 + rng.uniform(-8, 8), 70 + row * 100 + rng.uniform(-8, 8), rng.uniform(22, 32)))
    patch = []                                             # dense patch on the right: crowns touch
    while len(patch) < 10:
        c = (rng.uniform(640, 770), rng.uniform(60, 540), rng.uniform(20, 28))
        if all(np.hypot(c[0] - p[0], c[1] - p[1]) > 1.3 * max(c[2], p[2]) for p in patch):
            patch.append(c)
    for x, y, r in sorted(centres + patch, key=lambda c: c[1]):
        _tree_crown(img, (x, y), r, rng)
    return _finish(np.clip(img, 0, 255).astype(np.uint8), rng, noise=3.0), {"trees": len(centres) + len(patch)}


# ----- Topic 3: flowers --------------------------------------------------------------
FLOWER_CLASSES = ("daisy", "rose", "sunflower", "tulip")


def _foliage(s, rng):
    g = fractal_noise(s, s, rng)
    return np.array((40, 110, 50), np.float32) * (0.5 + 0.8 * g)[:, :, None]


def flower_image(cls: str, rng: np.random.Generator, size: int = 256) -> np.ndarray:
    s = size
    img = _foliage(s, rng)
    c = np.array([s / 2, s / 2]) + rng.normal(0, s * 0.04, 2)
    R = s * rng.uniform(0.3, 0.4)
    rot = rng.uniform(0, 2 * np.pi)
    jit = lambda col, k=0.1: np.clip(np.array(col) * rng.uniform(1 - k, 1 + k, 3), 0, 255).tolist()
    pt = lambda a, d: (int(c[0] + d * np.cos(a)), int(c[1] + d * np.sin(a)))
    if cls in ("daisy", "sunflower"):
        n = int(rng.integers(18, 25)) if cls == "daisy" else int(rng.integers(24, 32))
        petal, width, centre_r = ((235, 240, 245), 0.11, 0.22) if cls == "daisy" else ((10, 195, 245), 0.16, 0.42)
        for i in range(n):
            a = rot + 2 * np.pi * i / n
            cv2.ellipse(img, pt(a, R * 0.55), (int(R * 0.48), max(2, int(R * width))), np.degrees(a), 0, 360,
                        jit(petal, 0.06), -1, cv2.LINE_AA)
        centre_col = (20, 190, 240) if cls == "daisy" else (20, 45, 75)
        cv2.circle(img, pt(0, 0), int(R * centre_r), jit(centre_col), -1, cv2.LINE_AA)
        for _ in range(60 if cls == "sunflower" else 15):  # seeds / florets texture
            a, d = rng.uniform(0, 2 * np.pi), R * centre_r * np.sqrt(rng.random()) * 0.9
            cv2.circle(img, pt(a, d), 1, jit(centre_col, 0.4), -1)
    elif cls == "rose":
        base = [(45, 40, 200), (130, 100, 235), (70, 25, 165)][int(rng.integers(3))]
        for layer in range(6):                             # spiral of rounded petals, outside -> in
            k, d = 5 + layer, R * (0.75 - 0.12 * layer)
            for i in range(k):
                a = rot + 2 * np.pi * i / k + layer * 0.5
                shade = 1.0 - 0.07 * layer
                cv2.ellipse(img, pt(a, d * 0.6), (int(d * 0.55), int(d * 0.42)), np.degrees(a), 0, 360,
                            jit(np.array(base) * shade, 0.05), -1, cv2.LINE_AA)
                cv2.ellipse(img, pt(a, d * 0.6), (int(d * 0.55), int(d * 0.42)), np.degrees(a), 0, 360,
                            (np.array(base) * 0.6).tolist(), 1, cv2.LINE_AA)
    elif cls == "tulip":
        base = [(40, 30, 210), (20, 200, 240), (140, 40, 140)][int(rng.integers(3))]
        cx, cy = int(c[0]), int(c[1])
        cv2.line(img, (cx, cy + int(R * 0.5)), (cx + int(rng.normal(0, 8)), s), (40, 140, 60), 6, cv2.LINE_AA)
        cv2.ellipse(img, (cx + int(R * 0.45), cy + int(R * 1.1)), (int(R * 0.18), int(R * 0.9)), -20, 0, 360,
                    (45, 150, 65), -1, cv2.LINE_AA)        # leaf
        for dx, ang in ((-0.32, 12), (0.32, -12), (0.0, 0)):  # cup of 3 tall petals
            cv2.ellipse(img, (cx + int(dx * R), cy), (int(R * 0.38), int(R * 0.7)), ang, 0, 360,
                        jit(np.array(base) * (0.85 if dx else 1.0), 0.05), -1, cv2.LINE_AA)
        cv2.ellipse(img, (cx, cy - int(R * 0.55)), (int(R * 0.5), int(R * 0.2)), 0, 180, 360,
                    jit(np.array(base) * 0.7), -1, cv2.LINE_AA)
    return _finish(np.clip(img, 0, 255).astype(np.uint8), rng, noise=3.0)


# ----- Topic 21: leaves on white paper ----------------------------------------------------
LEAF_CLASSES = ("banyan", "mango", "neem", "peepal")


def leaf_image(cls: str, rng: np.random.Generator, size: int = 256) -> np.ndarray:
    s = size
    img = np.full((s, s, 3), 245, np.float32)
    t = np.linspace(0, 1, 240)
    if cls == "mango":
        L, w = 0.85, 0.15 * np.sin(np.pi * t) ** 0.8 * (1 - 0.3 * t)
    elif cls == "neem":
        tooth = np.abs(((t * 22) % 1) - 0.5) * 2           # saw-tooth edge
        L, w = 0.8, 0.13 * np.sin(np.pi * t) ** 0.9 * (0.88 + 0.16 * tooth) * (1 + 0.15 * (t < 0.5))
    elif cls == "banyan":
        L, w = 0.7, 0.30 * np.sin(np.pi * t) ** 0.55
    else:                                                  # peepal: broad heart-shaped base, long drip tip
        body = np.clip(np.where(t < 0.25, (t / 0.25) ** 0.35, 1 - (t - 0.25) / 0.6), 0, 1)
        L, w = 0.85, np.maximum(0.36 * body, np.where(t > 0.6, 0.04 * (1 - t) / 0.4, 0))
    L *= s * rng.uniform(0.9, 1.05)
    w = w * s * rng.uniform(0.9, 1.1)
    xs = (t - 0.5) * L
    poly = np.vstack([np.column_stack([xs, w]), np.column_stack([xs[::-1], -w[::-1]])])
    if cls == "neem":                                      # neem leaflets curve a little
        poly[:, 1] += 0.0006 * (poly[:, 0] ** 2) * rng.choice([-1, 1])
    a = rng.uniform(0, 2 * np.pi)
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    pts = (poly @ rot.T + s / 2).astype(np.int32)
    colour = np.clip(np.array((40, 120, 50)) * rng.uniform(0.75, 1.15, 3), 0, 255).tolist()
    shadow = np.zeros((s, s), np.float32)
    cv2.fillPoly(shadow, [pts + 4], 1.0)
    img *= (1 - 0.15 * cv2.GaussianBlur(shadow, (0, 0), 4))[:, :, None]
    cv2.fillPoly(img, [pts], colour, cv2.LINE_AA)
    base, tip = (np.array([-L / 2, 0]) @ rot.T + s / 2), (np.array([L / 2, 0]) @ rot.T + s / 2)
    stalk = (np.array([-L / 2 - s * 0.08, 0]) @ rot.T + s / 2)
    vein = np.clip(np.array(colour) * 1.35, 0, 255).tolist()
    cv2.line(img, tuple(stalk.astype(int)), tuple(base.astype(int)), np.array(colour).tolist(), 3, cv2.LINE_AA)
    veins = img.copy()
    cv2.line(veins, tuple(base.astype(int)), tuple(tip.astype(int)), vein, 2, cv2.LINE_AA)
    for k in np.linspace(0.15, 0.8, 7):                    # lateral veins
        p = base + (tip - base) * k
        for side in (1, -1):
            q = p + (rot @ np.array([0.12 * L, side * w[int(k * 239)] * 0.8]))
            cv2.line(veins, tuple(p.astype(int)), tuple(q.astype(int)), vein, 1, cv2.LINE_AA)
    leaf = np.zeros((s, s), np.uint8)
    cv2.fillPoly(leaf, [pts], 255)
    inside = cv2.erode(leaf, np.ones((5, 5), np.uint8)) > 0  # veins stay inside the blade
    img[inside] = veins[inside]
    return _finish(np.clip(img, 0, 255).astype(np.uint8), rng, noise=2.5)


# ----- Topic 6: synthetic fingerprints ------------------------------------------------------
def _fingerprint_master(seed: int, size: int = 420) -> np.ndarray:
    """Ridge pattern grown from noise by repeated oriented Gabor filtering (SFinGe-style).
    Orientation field = zero-pole model: one core + one delta (a 'loop' fingerprint)."""
    rng = np.random.default_rng(seed)
    ys, xs = np.mgrid[0:size, 0:size].astype(np.float32)
    z = xs + 1j * ys
    core = complex(rng.uniform(0.4, 0.6) * size, rng.uniform(0.35, 0.5) * size)
    delta = core + complex(rng.uniform(-0.25, 0.25) * size, rng.uniform(0.3, 0.4) * size)
    theta = 0.5 * (np.angle(z - core) - np.angle(z - delta)) + rng.uniform(-0.3, 0.3)
    bins = 16
    idx = np.round((theta % np.pi) / np.pi * bins).astype(int) % bins
    kernels = [cv2.getGaborKernel((21, 21), 3.5, k * np.pi / bins + np.pi / 2, 9.0, 1.0, 0) for k in range(bins)]
    img = rng.normal(0, 1, (size, size)).astype(np.float32)
    for _ in range(9):
        responses = np.stack([cv2.filter2D(img, -1, k) for k in kernels])
        img = np.take_along_axis(responses, idx[None], 0)[0]
        img = np.tanh(img / (img.std() + 1e-6) * 1.5)
    return img                                             # -1..1, ridges positive


def fingerprint_impression(master: np.ndarray, rng: np.random.Generator, size: int = 300) -> np.ndarray:
    """One 'scan' of a finger: rotation, shift, fingertip-shaped mask, pressure and noise."""
    m = master.shape[0]
    M = cv2.getRotationMatrix2D((m / 2, m / 2), rng.uniform(-12, 12), 1.0)
    M[:, 2] += rng.uniform(-14, 14, 2)
    warped = cv2.warpAffine(master, M, (m, m), borderValue=-1)
    o = (m - size) // 2
    ridges = warped[o:o + size, o:o + size]
    ridges = ridges + rng.uniform(-0.25, 0.25)                # dry / wet finger
    mask = np.zeros((size, size), np.float32)
    cv2.ellipse(mask, (size // 2 + int(rng.normal(0, 6)), size // 2 + 10), (int(size * 0.36), int(size * 0.46)),
                0, 0, 360, 1.0, -1)
    mask = cv2.GaussianBlur(mask, (0, 0), 6)
    gray = 235 - 180 * np.clip((ridges + 1) / 2, 0, 1) * mask
    gray += rng.normal(0, 8, gray.shape)
    return np.clip(cv2.GaussianBlur(gray, (3, 3), 0), 0, 255).astype(np.uint8)


# ----- Topic 7: a fictional demo bank note (NOT a real currency design) ----------------------
def demo_note(seed: int = 61) -> np.ndarray:
    rng = np.random.default_rng(seed)
    h, w = 300, 640
    img = _vgradient(h, w, (175, 205, 170), (150, 185, 205))
    img *= (0.95 + 0.08 * fractal_noise(h, w, rng))[:, :, None]
    for y in range(6, h, 4):                               # fine background wave lines (1 px)
        xs = np.arange(w)
        ys = (y + 1.5 * np.sin(xs / 9.0 + y / 13.0)).astype(np.int32)
        cv2.polylines(img, [np.column_stack([xs, ys]).astype(np.int32)], False, (150, 175, 150), 1, cv2.LINE_AA)
    cx, cy = 165, 150                                      # guilloche rosette (fine spirograph lines)
    t = np.linspace(0, 2 * np.pi, 3000)
    for k in range(18):
        r1, r2, d = 70, 70 / (7 + k % 3), 28 + k * 1.6
        x = (r1 - r2) * np.cos(t + k * 0.17) + d * np.cos((r1 - r2) / r2 * t)
        y = (r1 - r2) * np.sin(t + k * 0.17) - d * np.sin((r1 - r2) / r2 * t)
        cv2.polylines(img, [np.column_stack([cx + x, cy + y]).astype(np.int32)], True, (70, 110, 60), 1, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), 32, (60, 90, 50), 2, cv2.LINE_AA)
    star = np.array([[cx + 26 * np.cos(a) * (1 if i % 2 == 0 else 0.45), cy + 26 * np.sin(a) * (1 if i % 2 == 0 else 0.45)]
                     for i, a in enumerate(np.linspace(-np.pi / 2, 1.5 * np.pi, 11)[:-1])], np.int32)
    cv2.fillPoly(img, [star], (60, 90, 50), cv2.LINE_AA)
    img = np.clip(img, 0, 255).astype(np.uint8)            # OpenCV 5 draws text on 8-bit images only
    for y in (258, 268):                                   # microtext bands
        x = 30
        while x < w - 30:
            cv2.putText(img, "DIPSTUDIO", (x, y), cv2.FONT_HERSHEY_PLAIN, 0.55, (60, 80, 60), 1, cv2.LINE_AA)
            x += 52
    cv2.putText(img, "DIP STUDIO DEMO NOTE", (270, 55), cv2.FONT_HERSHEY_DUPLEX, 0.8, (50, 70, 50), 1, cv2.LINE_AA)
    cv2.putText(img, "SPECIMEN - NOT LEGAL TENDER", (270, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 160), 1, cv2.LINE_AA)
    cv2.putText(img, "500", (285, 200), cv2.FONT_HERSHEY_TRIPLEX, 2.2, (60, 95, 55), 3, cv2.LINE_AA)
    cv2.putText(img, "No. DS 0042137", (270, 235), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 40), 1, cv2.LINE_AA)
    x0, x1, y0, y1 = int(0.72 * w), int(0.92 * w), int(0.15 * h), int(0.85 * h)   # watermark window
    win = img[y0:y1, x0:x1].astype(np.float32)
    img[y0:y1, x0:x1] = (win * 0.3 + 0.7 * 238).astype(np.uint8)
    cv2.circle(img, ((x0 + x1) // 2, (y0 + y1) // 2), 38, (230, 230, 230), 3, cv2.LINE_AA)
    tx = int(0.62 * w)                                     # dashed security thread
    for y in range(0, h, 24):
        cv2.rectangle(img, (tx - 2, y), (tx + 2, y + 17), (45, 45, 50), -1)
    cv2.rectangle(img, (4, 4), (w - 5, h - 5), (60, 90, 60), 2)
    return _finish(np.clip(img, 0, 255).astype(np.uint8), rng, noise=1.5)


def note_variant(note: np.ndarray, kind: str, rng: np.random.Generator) -> np.ndarray:
    out = note.astype(np.float32)
    if kind == "fake_print":                               # scanned + printed: fine lines lost, colours off
        out = cv2.GaussianBlur(out, (0, 0), 1.6)
        hsv = cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] *= 0.6
        hsv[:, :, 0] = (hsv[:, :, 0] + 6) % 180
        out = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)
    elif kind == "fake_no_thread":                         # good copy, but no security thread
        tx = int(0.62 * note.shape[1])
        out[:, tx - 4:tx + 5] = cv2.GaussianBlur(out, (0, 0), 3)[:, tx - 4:tx + 5]
        out[:, tx - 4:tx + 5] = np.median(out[:, tx - 12:tx - 6], axis=1, keepdims=True)
        out = cv2.GaussianBlur(out, (0, 0), 0.7)
    elif kind == "fake_colour":                            # colour copier with wrong colour balance
        out = cv2.GaussianBlur(out, (0, 0), 0.9) * np.array([0.8, 1.0, 1.25])
    return np.clip(out, 0, 255).astype(np.uint8)


def photograph_note(note: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Put the note on a table and 'take a photo': perspective, rotation, uneven light, JPEG."""
    h, w = note.shape[:2]
    H, W = 520, 820
    table = np.array((60, 75, 95), np.float32) * (0.7 + 0.5 * fractal_noise(H, W, rng))[:, :, None]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    centre = np.array([W / 2, H / 2]) + rng.uniform(-30, 30, 2)
    a, s = np.radians(rng.uniform(-10, 10)), rng.uniform(0.95, 1.1)
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    dst = ((src - [w / 2, h / 2]) * s) @ rot.T + centre + rng.uniform(-12, 12, (4, 2))
    M = cv2.getPerspectiveTransform(src, dst.astype(np.float32))
    warped = cv2.warpPerspective(note, M, (W, H)).astype(np.float32)
    inside = cv2.warpPerspective(np.ones((h, w), np.float32), M, (W, H))[:, :, None]
    photo = warped * inside + table * (1 - inside)
    light = np.linspace(rng.uniform(0.85, 0.95), rng.uniform(1.0, 1.1), W)[None, :, None]
    photo = cv2.GaussianBlur(photo * light, (0, 0), 0.5) + rng.normal(0, 2.5, photo.shape)
    return np.clip(photo, 0, 255).astype(np.uint8)


# ----- Topic 5: front view of a car with a number plate ---------------------------------------
def car_with_plate(text: str, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    h, w = 480, 720
    img = _vgradient(h, w, (225, 205, 185), (200, 195, 190))                      # sky / building
    img[int(h * 0.62):] = np.array((95, 95, 100)) * (0.85 + 0.3 * fractal_noise(h - int(h * 0.62), w, rng))[:, :, None]
    body = np.clip(np.array([(40, 40, 160), (150, 150, 150), (120, 60, 20), (30, 30, 30), (230, 230, 230)][seed % 5])
                   * rng.uniform(0.9, 1.1), 0, 255).tolist()
    cx = w // 2 + int(rng.integers(-60, 60))
    cv2.rectangle(img, (cx - 230, 200), (cx + 230, 400), body, -1)                     # body
    cv2.fillPoly(img, [np.array([[cx - 170, 200], [cx + 170, 200], [cx + 130, 110], [cx - 130, 110]])], body)
    cv2.fillPoly(img, [np.array([[cx - 150, 195], [cx + 150, 195], [cx + 115, 122], [cx - 115, 122]])], (90, 70, 60))
    for dx in (-180, 180):                                                           # headlights, wheels
        cv2.ellipse(img, (cx + dx, 250), (38, 18), 0, 0, 360, (210, 240, 250), -1, cv2.LINE_AA)
        cv2.rectangle(img, (cx + dx - 35, 395), (cx + dx + 35, 440), (25, 25, 25), -1)
    cv2.rectangle(img, (cx - 110, 235), (cx + 110, 275), (40, 40, 40), -1)            # grille
    cv2.rectangle(img, (cx - 235, 330), (cx + 235, 345), (60, 60, 60), -1)            # bumper line
    pw, ph = 190, 52                                                                   # number plate
    x0, y0 = cx - pw // 2, 290
    cv2.rectangle(img, (x0, y0), (x0 + pw, y0 + ph), (245, 245, 245), -1)
    cv2.rectangle(img, (x0, y0), (x0 + pw, y0 + ph), (20, 20, 20), 2)
    img = np.clip(img, 0, 255).astype(np.uint8)
    font, scale = cv2.FONT_HERSHEY_DUPLEX, 1.15
    (tw, th), _ = cv2.getTextSize(text, font, scale, 2)
    cv2.putText(img, text, (cx - tw // 2, y0 + ph // 2 + th // 2), font, scale, (15, 15, 15), 2, cv2.LINE_AA)
    a = rng.uniform(-6, 6)                                                            # slight camera tilt
    M = cv2.getRotationMatrix2D((w / 2, h / 2), a, rng.uniform(0.92, 1.05))
    img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    return _finish(img, rng, noise=3.0)


# ----- Topics 4 / 8 / 11: public-domain test faces from scikit-image + drawn PPE --------------
def face_samples() -> None:
    """face_astronaut / face_cameraman (public domain, shipped with scikit-image) and copies of the
    astronaut with a drawn surgical mask / construction helmet. Needs the YuNet model in models/."""
    from skimage import data
    astronaut = cv2.cvtColor(data.astronaut(), cv2.COLOR_RGB2BGR)
    utils.write_image(OUT / "face_astronaut.png", astronaut)
    utils.write_image(OUT / "face_cameraman.png", data.camera())
    try:
        from core.faces import FaceDetector
        face = FaceDetector.shared().detect(astronaut, 0.7)[0]
    except (FileNotFoundError, IndexError):
        print("  (YuNet model missing - mask / helmet test faces skipped)")
        return
    x, y, w, h = face.box
    lm = face.landmarks.astype(int)
    masked = astronaut.copy()
    nose_y, chin_y = lm[2, 1] - int(0.08 * h), y + h + int(0.05 * h)
    pts = np.array([[x + int(0.05 * w), nose_y + 8], [x + int(0.5 * w), nose_y - 6], [x + int(0.95 * w), nose_y + 8],
                    [x + int(0.85 * w), chin_y - 10], [x + int(0.5 * w), chin_y], [x + int(0.15 * w), chin_y - 10]])
    cv2.fillPoly(masked, [pts], (225, 205, 160), cv2.LINE_AA)                     # light blue mask
    for k in range(1, 4):                                                          # pleats
        yy = nose_y + k * (chin_y - nose_y) // 4
        cv2.line(masked, (x + int(0.12 * w), yy), (x + int(0.88 * w), yy), (190, 170, 130), 1, cv2.LINE_AA)
    for side in (0, 1):                                                            # ear loops
        ex = x if side == 0 else x + w
        cv2.line(masked, (pts[0 if side == 0 else 2][0], nose_y + 8), (ex, lm[0, 1]), (235, 235, 235), 2, cv2.LINE_AA)
    utils.write_image(OUT / "face_astronaut_mask.png", masked)

    helmet = astronaut.copy()
    cx, top = x + w // 2, y + int(0.15 * h)
    cv2.ellipse(helmet, (cx, top), (int(0.62 * w), int(0.55 * h)), 0, 180, 360, (20, 200, 245), -1, cv2.LINE_AA)
    cv2.ellipse(helmet, (cx, top), (int(0.8 * w), int(0.1 * h)), 0, 0, 360, (15, 180, 230), -1, cv2.LINE_AA)  # brim
    cv2.line(helmet, (cx, top - int(0.55 * h)), (cx, top), (10, 160, 210), 4, cv2.LINE_AA)                  # ridge
    utils.write_image(OUT / "face_astronaut_helmet.png", helmet)


def main() -> None:
    smear, truth = blood_smear()
    utils.write_image(OUT / "blood_smear_synthetic.png", smear)
    print("blood_smear_synthetic.png  ground truth:", truth)
    for kind in ("banana", "mango"):
        for state in ("raw", "ripe", "rotten"):
            utils.write_image(OUT / f"{kind}_{state}_synthetic.png", fruit(kind, state))
            print(f"{kind}_{state}_synthetic.png")

    field, truth = drone_field()
    utils.write_image(OUT / "drone_field_synthetic.png", field)
    print("drone_field_synthetic.png  ground truth:", truth)

    rng = np.random.default_rng(42)
    for cls in CBIR_CLASSES:
        for i in range(10):
            utils.write_image(OUT / "cbir_db" / cls / f"{cls}_{i:02d}.jpg", cbir_image(cls, rng))
        utils.write_image(OUT / f"cbir_query_{cls}.jpg", cbir_image(cls, rng))
    print(f"cbir_db/  ({len(CBIR_CLASSES)} classes x 10 images) + cbir_query_*.jpg")

    authentic, forged, gt = copy_move_scene()
    utils.write_image(OUT / "copymove_authentic.png", authentic)
    utils.write_image(OUT / "copymove_forged.png", forged)
    cv2.imencode(".jpg", forged, [cv2.IMWRITE_JPEG_QUALITY, 75])[1].tofile(str(OUT / "copymove_forged_jpeg75.jpg"))
    utils.write_image(OUT / "copymove_ground_truth.png", gt)
    print("copymove_authentic.png, copymove_forged.png, copymove_forged_jpeg75.jpg, copymove_ground_truth.png")

    road, truth = road_scene()
    utils.write_image(OUT / "road_synthetic.png", road)
    print("road_synthetic.png  ground truth:", truth)

    damaged, original, reference = old_photo()
    utils.write_image(OUT / "old_photo_synthetic.png", damaged)
    utils.write_image(OUT / "old_photo_ground_truth.png", original)
    utils.write_image(OUT / "colour_reference_landscape.png", reference)
    print("old_photo_synthetic.png, old_photo_ground_truth.png, colour_reference_landscape.png")

    for green, name in ((False, "portrait_room"), (True, "portrait_greenscreen")):
        img, mask = subject_scene(green_screen=green)
        utils.write_image(OUT / f"{name}_synthetic.png", img)
        utils.write_image(OUT / f"{name}_mask.png", mask)
    print("portrait_room_synthetic.png, portrait_greenscreen_synthetic.png (+ _mask.png ground truth)")

    orchard, truth = aerial_orchard()
    utils.write_image(OUT / "orchard_aerial_synthetic.png", orchard)
    print("orchard_aerial_synthetic.png  ground truth:", truth)

    rng = np.random.default_rng(7)
    for classes, maker, folder, prefix in ((FLOWER_CLASSES, flower_image, "flowers_db", "flower"),
                                           (LEAF_CLASSES, leaf_image, "leaves_db", "leaf")):
        for cls in classes:
            for i in range(16):
                utils.write_image(OUT / folder / cls / f"{cls}_{i:02d}.jpg", maker(cls, rng))
            utils.write_image(OUT / f"{prefix}_query_{cls}.jpg", maker(cls, rng))
        print(f"{folder}/ ({len(classes)} classes x 16 images) + {prefix}_query_*.jpg")

    rng = np.random.default_rng(3)
    for finger, seed in (("A", 101), ("B", 202)):
        master = _fingerprint_master(seed)
        for impression in (1, 2):
            utils.write_image(OUT / f"fingerprint_{finger}{impression}.png", fingerprint_impression(master, rng))
    print("fingerprint_A1/A2.png (same finger), fingerprint_B1/B2.png (another finger)")

    rng = np.random.default_rng(8)
    note = demo_note()
    utils.write_image(OUT / "note_reference_genuine.png", note)
    for kind in ("genuine", "fake_print", "fake_no_thread", "fake_colour"):
        variant = note if kind == "genuine" else note_variant(note, kind, rng)
        photo = photograph_note(variant, rng)
        cv2.imencode(".jpg", photo, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tofile(str(OUT / f"note_photo_{kind}.jpg"))
    print("note_reference_genuine.png + note_photo_{genuine,fake_print,fake_no_thread,fake_colour}.jpg")

    for i, plate in enumerate(("LEB 1234", "ICT 4521", "KHI 7788")):
        utils.write_image(OUT / f"car_plate_{plate.replace(' ', '_')}.png", car_with_plate(plate, seed=i))
    print("car_plate_LEB_1234.png, car_plate_ICT_4521.png, car_plate_KHI_7788.png")

    face_samples()
    print("face_astronaut.png, face_cameraman.png (+ _mask / _helmet test edits)")

    goldfish = OUT / "fish_query_goldfish.jpg"                     # downloaded with the fish dataset
    if goldfish.exists():
        utils.write_image(OUT / "fish_underwater_simulated.jpg", underwater(utils.read_image(goldfish)))
        print("fish_underwater_simulated.jpg")


def underwater(img: np.ndarray) -> np.ndarray:
    """Simulate a photo taken under water: I = J * t + A * (1 - t), red light fades fastest."""
    f = cv2.resize(img, (512, 512), interpolation=cv2.INTER_CUBIC).astype(np.float32) / 255
    t = np.array([0.85, 0.7, 0.3])                                # transmission per channel (B, G, R)
    A = np.array([0.75, 0.6, 0.1])                                # blue-green water light
    out = f * t + A * (1 - t)
    return (np.clip(cv2.GaussianBlur(out, (0, 0), 1.2), 0, 1) * 255).astype(np.uint8)


if __name__ == "__main__":
    main()
