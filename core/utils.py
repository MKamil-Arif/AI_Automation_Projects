"""Small image helpers reused by many topics."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


# ----- reading / writing ------------------------------------------------------
def read_image(path: str | Path) -> np.ndarray:
    """cv2.imread fails on some Windows paths, so decode from raw bytes instead."""
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


def write_image(path: str | Path, image: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix or ".png", image)
    if not ok:
        raise IOError(f"Could not encode image for {path}")
    buf.tofile(str(path))


def decode_image(data: bytes) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Uploaded file is not a valid image.")
    return image


def encode_png(image: np.ndarray) -> bytes:
    return cv2.imencode(".png", image)[1].tobytes()


# ----- conversions --------------------------------------------------------------
def ensure_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image


def ensure_gray(image: np.ndarray) -> np.ndarray:
    return image if image.ndim == 2 else cv2.cvtColor(ensure_bgr(image), cv2.COLOR_BGR2GRAY)


def to_display(image: np.ndarray) -> np.ndarray:
    """BGR (OpenCV) -> RGB (Streamlit / matplotlib). Grayscale stays as is, BGRA keeps its alpha."""
    if image.dtype != np.uint8:
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if image.ndim == 2:
        return image
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def resize_max(image: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    """Shrink so the longest side is <= max_side. Returns (image, scale)."""
    h, w = image.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:
        image = cv2.resize(image, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    return image, scale


def normalize_to_uint8(image: np.ndarray) -> np.ndarray:
    return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def enlarge(image: np.ndarray, factor: int) -> np.ndarray:
    """Blow up a tiny image (e.g. a 32x128 watermark) without blurring it."""
    return cv2.resize(image, None, fx=factor, fy=factor, interpolation=cv2.INTER_NEAREST)


# ----- drawing --------------------------------------------------------------------
def put_label(image: np.ndarray, text: str, org: tuple[int, int],
              color: tuple[int, int, int] = (255, 255, 255), scale: float = 0.6) -> None:
    """Text with a dark background box so it is readable on any image."""
    font, thick = cv2.FONT_HERSHEY_SIMPLEX, max(1, round(scale * 2))
    (tw, th), base = cv2.getTextSize(text, font, scale, thick)
    x, y = org
    cv2.rectangle(image, (x - 2, y - th - 4), (x + tw + 2, y + base), (0, 0, 0), -1)
    cv2.putText(image, text, (x, y), font, scale, color, thick, cv2.LINE_AA)


def histogram_image(image: np.ndarray, width: int = 256, height: int = 160,
                    mask: np.ndarray | None = None) -> np.ndarray:
    """Draws the B, G, R (or gray) histogram as a small picture."""
    canvas = np.full((height, width, 3), 30, np.uint8)
    if image.ndim == 2:
        channels = [(image, (220, 220, 220))]
    else:
        channels = [(image[:, :, i], c) for i, c in enumerate([(255, 80, 80), (80, 255, 80), (80, 80, 255)])]
    for chan, color in channels:
        hist = cv2.calcHist([chan], [0], mask, [256], [0, 256]).ravel()
        hist = hist / (hist.max() + 1e-9) * (height - 10)
        pts = np.column_stack([np.linspace(0, width - 1, 256), height - 1 - hist]).astype(np.int32)
        cv2.polylines(canvas, [pts], False, color, 1, cv2.LINE_AA)
    return canvas


# ----- morphology helpers ---------------------------------------------------------
def disk(radius: float) -> np.ndarray:
    r = max(1, int(radius))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def fill_small_holes(mask: np.ndarray, max_area: float) -> np.ndarray:
    """Fill enclosed holes up to max_area pixels (e.g. pale cell centres); big gaps stay open."""
    holes = cv2.bitwise_not(mask)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(holes, connectivity=4)
    h, w = mask.shape
    small = np.zeros(n, bool)
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        small[i] = area <= max_area and x > 0 and y > 0 and x + bw < w and y + bh < h
    out = mask.copy()
    out[small[labels]] = 255
    return out


def granulometry(mask: np.ndarray, max_side: int = 320) -> tuple[float, np.ndarray, np.ndarray]:
    """Typical object radius of a binary mask by a pattern spectrum.

    The mask is opened with ever larger disks; an object of radius R survives openings up to R
    and vanishes after, so the radius where the MOST area disappears is the typical object size.
    Touching objects and small specks do not fool it. Returns (radius in mask pixels, radii, spectrum).
    """
    h, w = mask.shape
    s = min(1.0, max_side / max(h, w))                     # small copy: big openings are costly
    small = cv2.resize(mask, None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST) if s < 1 else mask
    small = fill_small_holes(small, 0.03 * small.size)     # holes would turn objects into thin rings
    radii = np.arange(2, max(6, min(small.shape) // 5), 2)
    areas = np.array([cv2.countNonZero(cv2.morphologyEx(small, cv2.MORPH_OPEN, disk(r))) for r in radii])
    if areas[0] == 0:
        return 10.0, radii, np.zeros(max(len(radii) - 1, 1))
    spectrum = -np.diff(areas)                             # area lost between radius r and r + 2
    return float((radii[int(np.argmax(spectrum))] + 1) / s), radii, spectrum


def plot_spectrum(spectrum: np.ndarray, height: int = 140, bar: int = 10) -> np.ndarray:
    """Bar chart of a pattern spectrum, the peak highlighted."""
    canvas = np.full((height, len(spectrum) * bar + 10, 3), 30, np.uint8)
    top, best = max(spectrum.max(), 1), int(np.argmax(spectrum))
    for i, v in enumerate(spectrum):
        hgt = int(max(v, 0) / top * (height - 15))
        colour = (0, 200, 255) if i == best else (180, 180, 180)
        cv2.rectangle(canvas, (5 + i * bar, height - 5 - hgt), (5 + i * bar + bar - 2, height - 5), colour, -1)
    return canvas


# ----- quality metrics ------------------------------------------------------------
def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else float(10 * np.log10(255.0 ** 2 / mse))


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Structural Similarity (Wang et al. 2004) on the grayscale images."""
    x = ensure_gray(a).astype(np.float64)
    y = ensure_gray(b).astype(np.float64)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    blur = lambda img: cv2.GaussianBlur(img, (11, 11), 1.5)
    mx, my = blur(x), blur(y)
    sxx = blur(x * x) - mx ** 2
    syy = blur(y * y) - my ** 2
    sxy = blur(x * y) - mx * my
    ssim_map = ((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx ** 2 + my ** 2 + c1) * (sxx + syy + c2))
    return float(ssim_map.mean())
