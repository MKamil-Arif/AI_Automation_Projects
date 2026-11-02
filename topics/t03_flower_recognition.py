"""
Topic 3 - Flower Species Recognition (Rose, Sunflower, Tulip, Daisy, ...) from colour + petal features.

  1. Segment the flower: pixels that are NOT green (Excess Green below Otsu), the big blob nearest
     the centre, holes filled
  2. Features (all roughly 0..1):
       colour   - HSV histogram of the flower (hue 18 bins, saturation, value) + white share
       petals   - radial signature r(angle) of the outline; its FFT peak = number of petals
       shape    - circularity, solidity, aspect ratio, extent
       centre   - how much darker the middle is (sunflower) than the petals
       texture  - LBP histogram
  3. RBF-SVM trained on a folder with one sub-folder per species (core.recognition)

Default model: TF-Flowers (daisy, dandelion, roses, sunflowers, tulips), 3670 real photos; deep features
94.8 %, classical colour + petal features 60.9 % on a held-out 25 % (EVALUATION.md).
Run alone:  python -m topics.t03_flower_recognition samples/flower_query_rose.jpg "mode=Train from folder"
"""
from __future__ import annotations

import cv2
import numpy as np
from skimage.feature import local_binary_pattern

from core import utils
from core.recognition import RecognitionModule
from core.registry import register


@register
class FlowerRecognizer(RecognitionModule):
    topic_no = 3
    name = "Flower Species Recognition"
    description = ("Recognises the flower species with an SVM on pretrained CNN features (default, 94.8 % on "
                   "real photos) or on classical colour + petal-shape features (radial signature FFT = petal count).")
    samples = (tuple(f"flower_photo_{c}.jpg" for c in ("daisy", "roses", "sunflowers", "tulips"))   # real (fetch_datasets)
               + tuple(f"flower_query_{c}.jpg" for c in ("daisy", "rose", "sunflower", "tulip")))
    MODEL_FILE = "flower_svm.joblib"
    FEATURE_SETS = ("Classical (colour + shape)", "Deep (CNN features)", "Classical + Deep")
    DEFAULT_FEATURES = "Deep (CNN features)"          # best on TF-Flowers: 94.8 % (classical alone 60.9 %)
    DEFAULT_DATASET = "samples/flowers_db"
    REAL_DATASET = "datasets/real/tf_flowers"

    # ----- segmentation ---------------------------------------------------------
    @staticmethod
    def segment(img: np.ndarray) -> np.ndarray:
        f = img.astype(np.float32)
        exg = (2 * f[:, :, 1] - f[:, :, 0] - f[:, :, 2]) / (f.sum(axis=2) + 1e-6)
        exg8 = np.clip((exg + 1) / 3 * 255, 0, 255).astype(np.uint8)
        _, mask = cv2.threshold(cv2.GaussianBlur(exg8, (5, 5), 0), 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(cv2.morphologyEx(mask, cv2.MORPH_OPEN, k), cv2.MORPH_CLOSE, k)

        h, w = mask.shape
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        centre = np.array([w / 2, h / 2])

        def score(c):                                     # big and central wins
            m = cv2.moments(c)
            if m["m00"] == 0:
                return 0
            d = np.hypot(m["m10"] / m["m00"] - centre[0], m["m01"] / m["m00"] - centre[1]) / max(h, w)
            return m["m00"] * (1 - d)

        out = np.zeros_like(mask)
        if contours and 0.01 * h * w < cv2.contourArea(max(contours, key=score)) < 0.95 * h * w:
            cv2.drawContours(out, [max(contours, key=score)], -1, 255, -1)
        else:                                             # fallback: central ellipse
            cv2.ellipse(out, (w // 2, h // 2), (w // 3, h // 3), 0, 0, 360, 255, -1)
        return out

    # ----- features -------------------------------------------------------------
    @staticmethod
    def radial_signature(mask: np.ndarray, n: int = 360) -> np.ndarray:
        """Distance from the centroid to the outline, for every angle (petals = bumps)."""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        c = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(np.float64)
        m = cv2.moments(mask, binaryImage=True)
        cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
        angles = np.arctan2(c[:, 1] - cy, c[:, 0] - cx)
        radii = np.hypot(c[:, 0] - cx, c[:, 1] - cy)
        bins = ((angles + np.pi) / (2 * np.pi) * n).astype(int) % n
        sig = np.zeros(n)
        np.maximum.at(sig, bins, radii)
        for i in np.nonzero(sig == 0)[0]:                  # fill empty bins from neighbours
            sig[i] = sig[i - 1]
        return sig / (sig.max() + 1e-9)

    @staticmethod
    def petal_count(sig: np.ndarray) -> tuple[int, float]:
        """FFT of the radial signature: the strongest frequency (3..40) = number of petals."""
        spectrum = np.abs(np.fft.rfft(sig - sig.mean()))
        k = int(np.argmax(spectrum[3:41]) + 3)
        strength = float(spectrum[k] / (spectrum[1:41].mean() + 1e-9))
        return k, strength

    def classical_features(self, img: np.ndarray) -> np.ndarray:
        img = self.prepare(img)
        mask = self.segment(img)
        inside = mask > 0
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = hsv[:, :, 0][inside], hsv[:, :, 1][inside], hsv[:, :, 2][inside]
        hue_hist = np.histogram(h, bins=18, range=(0, 180), weights=s.astype(np.float64))[0]
        hue_hist /= hue_hist.sum() + 1e-9
        sat_hist = np.histogram(s, bins=4, range=(0, 256))[0] / max(len(s), 1)
        val_hist = np.histogram(v, bins=4, range=(0, 256))[0] / max(len(v), 1)
        white = np.mean((s < 45) & (v > 170))

        sig = self.radial_signature(mask)
        petals, strength = self.petal_count(sig)
        contour = max(cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0], key=cv2.contourArea)
        area, perim = cv2.contourArea(contour), cv2.arcLength(contour, True)
        (_, _), (rw, rh), _ = cv2.minAreaRect(contour)
        shape = [4 * np.pi * area / (perim ** 2 + 1e-9),
                 area / (cv2.contourArea(cv2.convexHull(contour)) + 1e-9),
                 min(rw, rh) / (max(rw, rh) + 1e-9),
                 area / (rw * rh + 1e-9),
                 petals / 40, min(strength / 10, 1), float(sig.std())]

        # centre darkness: inner 30 % of the radius vs. the whole flower
        m = cv2.moments(mask, binaryImage=True)
        cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
        yy, xx = np.indices(mask.shape)
        r = np.sqrt(area / np.pi)
        centre = inside & ((xx - cx) ** 2 + (yy - cy) ** 2 < (0.3 * r) ** 2)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        darkness = (gray[inside].mean() - gray[centre].mean()) / 255 if centre.any() else 0.0

        lbp = local_binary_pattern(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 8, 1, method="uniform")
        lbp_hist = np.histogram(lbp[inside], bins=10, range=(0, 10))[0] / max(inside.sum(), 1)
        return np.concatenate([hue_hist, sat_hist, val_hist, [white], shape, [darkness + 0.5], lbp_hist])

    # ----- explanation ----------------------------------------------------------
    def explain(self, img: np.ndarray) -> tuple[np.ndarray, dict]:
        self._add_step("Original", img)
        mask = self.segment(img)
        self._add_step("1. Flower mask (not-green + central blob)", mask)
        self._add_step("2. Segmented flower", cv2.bitwise_and(img, img, mask=mask))
        sig = self.radial_signature(mask)
        petals, strength = self.petal_count(sig)
        self._add_step(f"3. Radial signature (petals ~ {petals})", self._plot(sig))
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        self._add_step("4. Hue histogram of the flower", utils.histogram_image(hsv[:, :, 0], mask=mask))

        out = img.copy()
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, (0, 255, 255), 2, cv2.LINE_AA)
        return out, {"Petal count (FFT)": petals if strength > 3 else "unclear",
                     "Main colour": self.colour_name(hsv, mask)}

    @staticmethod
    def colour_name(hsv: np.ndarray, mask: np.ndarray) -> str:
        h, s, v = (hsv[:, :, i][mask > 0] for i in range(3))
        if np.mean((s < 45) & (v > 170)) > 0.4:                # low saturation + bright = white
            return "white"
        coloured = s >= 45
        deg = int(np.median(h[coloured])) * 2 if coloured.any() else 0
        for limit, name in ((15, "red"), (40, "orange"), (70, "yellow"), (160, "green"),
                            (260, "blue"), (300, "purple"), (340, "pink"), (360, "red")):
            if deg < limit:
                return name
        return "red"

    @staticmethod
    def _plot(sig: np.ndarray, height: int = 150) -> np.ndarray:
        canvas = np.full((height, len(sig), 3), 30, np.uint8)
        ys = (height - 10 - sig * (height - 20)).astype(np.int32)
        cv2.polylines(canvas, [np.column_stack([np.arange(len(sig)), ys]).astype(np.int32)], False,
                      (80, 220, 255), 1, cv2.LINE_AA)
        return canvas


if __name__ == "__main__":
    from core.cli import run_cli
    run_cli(FlowerRecognizer)
