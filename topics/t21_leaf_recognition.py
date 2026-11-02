"""
Topic 21 - Tree Species Recognition from a Leaf (Neem, Mango, Banyan, Peepal, ...).

  1. Segment the leaf: leaf is darker / more colourful than the paper -> Otsu, biggest blob,
     holes filled
  2. Shape features (all roughly 0..1):
       aspect ratio, rectangularity, circularity, solidity, convexity, edge roughness
                       (outline length vs. a smoothed outline: saw-toothed edges like neem stand out)
       width profile - leaf turned along its main axis (PCA); width in 10 slices from the wide
                       end to the narrow end (heart-shaped peepal vs. lance-shaped mango)
       Fourier descriptors - |FFT| of the outline, normalised -> shape independent of size,
                       rotation and starting point
       Hu moments (log scale)
     Texture / colour: GLCM contrast + homogeneity inside the leaf, mean hue and saturation
  3. RBF-SVM trained on a folder with one sub-folder per species (core.recognition)

Default model: 6 medicinal species (arjun, curry, marsh pennywort, mint, neem, rubble) from real photos
(evaluation/fetch_datasets.py leaves; 100 % on a held-out 25 %). Retrain on your own folders for other species.
Run alone:  python -m topics.t21_leaf_recognition samples/leaf_query_neem.jpg "mode=Train from folder"
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from skimage.feature import graycomatrix, graycoprops

from core.recognition import RecognitionModule
from core.registry import register


@register
class LeafRecognizer(RecognitionModule):
    topic_no = 21
    name = "Tree Species from Leaf"
    description = ("Recognises the plant species from a leaf photo using shape (Fourier descriptors, width "
                   "profile, Hu moments, serration) and texture features, combined with pretrained CNN features.")
    samples = (tuple(f"leaf_photo_{c}_leaf.jpg" for c in ("neem", "curry", "mint", "arjun"))        # real (fetch_datasets)
               + tuple(f"leaf_query_{c}.jpg" for c in ("banyan", "mango", "neem", "peepal")))
    MODEL_FILE = "leaf_svm.joblib"
    FEATURE_SETS = ("Classical (colour + shape)", "Deep (CNN features)", "Classical + Deep")
    DEFAULT_FEATURES = "Classical + Deep"
    DEFAULT_DATASET = "samples/leaves_db"
    REAL_DATASET = "datasets/real/medicinal_leaves"

    N_FOURIER = 12
    N_SLICES = 10

    # ----- segmentation ---------------------------------------------------------
    @staticmethod
    def segment(img: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # leaf = saturated or dark; paper = bright and gray
        score = cv2.addWeighted(hsv[:, :, 1], 0.6, 255 - hsv[:, :, 2], 0.4, 0)
        _, mask = cv2.threshold(cv2.GaussianBlur(score, (5, 5), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        out = np.zeros_like(mask)
        if contours:
            cv2.drawContours(out, [max(contours, key=cv2.contourArea)], -1, 255, -1)
        # the thin stalk (petiole) is not part of the blade shape: a small opening removes it
        return cv2.morphologyEx(out, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))

    # ----- features -------------------------------------------------------------
    def width_profile(self, contour: np.ndarray) -> np.ndarray:
        """Width of the leaf in N slices along its main axis, wide end first, divided by length."""
        pts = contour.reshape(-1, 2).astype(np.float64)
        centred = pts - pts.mean(axis=0)
        _, _, vt = np.linalg.svd(centred, full_matrices=False)  # PCA: vt[0] = main axis
        along, across = centred @ vt[0], centred @ vt[1]
        length = along.max() - along.min() + 1e-9
        idx = np.clip(((along - along.min()) / length * self.N_SLICES).astype(int), 0, self.N_SLICES - 1)
        widths = np.array([np.ptp(across[idx == i]) if np.any(idx == i) else 0 for i in range(self.N_SLICES)])
        if widths[: self.N_SLICES // 2].sum() < widths[self.N_SLICES // 2:].sum():
            widths = widths[::-1]                             # same orientation for every leaf
        return widths / length

    def fourier_descriptors(self, contour: np.ndarray) -> np.ndarray:
        pts = contour.reshape(-1, 2).astype(np.float64)
        # resample to a fixed number of points so all leaves are comparable
        idx = np.linspace(0, len(pts) - 1, 256).astype(int)
        z = pts[idx, 0] + 1j * pts[idx, 1]
        f = np.fft.fft(z - z.mean())
        mags = np.abs(f)
        return mags[2:2 + self.N_FOURIER] / (mags[1] + 1e-9)      # / |F1| -> scale invariant

    @staticmethod
    def roughness(contour: np.ndarray) -> float:
        """How much longer the outline is than a smoothed copy of it: saw-teeth make it longer."""
        pts = contour.reshape(-1, 2).astype(np.float64)
        smooth = np.stack([gaussian_filter1d(pts[:, k], 4, mode="wrap") for k in (0, 1)], axis=1)
        smooth_len = np.sum(np.hypot(*np.diff(np.vstack([smooth, smooth[:1]]), axis=0).T))
        return float(1 - smooth_len / (cv2.arcLength(contour, True) + 1e-9))

    def classical_features(self, img: np.ndarray) -> np.ndarray:
        img = self.prepare(img)
        mask = self.segment(img)
        contour = max(cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0], key=cv2.contourArea)
        area, perim = cv2.contourArea(contour), cv2.arcLength(contour, True)
        hull = cv2.convexHull(contour)
        (_, _), (rw, rh), _ = cv2.minAreaRect(contour)
        shape = [min(rw, rh) / (max(rw, rh) + 1e-9),
                 area / (rw * rh + 1e-9),
                 4 * np.pi * area / (perim ** 2 + 1e-9),
                 area / (cv2.contourArea(hull) + 1e-9),
                 cv2.arcLength(hull, True) / (perim + 1e-9),     # convexity: < 1 for serrated edges
                 self.roughness(contour) * 8]                    # strongest cue for toothed edges
        hu = cv2.HuMoments(cv2.moments(contour)).ravel()
        hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-30) / 30

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        x, y, w, h = cv2.boundingRect(contour)
        patch = (gray[y:y + h, x:x + w] // 8).astype(np.uint8)
        glcm = graycomatrix(patch, [1, 2], [0, np.pi / 2], levels=32, symmetric=True, normed=True)
        texture = [graycoprops(glcm, "contrast").mean() / 50, graycoprops(glcm, "homogeneity").mean()]
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        colour = [hsv[:, :, 0][mask > 0].mean() / 180, hsv[:, :, 1][mask > 0].mean() / 255]
        return np.concatenate([shape, self.width_profile(contour) * 2, self.fourier_descriptors(contour) * 3,
                               hu, texture, colour])

    # ----- explanation ----------------------------------------------------------
    def explain(self, img: np.ndarray) -> tuple[np.ndarray, dict]:
        self._add_step("Original", img)
        mask = self.segment(img)
        self._add_step("1. Leaf mask (Otsu + biggest blob, stalk removed)", mask)
        contour = max(cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0], key=cv2.contourArea)
        hull = cv2.convexHull(contour)
        vis = img.copy()
        cv2.drawContours(vis, [hull], -1, (255, 160, 0), 1, cv2.LINE_AA)
        cv2.drawContours(vis, [contour], -1, (0, 0, 255), 1, cv2.LINE_AA)
        self._add_step("2. Outline (red) and convex hull (blue)", vis)
        profile = self.width_profile(contour)
        self._add_step("3. Width profile along the main axis (wide end left)", self._bars(profile))

        (_, _), (rw, rh), _ = cv2.minAreaRect(contour)
        rough = self.roughness(contour)
        out = img.copy()
        cv2.drawContours(out, [contour], -1, (0, 255, 255), 2, cv2.LINE_AA)
        return out, {"Length : width": f"{max(rw, rh) / max(min(rw, rh), 1):.1f} : 1",
                     "Edge": f"{'serrated' if rough > 0.1 else 'smooth'} (roughness {rough:.2f})",
                     "Widest at": f"{(np.argmax(profile) + 0.5) / self.N_SLICES:.0%} of length"}

    @staticmethod
    def _bars(values: np.ndarray, height: int = 140, bar: int = 30) -> np.ndarray:
        canvas = np.full((height, len(values) * bar + 10, 3), 30, np.uint8)
        top = values.max() + 1e-9
        for i, v in enumerate(values):
            hgt = int(v / top * (height - 20))
            cv2.rectangle(canvas, (5 + i * bar, height - 10 - hgt), (5 + i * bar + bar - 6, height - 10),
                          (90, 200, 90), -1)
        return canvas


if __name__ == "__main__":
    from core.cli import run_cli
    run_cli(LeafRecognizer)
