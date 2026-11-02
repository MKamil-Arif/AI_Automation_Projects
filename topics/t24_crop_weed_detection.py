"""
Topic 24 - Crop / Weed Detection from Drone Images (green pixel segmentation).

  1. Vegetation mask   : Excess Green index ExG = 2g - r - b (chromatic coordinates) + Otsu
                         (alternatives: ExG - ExR, or an HSV green range; all thresholds adapt per photo)
  2. Row direction     : project all vegetation pixels onto a line at every angle 0..179 deg.
                         At the row angle the projection profile has the sharpest peaks.
  3. Row positions     : peaks of that profile; row spacing from its autocorrelation
  4. Crop or weed      : a plant lying mostly inside a row band = crop, between rows = weed
                         (rule without rows: big plants = crop, small plants = weed)
                         Trained rule: 11 shape + colour features per plant (size, solidity, extent,
                         circularity, elongation, leaf thickness, ExG, HSV) -> SVM learned from the
                         CWFID field images (Haug & Ostermann 2014), for close-up photos without rows

Run alone:  python -m topics.t24_crop_weed_detection samples/drone_field_synthetic.png
Train the plant SVM:  python -m topics.t24_crop_weed_detection train datasets/real/cwfid
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy.signal import find_peaks

from core import utils
from core.base import DIPModule, ModuleResult, Param
from core.recognition import MODELS_DIR
from core.registry import register

PLANT_MODEL = MODELS_DIR / "cropweed_svm.joblib"


@dataclass
class CropRows:
    """Detected rows: all positions are measured along the row normal (d = x cos t + y sin t)."""
    normal_deg: float
    spacing: float
    centres: np.ndarray
    profile: np.ndarray
    origin: float              # d value of profile[0]

    @property
    def row_angle(self) -> float:
        """Direction the rows run in, degrees from the horizontal axis."""
        return (self.normal_deg + 90) % 180

    def distance_map(self, shape: tuple[int, int]) -> np.ndarray:
        """Distance of every pixel to the nearest row centre line."""
        t = np.radians(self.normal_deg)
        ys, xs = np.indices(shape, dtype=np.float32)
        d = xs * np.cos(t) + ys * np.sin(t) - self.origin
        return np.min(np.abs(d[:, :, None] - self.centres[None, None, :]), axis=2)


@register
class CropWeedDetector(DIPModule):
    topic_no = 24
    name = "Crop / Weed Detection (Drone)"
    description = ("Finds vegetation with the Excess Green index, then marks weeds: plants between the "
                   "detected crop rows, small plants, or an SVM on each plant's shape and colour (close-ups).")
    samples = ("drone_field_synthetic.png",)

    INDICES = ("ExG (Excess Green)", "ExG - ExR", "HSV green range")
    STRATEGIES = ("Crop rows", "Plant size", "Trained (plant shape + colour)")
    WORK_SIZE = 1200
    _svm = None

    def parameters(self) -> list[Param]:
        return [
            Param("index", "Vegetation index", "select", self.INDICES[0], options=self.INDICES),
            Param("strategy", "Crop / weed rule", "select", "Crop rows", options=self.STRATEGIES,
                  help="Crop rows: plants between rows are weeds. Plant size: small plants are weeds. "
                       "Trained: an SVM on each plant's shape and colour (close-up photos, no rows needed)."),
            Param("row_band", "Row band width (% of row spacing)", "slider", 40, 10, 90, 5,
                  visible_when={"strategy": "Crop rows"}),
            Param("crop_size", "Crop if area >= (% of largest plant)", "slider", 25, 5, 90, 5,
                  visible_when={"strategy": "Plant size"}),
            Param("min_area", "Ignore blobs smaller than (pixels)", "slider", 20, 0, 500, 10),
        ]

    # ----- pipeline -----------------------------------------------------------
    def process(self, image, *, index, strategy, row_band, crop_size, min_area) -> ModuleResult:
        img, _ = utils.resize_max(utils.ensure_bgr(image), self.WORK_SIZE)
        self._add_step("Original", img)
        labels, stats, plants, is_crop, rows = self.classify_plants(img, index, strategy, row_band, crop_size, min_area)

        crop_ids, weed_ids = plants[is_crop[plants]], plants[~is_crop[plants]]
        crop_mask = np.isin(labels, crop_ids)
        weed_mask = np.isin(labels, weed_ids)
        output = self._annotate(img, crop_mask, weed_mask, stats[weed_ids], rows)

        veg_px = crop_mask.sum() + weed_mask.sum()
        info = {
            "Crop plants": len(crop_ids),
            "Weed patches": len(weed_ids),
            "Weed share": f"{weed_mask.sum() / max(veg_px, 1):.1%}",
            "Vegetation cover": f"{veg_px / labels.size:.1%}",
        }
        if rows is not None:
            info.update({"Crop rows": len(rows.centres),
                         "Row angle": f"{rows.row_angle:.1f} deg",
                         "Row spacing": f"{rows.spacing:.0f} px"})
        return ModuleResult(output, info=info)

    def classify_plants(self, img, index, strategy, row_band=40, crop_size=25, min_area=20):
        """-> (component labels, component stats, plant ids, is_crop per label, rows or None).
        The evaluation calls this directly to compare the crop / weed masks with the ground truth."""
        veg_index, mask = self.vegetation_mask(img, index)
        self._add_step(f"1. {index}", veg_index)
        self._add_step("2. Vegetation mask", mask)

        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        areas = stats[:, cv2.CC_STAT_AREA]
        plants = np.array([i for i in range(1, n) if areas[i] >= max(min_area, 1)])
        if plants.size == 0:
            raise ValueError("No vegetation found - try another vegetation index.")

        rows = None
        if strategy == "Crop rows":
            rows = self.detect_rows(mask)
            if rows is None:
                raise ValueError("Could not find crop rows - try the 'Plant size' or 'Trained' rule.")
            self._add_step(f"3. Projection profile at {rows.row_angle:.1f} deg (peaks = rows)",
                           self._plot_profile(rows))
            in_band = rows.distance_map(mask.shape) <= row_band / 100 * rows.spacing / 2
            band_vis = img.copy()
            band_vis[in_band] = (0.5 * band_vis[in_band] + (0, 90, 0)).astype(np.uint8)
            self._add_step("4. Crop-row bands", band_vis)
            share_in_band = np.bincount(labels[in_band], minlength=n) / np.maximum(areas, 1)
            is_crop = share_in_band > 0.5
        elif strategy.startswith("Trained"):
            model = self._plant_model()
            is_crop = np.zeros(n, bool)
            is_crop[plants] = model.predict(self.plant_features(img, labels, stats, plants)) == 1
        else:
            is_crop = areas >= crop_size / 100 * areas[plants].max()
        return labels, stats, plants, is_crop, rows

    # ----- trained rule: plant shape + colour -> SVM ----------------------------
    @staticmethod
    def plant_features(img: np.ndarray, labels: np.ndarray, stats: np.ndarray, plants: np.ndarray) -> np.ndarray:
        """11 numbers per plant: size, solidity, extent, circularity, elongation, leaf thickness, ExG,
        mean H, S, V and V spread. Carrot seedlings have thin feathery leaves (low solidity and
        thickness); weeds are broader and bigger."""
        f = img.astype(np.float32)
        exg = (2 * f[:, :, 1] - f[:, :, 0] - f[:, :, 2]) / (f.sum(axis=2) + 1e-6)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        out = []
        for i in plants:
            x, y, w, h, area = stats[i]
            region = (labels[y:y + h, x:x + w] == i).astype(np.uint8)
            contour = max(cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], key=cv2.contourArea)
            hull = cv2.contourArea(cv2.convexHull(contour)) + 1e-6
            perimeter = cv2.arcLength(contour, True) + 1e-6
            m = cv2.moments(region, binaryImage=True)
            cov = np.array([[m["mu20"], m["mu11"]], [m["mu11"], m["mu02"]]]) / max(m["m00"], 1e-6)
            small, big = np.sort(np.linalg.eigvalsh(cov))
            thickness = cv2.distanceTransform(region, cv2.DIST_L2, 3).max() / np.sqrt(area)
            inside = region > 0
            px = hsv[y:y + h, x:x + w][inside].astype(np.float32)
            out.append([np.log(area / labels.size), area / hull, area / (w * h), 4 * np.pi * area / perimeter ** 2,
                        np.sqrt(max(small, 0) / max(big, 1e-9)), thickness, float(exg[y:y + h, x:x + w][inside].mean()),
                        px[:, 0].mean() / 180, px[:, 1].mean() / 255, px[:, 2].mean() / 255, px[:, 2].std() / 255])
        return np.array(out, np.float32)

    @classmethod
    def _plant_model(cls):
        if cls._svm is None:
            if not PLANT_MODEL.exists():
                raise ValueError(f"models/{PLANT_MODEL.name} is not trained yet: python -m topics.t24_crop_weed_detection "
                                 "train datasets/real/cwfid   (data: python evaluation/fetch_datasets.py cwfid)")
            import joblib
            cls._svm = joblib.load(PLANT_MODEL)
        return cls._svm

    def train(self, folder: Path, index: str = INDICES[0], min_area: int = 20) -> dict:
        """CWFID layout: images/NNN_image.png + annotations/NNN_annotation.png (red = weed, green = crop).
        Uses the 'train' ids of train_test_split.yaml when present. Each plant blob gets the label
        that most of its annotated pixels have."""
        import joblib
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC
        folder = Path(folder)
        ids = sorted(int(p.name[:3]) for p in (folder / "images").glob("*_image.png"))
        split = folder / "train_test_split.yaml"
        if split.exists():
            ids = [int(v) for v in re.findall(r"\d+", re.search(r"train:\s*\[([^\]]*)\]", split.read_text()).group(1))]
        X, y = [], []
        for i in ids:
            img, _ = utils.resize_max(utils.read_image(folder / "images" / f"{i:03d}_image.png"), self.WORK_SIZE)
            ann = cv2.resize(utils.read_image(folder / "annotations" / f"{i:03d}_annotation.png"),
                             (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
            _, mask = self.vegetation_mask(img, index)
            n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
            crop_px = np.bincount(labels[(ann[:, :, 1] > 0) & (mask > 0)], minlength=n)
            weed_px = np.bincount(labels[(ann[:, :, 2] > 0) & (mask > 0)], minlength=n)
            plants = np.array([k for k in range(1, n) if stats[k, 4] >= min_area and crop_px[k] + weed_px[k] > 0])
            if plants.size:
                X.append(self.plant_features(img, labels, stats, plants))
                y.append((crop_px[plants] > weed_px[plants]).astype(int))        # 1 = crop
        X, y = np.vstack(X), np.concatenate(y)
        model = make_pipeline(StandardScaler(), SVC(C=10, class_weight="balanced")).fit(X, y)
        joblib.dump(model, PLANT_MODEL)
        type(self)._svm = None
        return {"images": len(ids), "plants": len(y), "crop plants": int(y.sum()), "saved": f"models/{PLANT_MODEL.name}"}

    # ----- step 1: vegetation -------------------------------------------------
    def vegetation_mask(self, img: np.ndarray, index: str) -> tuple[np.ndarray, np.ndarray]:
        f = img.astype(np.float32)
        total = f.sum(axis=2) + 1e-6
        b, g, r = (f[:, :, i] / total for i in range(3))     # chromatic coordinates
        exg = 2 * g - r - b
        if index == self.INDICES[0]:
            vis = np.clip((exg + 1) / 3 * 255, 0, 255).astype(np.uint8)
            _, mask = cv2.threshold(vis, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif index == self.INDICES[1]:
            value = exg - (1.4 * r - g)                          # ExG - ExR (Meyer & Neto 2008)
            vis = utils.normalize_to_uint8(value)
            # the textbook threshold is 0, but a camera colour cast (greenish soil in CWFID) breaks it:
            # Otsu adapts to every photo (vegetation IoU on CWFID 0.24 -> 0.82)
            _, mask = cv2.threshold(vis, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            s_min, _ = cv2.threshold(hsv[:, :, 1], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            mask = cv2.inRange(hsv, (30, max(40, int(s_min)), 30), (90, 255, 255))   # green AND saturated
            vis = hsv[:, :, 0]
        disk = lambda k: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, disk(3))      # remove soil speckles
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, disk(3))      # fill small gaps in leaves
        return vis, mask

    # ----- steps 2-3: crop rows -----------------------------------------------
    @staticmethod
    def _profile(xs, ys, normal_deg: float) -> tuple[np.ndarray, float]:
        t = np.radians(normal_deg)
        d = xs * np.cos(t) + ys * np.sin(t)
        origin = float(d.min())
        hist = np.bincount((d - origin).astype(np.int64)).astype(np.float64)
        kernel = cv2.getGaussianKernel(13, 3).ravel()
        return np.convolve(hist, kernel, mode="same"), origin

    def detect_rows(self, mask: np.ndarray) -> CropRows | None:
        ys, xs = np.nonzero(mask)
        if xs.size < 200:
            return None
        rng = np.random.default_rng(0)
        if xs.size > 40000:                                      # a sample is enough to find the angle
            pick = rng.choice(xs.size, 40000, replace=False)
            sx, sy = xs[pick].astype(np.float64), ys[pick].astype(np.float64)
        else:
            sx, sy = xs.astype(np.float64), ys.astype(np.float64)

        # peaky profile <=> large sum of squares (the total pixel count is the same at every angle)
        sharpness = lambda a: float(np.sum(self._profile(sx, sy, a)[0] ** 2))
        coarse = max(np.arange(0, 180, 2.0), key=sharpness)
        best = max(np.arange(coarse - 2, coarse + 2.01, 0.25), key=sharpness) % 180

        profile, origin = self._profile(xs.astype(np.float64), ys.astype(np.float64), best)
        centred = profile - profile.mean()
        acf = np.correlate(centred, centred, mode="full")[len(centred) - 1:]
        lags, _ = find_peaks(acf, height=0.15 * acf[0])
        lags = lags[lags > 8]
        if lags.size == 0:
            return None
        spacing = float(lags[0])
        centres, _ = find_peaks(profile, distance=max(1, int(0.6 * spacing)), height=0.25 * profile.max())
        if centres.size < 2:
            return None
        return CropRows(best, spacing, centres.astype(np.float32), profile, origin)

    # ----- drawing ------------------------------------------------------------
    @staticmethod
    def _plot_profile(rows: CropRows, height: int = 160) -> np.ndarray:
        prof = rows.profile
        width = len(prof)
        canvas = np.full((height, width, 3), 30, np.uint8)
        ys = (height - 8 - prof / (prof.max() + 1e-9) * (height - 16)).astype(np.int32)
        cv2.polylines(canvas, [np.column_stack([np.arange(width), ys]).astype(np.int32)], False,
                      (120, 220, 120), 1, cv2.LINE_AA)
        for c in rows.centres.astype(int):
            cv2.line(canvas, (c, 0), (c, height), (255, 160, 60), 1)
        return canvas

    @staticmethod
    def _annotate(img, crop_mask, weed_mask, weed_stats, rows: CropRows | None) -> np.ndarray:
        out = img.copy()
        out[crop_mask] = (0.45 * out[crop_mask] + (0, 140, 0)).astype(np.uint8)
        out[weed_mask] = (0.35 * out[weed_mask] + (0, 0, 165)).astype(np.uint8)
        if rows is not None:
            t = np.radians(rows.normal_deg)
            n, along = np.array([np.cos(t), np.sin(t)]), np.array([-np.sin(t), np.cos(t)])
            for c in rows.centres:
                p = (c + rows.origin) * n
                a, b = p - 3000 * along, p + 3000 * along
                cv2.line(out, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), (255, 160, 60), 1, cv2.LINE_AA)
        for x, y, w, h, _ in weed_stats:
            cv2.rectangle(out, (x - 3, y - 3), (x + w + 3, y + h + 3), (0, 0, 255), 1)
        utils.put_label(out, f"Crop: green   Weed: red ({len(weed_stats)})", (10, 25), (255, 255, 255), 0.6)
        return out


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "train":
        print(CropWeedDetector().train(Path(sys.argv[2])))
    else:
        from core.cli import run_cli
        run_cli(CropWeedDetector)
