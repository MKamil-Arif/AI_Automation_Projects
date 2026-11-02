"""
Topic 12 - Image Search Engine (Content-Based Image Retrieval).

Searches a folder of images for the ones that LOOK most like the query (colour / texture),
not by file name.

Feature extractors (Strategy pattern: same interface, swappable classes)
  DeepFeatures    pretrained ConvNeXt-Tiny embedding (understands objects, not only colour)
  ColorHistogram  HSV 3-D histogram
  ColorMoments    mean, standard deviation, skewness of H, S, V
  LBPTexture      Local Binary Pattern histograms at 2 scales
  GLCMTexture     Gray-Level Co-occurrence Matrix statistics (contrast, energy, ...)
  ColorTexture    colour histogram + LBP together
Distances: Chi-square, Euclidean, Cosine, Histogram intersection

Evaluate mode uses every database image once as the query (leave-one-out) and reports
Precision@k and mean Average Precision (mAP). Sub-folder names are the class labels.
Real dataset to try: Wang / Corel-1K (10 classes x 100 images).

Run alone:  python -m topics.t12_cbir_search_engine samples/cbir_query_sea.jpg feature=Color histogram
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar

import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

from core import utils
from core.base import DIPModule, ModuleResult, Param
from core.registry import register

PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ===== feature extractors =======================================================
class FeatureExtractor(ABC):
    """Turns an image into a fixed-length vector. Every subclass is one strategy."""

    name: str = ""
    is_histogram: bool = True        # histograms are already comparable; others get min-max scaled

    @abstractmethod
    def extract(self, img: np.ndarray) -> np.ndarray:
        ...

    def visualize(self, img: np.ndarray) -> tuple[str, np.ndarray] | None:
        """Optional picture of what the feature 'sees' (for Behind the scenes)."""
        return None


class ColorHistogram(FeatureExtractor):
    name = "Color histogram (HSV)"
    BINS = (16, 4, 4)

    def extract(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, list(self.BINS), [0, 180, 0, 256, 0, 256]).ravel()
        return hist / (hist.sum() + 1e-9)

    def visualize(self, img):
        hue = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 0]
        return "Hue histogram", utils.histogram_image(hue)


class ColorMoments(FeatureExtractor):
    name = "Color moments"
    is_histogram = False

    def extract(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.float64)
        mean = hsv.mean(axis=0)
        std = hsv.std(axis=0)
        skew = np.cbrt(((hsv - mean) ** 3).mean(axis=0))
        return np.concatenate([mean, std, skew])


class LBPTexture(FeatureExtractor):
    name = "Texture (LBP)"
    SCALES = ((8, 1), (16, 2))        # (neighbours P, radius R)

    def extract(self, img):
        gray = utils.ensure_gray(img)
        hists = []
        for p, r in self.SCALES:
            lbp = local_binary_pattern(gray, p, r, method="uniform")     # values 0..p+1
            hist, _ = np.histogram(lbp, bins=p + 2, range=(0, p + 2))
            hists.append(hist / (hist.sum() + 1e-9))
        return np.concatenate(hists) / len(self.SCALES)

    def visualize(self, img):
        lbp = local_binary_pattern(utils.ensure_gray(img), 8, 1, method="uniform")
        return "LBP texture map", utils.normalize_to_uint8(lbp)


class GLCMTexture(FeatureExtractor):
    name = "Texture (GLCM)"
    is_histogram = False
    PROPS = ("contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM")

    def extract(self, img):
        gray = (utils.ensure_gray(img) // 8).astype(np.uint8)          # 32 gray levels
        glcm = graycomatrix(gray, distances=[1, 2, 4], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                            levels=32, symmetric=True, normed=True)
        # average over the 4 angles -> rotation tolerant, one value per (property, distance)
        return np.concatenate([graycoprops(glcm, p).mean(axis=1) for p in self.PROPS])


class ColorTexture(FeatureExtractor):
    """Composition: re-uses the colour and LBP extractors."""
    name = "Color + Texture"

    def __init__(self) -> None:
        self.color, self.texture = ColorHistogram(), LBPTexture()

    def extract(self, img):
        return np.concatenate([0.5 * self.color.extract(img), 0.5 * self.texture.extract(img)])

    def visualize(self, img):
        return self.texture.visualize(img)


class DeepFeatures(FeatureExtractor):
    """Pretrained CNN (ImageNet) embedding - 'what is in the picture', not just colour."""
    name = "Deep (CNN features)"

    def extract(self, img):
        from core.deep import CNNEmbedder
        vec = np.maximum(CNNEmbedder.shared().embed(img), 0)   # non-negative so every distance works
        return vec / (np.linalg.norm(vec) + 1e-9)


EXTRACTORS: dict[str, FeatureExtractor] = {e.name: e for e in
                                           (DeepFeatures(), ColorTexture(), ColorHistogram(), ColorMoments(),
                                            LBPTexture(), GLCMTexture())}

DISTANCES = {
    "Chi-square": lambda q, db: 0.5 * np.sum((db - q) ** 2 / (db + q + 1e-10), axis=1),
    "Euclidean": lambda q, db: np.linalg.norm(db - q, axis=1),
    "Cosine": lambda q, db: 1 - (db @ q) / (np.linalg.norm(db, axis=1) * np.linalg.norm(q) + 1e-10),
    "Histogram intersection": lambda q, db: 1 - np.minimum(db, q).sum(axis=1)
                                            / np.maximum(np.minimum(db.sum(axis=1), q.sum()), 1e-10),
}


# ===== the database index ==========================================================
class ImageIndex:
    """Feature vectors of every image in a folder. Built once per (folder, feature) and cached."""

    WORK_SIZE = 256
    DISK_CACHE = PROJECT_ROOT / "outputs" / "cbir_index"      # features survive an app restart
    _cache: ClassVar[dict[tuple, "ImageIndex"]] = {}

    def __init__(self, folder: Path, extractor: FeatureExtractor, cache_file: Path | None = None) -> None:
        self.folder, self.extractor = folder, extractor
        self.paths = sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
        if not self.paths:
            raise ValueError(f"No images found in {folder}")
        # class label = name of the sub-folder (images directly in the folder get "-")
        self.labels = [p.parent.name if p.parent != folder else "-" for p in self.paths]
        if cache_file is not None and cache_file.exists():
            raw = np.load(cache_file)["raw"]
        else:
            raw = np.array([extractor.extract(self.prepare(utils.read_image(p))) for p in self.paths])
            if cache_file is not None:
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(cache_file, raw=raw)
        self._lo, self._hi = raw.min(axis=0), raw.max(axis=0)
        self.features = self.normalize(raw)

    @classmethod
    def get(cls, folder: Path, extractor: FeatureExtractor) -> "ImageIndex":
        import hashlib
        files = [p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTS]
        signature = (len(files), max((p.stat().st_mtime for p in files), default=0))
        key = (str(folder), extractor.name, signature)
        if key not in cls._cache:
            name = hashlib.sha1(repr(key).encode()).hexdigest()[:16]
            cls._cache[key] = cls(folder, extractor, cls.DISK_CACHE / f"{name}.npz")
        return cls._cache[key]

    @classmethod
    def prepare(cls, img: np.ndarray) -> np.ndarray:
        return utils.resize_max(utils.ensure_bgr(img), cls.WORK_SIZE)[0]

    def normalize(self, features: np.ndarray) -> np.ndarray:
        """Min-max scale non-histogram features to 0..1 so no single value dominates."""
        if self.extractor.is_histogram:
            return features
        return np.clip((features - self._lo) / (self._hi - self._lo + 1e-10), 0, 1)

    @property
    def has_classes(self) -> bool:
        return len(set(self.labels) - {"-"}) >= 2

    def search(self, query_features: np.ndarray, metric: str, k: int) -> list[tuple[Path, str, float]]:
        dist = DISTANCES[metric](query_features, self.features)
        order = np.argsort(dist)[:k]
        return [(self.paths[i], self.labels[i], float(dist[i])) for i in order]

    def evaluate(self, metric: str, k: int) -> dict:
        """Leave-one-out: each image queries all the others."""
        if not self.has_classes:
            raise ValueError("Evaluation needs the database split into class sub-folders.")
        labels = np.array(self.labels)
        precisions, aps = [], []
        for i in range(len(self.paths)):
            dist = DISTANCES[metric](self.features[i], self.features)
            dist[i] = np.inf                                   # do not find yourself
            order = np.argsort(dist)[:-1]
            relevant = labels[order] == labels[i]
            precisions.append(relevant[:k].mean())
            hits = np.cumsum(relevant)
            aps.append((hits[relevant] / (np.nonzero(relevant)[0] + 1)).mean() if relevant.any() else 0.0)
        precisions, aps = np.array(precisions), np.array(aps)
        per_class = [{"Class": c, "Images": int((labels == c).sum()),
                      f"P@{k}": round(float(precisions[labels == c].mean()), 3),
                      "AP": round(float(aps[labels == c].mean()), 3)} for c in sorted(set(labels))]
        return {"precision": float(precisions.mean()), "map": float(aps.mean()), "per_class": per_class}


# ===== the module ===================================================================
@register
class ImageSearchEngine(DIPModule):
    topic_no = 12
    name = "Image Search Engine (CBIR)"
    description = ("Finds the most similar images in a folder by colour and texture features "
                   "(HSV histogram, colour moments, LBP, GLCM) - not by file name.")
    samples = tuple(f"cbir_query_{c}.jpg" for c in
                    ("sunset", "forest", "sea", "bricks", "grass", "night", "desert", "tiles"))
    wide_output = True

    THUMB = 140

    def parameters(self) -> list[Param]:
        return [
            Param("mode", "Mode", "select", "Search", options=("Search", "Evaluate database"),
                  help="Evaluate: every database image is used once as the query -> Precision@k, mAP."),
            Param("database", "Image database folder", "text", "samples/cbir_db",
                  help="Relative to the project folder, or a full path. Sub-folders = classes."),
            Param("feature", "Feature", "select", next(iter(EXTRACTORS)), options=tuple(EXTRACTORS)),
            Param("metric", "Distance", "select", "Chi-square", options=tuple(DISTANCES)),
            Param("top_k", "Results to show (k)", "slider", 8, 1, 20, 1),
        ]

    def process(self, image, *, mode, database, feature, metric, top_k) -> ModuleResult:
        folder = Path(database)
        folder = folder if folder.is_absolute() else PROJECT_ROOT / folder
        if not folder.is_dir():
            raise ValueError(f"Database folder not found: {folder}")
        extractor = EXTRACTORS[feature]
        index = ImageIndex.get(folder, extractor)

        if mode == "Evaluate database":
            result = index.evaluate(metric, top_k)
            self._add_step(f"Precision@{top_k} per class", self._bar_chart(result["per_class"], f"P@{top_k}"))
            chart = self._bar_chart(result["per_class"], "AP")
            return ModuleResult(chart, info={
                "Database images": len(index.paths),
                f"Mean P@{top_k}": f"{result['precision']:.1%}",
                "mAP": f"{result['map']:.1%}",
                "Per class": result["per_class"],
            })

        query = ImageIndex.prepare(image)
        self._add_step("Query", query)
        vis = extractor.visualize(query)
        if vis:
            self._add_step(f"Query feature: {vis[0]}", vis[1])
        results = index.search(index.normalize(extractor.extract(query)), metric, top_k)
        for rank, (path, _label, dist) in enumerate(results[:3], 1):
            self._add_step(f"Match #{rank}: {path.name} (d={dist:.3f})", ImageIndex.prepare(utils.read_image(path)))

        rows = [{"Rank": r, "Image": p.relative_to(folder).as_posix(), "Class": lab, "Distance": round(d, 4)}
                for r, (p, lab, d) in enumerate(results, 1)]
        info = {"Database images": len(index.paths), "Best match": results[0][0].name, "Results": rows}
        if index.has_classes:
            top_class = max(set(lab for _, lab, _ in results), key=[lab for _, lab, _ in results].count)
            info["Most common class"] = top_class
        return ModuleResult(self._result_grid(query, results), info=info)

    # ----- drawing ------------------------------------------------------------
    def _thumb(self, img: np.ndarray) -> np.ndarray:
        """Letter-box an image into a THUMB x THUMB square."""
        t = self.THUMB
        h, w = img.shape[:2]
        s = t / max(h, w)
        small = cv2.resize(img, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)
        out = np.full((t, t, 3), 25, np.uint8)
        y, x = (t - small.shape[0]) // 2, (t - small.shape[1]) // 2
        out[y:y + small.shape[0], x:x + small.shape[1]] = utils.ensure_bgr(small)
        return out

    def _result_grid(self, query: np.ndarray, results, cols: int = 4) -> np.ndarray:
        t, gap, label_h = self.THUMB, 10, 34
        rows = int(np.ceil(len(results) / cols))
        cell_h = t + label_h
        width = t + 3 * gap + cols * (t + gap)
        canvas = np.full((max(rows, 1) * (cell_h + gap) + gap, width, 3), 245, np.uint8)

        canvas[gap:gap + t, gap:gap + t] = self._thumb(query)
        cv2.rectangle(canvas, (gap - 3, gap - 3), (gap + t + 2, gap + t + 2), (0, 140, 255), 3)
        cv2.putText(canvas, "QUERY", (gap, gap + t + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 110, 230), 2, cv2.LINE_AA)

        x0 = t + 3 * gap
        for i, (path, label, dist) in enumerate(results):
            r, c = divmod(i, cols)
            x, y = x0 + c * (t + gap), gap + r * (cell_h + gap)
            canvas[y:y + t, x:x + t] = self._thumb(utils.read_image(path))
            cv2.putText(canvas, f"#{i + 1}  d={dist:.3f}", (x, y + t + 14), cv2.FONT_HERSHEY_SIMPLEX,
                        0.42, (30, 30, 30), 1, cv2.LINE_AA)
            cv2.putText(canvas, label[:18], (x, y + t + 29), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, (110, 110, 110), 1, cv2.LINE_AA)
        return canvas

    @staticmethod
    def _bar_chart(rows: list[dict], key: str, width: int = 640) -> np.ndarray:
        bar_h, gap, left = 26, 8, 110
        height = len(rows) * (bar_h + gap) + gap + 24
        canvas = np.full((height, width, 3), 245, np.uint8)
        cv2.putText(canvas, key, (left, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 60, 60), 1, cv2.LINE_AA)
        for i, row in enumerate(rows):
            y = 24 + gap + i * (bar_h + gap)
            value = float(row[key])
            cv2.putText(canvas, str(row["Class"])[:14], (8, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (40, 40, 40), 1, cv2.LINE_AA)
            bar_w = int(value * (width - left - 70))
            colour = (80, 170, 60) if value >= 0.8 else (40, 170, 230) if value >= 0.5 else (60, 60, 220)
            cv2.rectangle(canvas, (left, y), (left + bar_w, y + bar_h), colour, -1)
            cv2.putText(canvas, f"{value:.0%}", (left + bar_w + 6, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (40, 40, 40), 1, cv2.LINE_AA)
        return canvas


if __name__ == "__main__":
    from core.cli import run_cli
    run_cli(ImageSearchEngine)
