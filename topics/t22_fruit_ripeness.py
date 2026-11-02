"""
Topic 22 - Fruit Ripeness Detector (Raw / Ripe / Rotten) for Mango, Banana, ...

Colour rules (works without any dataset)
  1. Segment the fruit from the background (Otsu on saturation, or GrabCut)
  2. Label every fruit pixel by HSV colour: green, yellow, orange/red, dark (rot / bruise)
  3. Decide with per-fruit thresholds: lots of dark -> Rotten, lots of green -> Raw, else Ripe

Trained model (default): HSV colour-histogram features + SVM, trained on 600 real banana / mango
  photos (evaluation/fetch_datasets.py fruit): 93.5 % on 200 unseen photos, where the colour rules get
  36 % (shadows count as "dark", olive-green raw mangoes look yellow). That dataset has no rotten fruit,
  so only the colour rules can say Rotten. Retrain on your own class folders with:
      python -m topics.t22_fruit_ripeness train <dataset_dir>

Run alone:  python -m topics.t22_fruit_ripeness samples/banana_ripe_synthetic.png fruit=Banana
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import joblib
import numpy as np

from core import utils
from core.base import DIPModule, ModuleResult, Param
from core.registry import register

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class FruitProfile:
    """Decision thresholds for one fruit, as fractions of the fruit's skin area."""
    raw_green: float        # green >= this -> Raw
    rotten_dark: float      # dark  >= this -> Rotten


class ColorHistogramClassifier:
    """HSV colour histogram + SVM. The detector *has a* classifier (composition)."""

    BINS = (18, 4, 4)       # hue, saturation, value

    def __init__(self) -> None:
        self._model = None
        self.labels: list[str] = []

    @property
    def is_trained(self) -> bool:
        return self._model is not None

    @classmethod
    def features(cls, img: np.ndarray, mask: np.ndarray, colour_fractions: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], mask, list(cls.BINS), [0, 180, 0, 256, 0, 256]).ravel()
        hist /= hist.sum() + 1e-9
        return np.concatenate([hist, colour_fractions])

    @staticmethod
    def _make_model(y: list[str]):
        """RBF SVM, wrapped so it also gives a probability (confidence).
        No scaler: histogram bins are already 0..1, and scaling near-empty bins blows up noise."""
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.svm import SVC

        folds = int(min(5, np.unique(y, return_counts=True)[1].min()))
        return CalibratedClassifierCV(SVC(kernel="rbf", C=10, gamma="scale"), cv=folds, ensemble=False)

    def fit(self, X: np.ndarray, y: list[str], test_size: float = 0.25) -> dict:
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
        from sklearn.model_selection import train_test_split

        self.labels = sorted(set(y))
        report = {"images": len(y), "classes": self.labels}
        _, counts = np.unique(y, return_counts=True)
        if counts.min() < 2:
            raise ValueError("Every class folder needs at least 2 images.")
        if counts.min() >= 4:
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)
            model = self._make_model(y_tr).fit(X_tr, y_tr)
            pred = model.predict(X_te)
            report["test_accuracy"] = accuracy_score(y_te, pred)
            report["confusion_matrix"] = confusion_matrix(y_te, pred, labels=self.labels)
            report["report"] = classification_report(y_te, pred, labels=self.labels, zero_division=0)
        else:
            report["warning"] = "Too few images for a test split - trained on everything, no accuracy."
        self._model = self._make_model(y).fit(X, y)    # final model uses all data
        return report

    def predict(self, x: np.ndarray) -> tuple[str, float]:
        if not self.is_trained:
            raise RuntimeError("Classifier is not trained.")
        probs = self._model.predict_proba(x.reshape(1, -1))[0]
        best = int(np.argmax(probs))
        return str(self._model.classes_[best]), float(probs[best])

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self._model, "labels": self.labels}, path)

    @classmethod
    def load(cls, path: Path) -> "ColorHistogramClassifier":
        data = joblib.load(path)
        clf = cls()
        clf._model, clf.labels = data["model"], data["labels"]
        return clf


@register
class FruitRipenessDetector(DIPModule):
    topic_no = 22
    name = "Fruit Ripeness Detector"
    description = ("Tells whether a mango / banana is Raw or Ripe with an SVM on HSV colour histograms "
                   "trained on real photos, or Raw / Ripe / Rotten with HSV colour rules.")
    samples = (tuple(f"fruit_photo_{state}_{fruit}.jpg" for fruit in ("banana", "mango") for state in ("raw", "ripe"))
               + tuple(f"{fruit}_{state}_synthetic.png" for fruit in ("banana", "mango") for state in ("raw", "ripe", "rotten")))

    PROFILES = {
        "Banana": FruitProfile(raw_green=0.40, rotten_dark=0.30),
        "Mango": FruitProfile(raw_green=0.55, rotten_dark=0.20),
        "Other": FruitProfile(raw_green=0.50, rotten_dark=0.25),
    }
    # colour class -> BGR colour used to paint the colour map
    COLOURS = {
        "Green": (60, 170, 60),
        "Yellow": (0, 220, 255),
        "Orange/Red": (0, 110, 255),
        "Dark (rot)": (40, 50, 70),
        "Other": (170, 170, 170),
    }
    LABEL_COLOURS = {"Raw": (60, 170, 60), "Ripe": (0, 200, 255), "Rotten": (40, 60, 140)}
    MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "fruit_ripeness_svm.joblib"
    WORK_SIZE = 600         # every image is resized to this before analysis (training too)

    def __init__(self) -> None:
        super().__init__()
        self._classifier: ColorHistogramClassifier | None = None

    def parameters(self) -> list[Param]:
        return [
            Param("fruit", "Fruit", "select", "Banana", options=tuple(self.PROFILES)),
            Param("method", "Decision method", "select", "Trained SVM model",
                  options=("Trained SVM model", "Colour rules"),
                  help="SVM trained on real banana / mango photos (93.5 % on unseen photos); the colour rules "
                       "need a plain background and no shadows (36 % on the same photos)."),
            Param("segmentation", "Fruit segmentation", "select", "Saturation (fast)",
                  options=("Saturation (fast)", "GrabCut (accurate)"),
                  help="Saturation works on plain backgrounds; GrabCut handles busier ones."),
        ]

    # ----- pipeline -----------------------------------------------------------
    def process(self, image, *, fruit, method, segmentation) -> ModuleResult:
        img, _ = utils.resize_max(utils.ensure_bgr(image), self.WORK_SIZE)
        self._add_step("Original", img)

        mask = self.segment(img, segmentation)
        self._add_step(f"1. Fruit mask ({segmentation})", mask)
        self._add_step("2. Segmented fruit", cv2.bitwise_and(img, img, mask=mask))

        fractions, colour_map = self.colour_fractions(img, mask)
        self._add_step("3. Colour classes (green / yellow / orange / dark)", colour_map)
        hue = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 0]
        self._add_step("4. Hue histogram of the fruit (0-179)", utils.histogram_image(hue, mask=mask))

        green, yellow, orange, dark = (fractions[k] for k in ("Green", "Yellow", "Orange/Red", "Dark (rot)"))
        note = None
        if method != "Colour rules" and not self.MODEL_PATH.exists():
            method, note = "Colour rules", f"models/{self.MODEL_PATH.name} not found - used the colour rules"
        if method == "Colour rules":
            label, reason = self.decide(fractions, self.PROFILES[fruit])
        else:
            features = ColorHistogramClassifier.features(img, mask, np.array(list(fractions.values())))
            label, prob = self.classifier.predict(features)
            reason = f"SVM confidence {prob:.0%} ({label})"
            label = label.split("_")[0].title()               # class folders like ripe_banana -> Ripe

        coloured = green + yellow + orange
        info = {
            "Ripeness": label,
            "Why": reason,
            "Green": f"{green:.0%}",
            "Yellow": f"{yellow:.0%}",
            "Orange/Red": f"{orange:.0%}",
            "Dark / rot": f"{dark:.0%}",
            # share of yellow/orange among the coloured (non-dark) skin; meaningless if almost all is dark
            "Ripeness index": f"{(yellow + orange) / coloured:.0%}" if coloured >= 0.1 else "-",
        }
        if note:
            info["Note"] = note
        return ModuleResult(self._annotate(img, mask, label, fruit), info=info)

    # ----- segmentation -------------------------------------------------------
    def segment(self, img: np.ndarray, method: str) -> np.ndarray:
        if method.startswith("GrabCut"):
            mask = self._grabcut(img)
        else:
            sat = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 1], (7, 7), 0)
            _, mask = cv2.threshold(sat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            disk = lambda r: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, disk(7))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, disk(5))
        return self._largest_blob(mask)

    @staticmethod
    def _grabcut(img: np.ndarray) -> np.ndarray:
        """GrabCut on a small copy (fast), mask scaled back to full size."""
        small, _ = utils.resize_max(img, 320)
        h, w = small.shape[:2]
        mask = np.zeros((h, w), np.uint8)
        rect = (int(w * 0.04), int(h * 0.04), int(w * 0.92), int(h * 0.92))
        bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
        cv2.grabCut(small, mask, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
        fg = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
        return cv2.resize(fg, img.shape[1::-1], interpolation=cv2.INTER_NEAREST)

    @staticmethod
    def _largest_blob(mask: np.ndarray) -> np.ndarray:
        """Keep the biggest object and fill its holes (dark rotten spots inside the fruit)."""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours or cv2.contourArea(max(contours, key=cv2.contourArea)) < 0.01 * mask.size:
            return np.full_like(mask, 255)                 # nothing found -> use whole image
        out = np.zeros_like(mask)
        cv2.drawContours(out, [max(contours, key=cv2.contourArea)], -1, 255, -1)
        return out

    # ----- colour analysis ----------------------------------------------------
    def colour_fractions(self, img: np.ndarray, mask: np.ndarray) -> tuple[dict[str, float], np.ndarray]:
        """Share of skin pixels in every colour class + a painted colour map.

        Gray / white pixels ("Other": highlights, or background leaking into the mask)
        are painted gray but left out of the percentages.
        """
        h, s, v = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
        h, s, v = h.astype(np.int16), s.astype(np.int16), v.astype(np.int16)
        dark = (v < 75) | ((v < 130) & (s > 60) & (h < 25))     # black rot or brown bruise
        washed = ~dark & (s < 40)                                # highlights / gray
        rest = ~dark & ~washed
        classes = {
            "Green": rest & (h >= 34) & (h <= 90),
            "Yellow": rest & (h >= 20) & (h < 34),
            "Orange/Red": rest & ((h < 20) | (h >= 160)),
            "Dark (rot)": dark,
        }
        other = ~np.logical_or.reduce(list(classes.values()))

        inside = mask > 0
        colour_map = np.zeros_like(img)
        colour_map[other & inside] = self.COLOURS["Other"]
        total = max(int((inside & ~other).sum()), 1)
        fractions = {}
        for name, pixels in classes.items():
            pixels = pixels & inside
            fractions[name] = float(pixels.sum()) / total
            colour_map[pixels] = self.COLOURS[name]
        return fractions, colour_map

    @staticmethod
    def decide(fractions: dict[str, float], profile: FruitProfile) -> tuple[str, str]:
        dark, green = fractions["Dark (rot)"], fractions["Green"]
        if dark >= profile.rotten_dark:
            return "Rotten", f"{dark:.0%} of the skin is dark / brown"
        if green >= profile.raw_green:
            return "Raw", f"{green:.0%} of the skin is still green"
        return "Ripe", "Mostly yellow / orange skin"

    # ----- trained model ------------------------------------------------------
    @property
    def classifier(self) -> ColorHistogramClassifier:
        if self._classifier is None:
            if not self.MODEL_PATH.exists():
                raise FileNotFoundError(
                    "No trained model yet. Train it first:\n"
                    "python -m topics.t22_fruit_ripeness train <dataset_dir>")
            self._classifier = ColorHistogramClassifier.load(self.MODEL_PATH)
        return self._classifier

    def train(self, dataset_dir: str | Path, model_path: Path | None = None) -> dict:
        """dataset_dir/<class_name>/*.jpg  ->  trains + saves the SVM, returns a report."""
        X, y = [], []
        for class_dir in sorted(p for p in Path(dataset_dir).iterdir() if p.is_dir()):
            for file in sorted(class_dir.iterdir()):
                if file.suffix.lower() not in IMAGE_EXTS:
                    continue
                img, _ = utils.resize_max(utils.read_image(file), self.WORK_SIZE)
                mask = self.segment(img, "Saturation (fast)")
                fractions, _ = self.colour_fractions(img, mask)
                X.append(ColorHistogramClassifier.features(img, mask, np.array(list(fractions.values()))))
                y.append(class_dir.name)
        if len(set(y)) < 2:
            raise ValueError("Need at least 2 class folders with images inside the dataset folder.")
        clf = ColorHistogramClassifier()
        report = clf.fit(np.array(X), y)
        clf.save(model_path or self.MODEL_PATH)
        self._classifier = clf
        return report

    # ----- drawing ------------------------------------------------------------
    def _annotate(self, img: np.ndarray, mask: np.ndarray, label: str, fruit: str) -> np.ndarray:
        out = img.copy()
        colour = self.LABEL_COLOURS.get(label, (255, 255, 255))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, colour, 3, cv2.LINE_AA)
        utils.put_label(out, f"{fruit}: {label.upper()}", (12, 34), colour, 0.9)
        return out


if __name__ == "__main__":
    import sys

    if len(sys.argv) >= 3 and sys.argv[1] == "train":
        result = FruitRipenessDetector().train(sys.argv[2])
        print(f"Trained on {result['images']} images, classes: {result['classes']}")
        if "test_accuracy" in result:
            print(f"Test accuracy: {result['test_accuracy']:.1%}")
            print("Confusion matrix (rows = true, cols = predicted):")
            print(result["confusion_matrix"])
            print(result["report"])
        else:
            print(result["warning"])
        print(f"Model saved to {FruitRipenessDetector.MODEL_PATH}")
    else:
        from core.cli import run_cli
        run_cli(FruitRipenessDetector)
