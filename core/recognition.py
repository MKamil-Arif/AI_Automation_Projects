"""
Shared code for the "what is this?" topics (flower species, leaf species, ...).

FeatureClassifier  : feature vectors + SVM -> train from a folder, evaluate, save, load, predict
RecognitionModule  : a DIPModule with two modes, "Predict" and "Train from folder".
                     A topic only says HOW to turn an image into features (features()) and
                     what to show while doing it (explain()).

OOP: multi-level inheritance  DIPModule -> RecognitionModule -> FlowerRecognizer
     composition              RecognitionModule has a FeatureClassifier
"""
from __future__ import annotations

from abc import abstractmethod
from pathlib import Path
from typing import Callable

import cv2
import joblib
import numpy as np

from core import utils
from core.base import DIPModule, ModuleResult, Param

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class FeatureClassifier:
    """An RBF-SVM over feature vectors produced by `extract(image)`."""

    def __init__(self, extract: Callable[[np.ndarray], np.ndarray]) -> None:
        self.extract = extract
        self._model = None
        self.labels: list[str] = []

    @staticmethod
    def _make_model(y: list[str]):
        """SVM wrapped in a calibrator so it also returns a confidence (probability).
        Features are expected to be roughly 0..1 already, so no scaler is used."""
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.svm import SVC

        folds = int(min(5, np.unique(y, return_counts=True)[1].min()))
        return CalibratedClassifierCV(SVC(kernel="rbf", C=10, gamma="scale"), cv=folds, ensemble=False)

    def train_folder(self, folder: Path, test_size: float = 0.25) -> dict:
        """folder/<class name>/*.jpg -> trains, returns accuracy + confusion matrix on a held-out part."""
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
        from sklearn.model_selection import train_test_split

        X, y = [], []
        for class_dir in sorted(p for p in folder.iterdir() if p.is_dir()):
            for file in sorted(class_dir.iterdir()):
                if file.suffix.lower() in IMAGE_EXTS:
                    X.append(self.extract(utils.read_image(file)))
                    y.append(class_dir.name)
        if len(set(y)) < 2:
            raise ValueError(f"Need at least 2 class sub-folders with images in {folder}")
        X = np.array(X)
        counts = np.unique(y, return_counts=True)[1]
        if counts.min() < 2:
            raise ValueError("Every class folder needs at least 2 images.")

        self.labels = sorted(set(y))
        report = {"images": len(y), "classes": self.labels}
        if counts.min() >= 4:
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)
            pred = self._make_model(y_tr).fit(X_tr, y_tr).predict(X_te)
            report["accuracy"] = float(accuracy_score(y_te, pred))
            report["test_images"] = len(y_te)
            report["confusion"] = confusion_matrix(y_te, pred, labels=self.labels)
            per_class = classification_report(y_te, pred, labels=self.labels, output_dict=True, zero_division=0)
            report["per_class"] = [{"Class": c, "Precision": round(per_class[c]["precision"], 3),
                                    "Recall": round(per_class[c]["recall"], 3),
                                    "F1": round(per_class[c]["f1-score"], 3)} for c in self.labels]
        self._model = self._make_model(y).fit(X, y)          # the saved model learns from everything
        return report

    def predict(self, img: np.ndarray) -> tuple[str, float, dict[str, float]]:
        vec = self.extract(img).reshape(1, -1)
        expected = getattr(self._model, "n_features_in_", vec.shape[1])
        if vec.shape[1] != expected:
            raise ValueError(f"This model was trained with a different feature extractor ({expected} features, now "
                             f"{vec.shape[1]}). Train it again with 'Train from folder'.")
        probs = self._model.predict_proba(vec)[0]
        ranking = dict(sorted(zip(map(str, self._model.classes_), map(float, probs)), key=lambda kv: -kv[1]))
        best = next(iter(ranking))
        return best, ranking[best], ranking

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self._model, "labels": self.labels}, path)

    @classmethod
    def load(cls, path: Path, extract: Callable[[np.ndarray], np.ndarray]) -> "FeatureClassifier":
        data = joblib.load(path)
        clf = cls(extract)
        clf._model, clf.labels = data["model"], data["labels"]
        return clf


def confusion_matrix_image(cm: np.ndarray, labels: list[str], cell: int = 70) -> np.ndarray:
    """Draw a confusion matrix (rows = true class, columns = predicted)."""
    n, left, top = len(labels), 110, 40
    canvas = np.full((top + n * cell + 10, left + n * cell + 10, 3), 250, np.uint8)
    peak = max(int(cm.max()), 1)
    for i in range(n):
        cv2.putText(canvas, labels[i][:12], (6, top + i * cell + cell // 2 + 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (40, 40, 40), 1, cv2.LINE_AA)
        cv2.putText(canvas, labels[i][:7], (left + i * cell + 4, top - 12), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (40, 40, 40), 1, cv2.LINE_AA)
        for j in range(n):
            v = int(cm[i, j])
            shade = int(255 - 180 * v / peak)
            colour = (shade, shade, 255) if i != j else (shade, 255, shade)  # diagonal green, errors red
            x, y = left + j * cell, top + i * cell
            cv2.rectangle(canvas, (x, y), (x + cell - 2, y + cell - 2), colour if v else (240, 240, 240), -1)
            cv2.putText(canvas, str(v), (x + cell // 2 - 8, y + cell // 2 + 6), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (20, 20, 20), 1, cv2.LINE_AA)
    return canvas


class RecognitionModule(DIPModule):
    """Base class for species-recognition topics: Predict with a saved SVM, or train one."""

    MODEL_FILE: str = ""            # file name inside models/
    DEFAULT_DATASET: str = ""       # folder (relative to the project) with one sub-folder per class
    REAL_DATASET: str = ""          # bigger real dataset (evaluation/fetch_datasets.py): the default once downloaded
    WORK_SIZE = 320
    confirm_modes = ("Train from folder",)   # training can take minutes: only on a button press
    # Topics with their own classical features can offer a choice (each choice has its own model file)
    FEATURE_SETS: tuple[str, ...] = ()       # e.g. ("Classical", "Deep (CNN features)", "Classical + Deep")
    DEFAULT_FEATURES: str = ""

    def __init__(self) -> None:
        super().__init__()
        self._feature_set = self.DEFAULT_FEATURES

    # ----- what a topic provides -------------------------------------------------
    def features(self, img: np.ndarray) -> np.ndarray:
        """Image -> feature vector. Must not call _add_step (it is also used while training).
        Default: the chosen FEATURE_SET built from classical_features() and/or the pretrained CNN."""
        kind = self.feature_kind
        if kind == "deep":
            return self.deep_features(img)
        if kind == "both":                  # both blocks unit length, so neither one dominates
            c = self.classical_features(img)
            return np.concatenate([c / (np.linalg.norm(c) + 1e-9), self.deep_features(img)])
        return self.classical_features(img)

    @property
    def feature_kind(self) -> str:
        """'classical', 'deep' or 'both'. Decided by the words Classical / Deep, NOT by '+':
        "Classical (colour + shape)" contains a '+' but is the classical set."""
        kind = self._feature_set
        if kind.startswith("Deep"):
            return "deep"
        return "both" if "Deep" in kind else "classical"

    def classical_features(self, img: np.ndarray) -> np.ndarray:
        raise NotImplementedError(f"{type(self).__name__} has no classical features")

    def deep_features(self, img: np.ndarray) -> np.ndarray:
        from core.deep import CNNEmbedder
        return CNNEmbedder.shared().embed(self.prepare(img))

    @abstractmethod
    def explain(self, img: np.ndarray) -> tuple[np.ndarray, dict]:
        """Show the pipeline for one image (_add_step) -> (annotated image, extra info)."""

    # ----- shared behaviour ----------------------------------------------------
    def parameters(self) -> list[Param]:
        params = [
            Param("mode", "Mode", "select", "Predict", options=("Predict", "Train from folder"),
                  help="Train once (on the sample dataset or your own), then Predict."),
            Param("dataset", "Dataset folder (one sub-folder per class)", "text", self.default_dataset,
                  visible_when={"mode": "Train from folder"}),
        ]
        if self.FEATURE_SETS:
            params.append(Param("features", "Features", "select", self.DEFAULT_FEATURES, options=self.FEATURE_SETS,
                                help="Classical = hand-made colour / shape / texture features; Deep = pretrained "
                                     "CNN. Each choice is trained and saved separately."))
        return params

    @property
    def default_dataset(self) -> str:
        real = self.REAL_DATASET and (PROJECT_ROOT / self.REAL_DATASET).is_dir()
        return self.REAL_DATASET if real else self.DEFAULT_DATASET

    @property
    def model_path(self) -> Path:
        if not self.FEATURE_SETS:
            return MODELS_DIR / self.MODEL_FILE
        return MODELS_DIR / f"{Path(self.MODEL_FILE).stem}_{self.feature_kind}.joblib"

    def prepare(self, img: np.ndarray) -> np.ndarray:
        return utils.resize_max(utils.ensure_bgr(img), self.WORK_SIZE)[0]

    def process(self, image, *, mode, dataset, features: str = "") -> ModuleResult:
        self._feature_set = features or self.DEFAULT_FEATURES
        if mode == "Train from folder":
            return self._train(dataset)
        if not self.model_path.exists():
            raise ValueError(f"No trained model yet. Choose mode 'Train from folder' first "
                             f"(default dataset: {self.default_dataset}).")
        img = self.prepare(image)
        annotated, extra = self.explain(img)
        label, confidence, ranking = FeatureClassifier.load(self.model_path, self.features).predict(img)
        utils.put_label(annotated, f"{label} ({confidence:.0%})", (10, 28), (255, 255, 255), 0.8)
        return ModuleResult(annotated, info={
            "Prediction": label,
            "Confidence": f"{confidence:.0%}",
            **extra,
            "All classes": [{"Class": c, "Probability": f"{p:.1%}"} for c, p in ranking.items()],
        })

    def _train(self, dataset: str) -> ModuleResult:
        folder = Path(dataset)
        folder = folder if folder.is_absolute() else PROJECT_ROOT / folder
        if not folder.is_dir():
            raise ValueError(f"Dataset folder not found: {folder}")
        clf = FeatureClassifier(self.features)
        report = clf.train_folder(folder)
        clf.save(self.model_path)
        info = {"Images": report["images"], "Classes": len(report["classes"]),
                "Model saved": f"models/{self.model_path.name}"}
        if "accuracy" in report:
            info["Test accuracy"] = f"{report['accuracy']:.1%}"
            info["Per class (test split)"] = report["per_class"]
            chart = confusion_matrix_image(report["confusion"], report["classes"])
        else:
            info["Note"] = "Too few images for a test split - trained on all, accuracy not measured."
            chart = np.full((120, 400, 3), 250, np.uint8)
        return ModuleResult(chart, info=info)
