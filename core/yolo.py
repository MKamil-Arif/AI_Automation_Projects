"""
Learned object detectors (YOLOv8n fine-tuned in this project, see evaluation/train_detectors.py).

YOLO ("You Only Look Once") looks at the whole image once and predicts boxes + confidence for
every object. We start from the official COCO-pretrained YOLOv8n and fine-tune it on a few hundred
labelled photos of potholes / number plates.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


class YoloDetector:
    _cache: dict[Path, "YoloDetector"] = {}

    def __init__(self, weights: Path) -> None:
        from ultralytics import YOLO
        self.model = YOLO(str(weights))

    @classmethod
    def get(cls, name: str) -> "YoloDetector | None":
        """The detector stored as models/<name>, or None if it has not been trained yet."""
        path = MODELS_DIR / name
        if not path.exists():
            return None
        if path not in cls._cache:
            cls._cache[path] = cls(path)
        return cls._cache[path]

    def detect(self, img_bgr: np.ndarray, conf: float = 0.25, imgsz: int = 640) -> list[tuple[int, int, int, int, float]]:
        """Boxes as (x, y, w, h, confidence), best first."""
        result = self.model.predict(img_bgr, conf=conf, imgsz=imgsz, device="cpu", verbose=False)[0]
        boxes = []
        for (x0, y0, x1, y1), c in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist()):
            boxes.append((int(x0), int(y0), int(x1 - x0), int(y1 - y0), float(c)))
        return sorted(boxes, key=lambda b: -b[4])
