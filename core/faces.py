"""
Shared face detection for the face topics (4 attendance, 8 mask/helmet, 11 emotion).

YuNet (OpenCV Zoo, 2023) - a tiny CNN face detector: box + 5 landmarks (eyes, nose, mouth corners).
The model file lives in models/ (downloaded once from github.com/opencv/opencv_zoo).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
YUNET = MODELS_DIR / "face_detection_yunet_2023mar.onnx"

try:                                                      # OpenCV 5 prints a harmless DNN warning
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
except AttributeError:
    pass


@dataclass
class Face:
    box: tuple[int, int, int, int]        # x, y, w, h
    landmarks: np.ndarray                  # 5 x 2: right eye, left eye, nose, right mouth, left mouth
    score: float
    raw: np.ndarray                        # YuNet's 15-value row (needed by SFace alignment)

    def crop(self, img: np.ndarray, margin: float = 0.15, square: bool = True) -> np.ndarray:
        x, y, w, h = self.box
        if square:
            side = max(w, h)
            x, y, w, h = x + (w - side) // 2, y + (h - side) // 2, side, side
        mx, my = int(w * margin), int(h * margin)
        H, W = img.shape[:2]
        x0, y0, x1, y1 = max(0, x - mx), max(0, y - my), min(W, x + w + mx), min(H, y + h + my)
        return img[y0:y1, x0:x1]


class FaceDetector:
    """Wraps cv2.FaceDetectorYN. One instance is shared (loading the model once)."""

    _shared: "FaceDetector | None" = None

    def __init__(self, score_threshold: float = 0.8) -> None:
        if not YUNET.exists():
            raise FileNotFoundError(f"Face model missing: {YUNET.name}. Download it from github.com/opencv/opencv_zoo "
                                    "(models/face_detection_yunet) into the models/ folder.")
        self._net = cv2.FaceDetectorYN.create(str(YUNET), "", (320, 320), score_threshold, 0.3, 5000)

    @classmethod
    def shared(cls) -> "FaceDetector":
        if cls._shared is None:
            cls._shared = cls()
        return cls._shared

    def detect(self, img: np.ndarray, min_score: float = 0.8) -> list[Face]:
        self._net.setScoreThreshold(float(min_score))
        self._net.setInputSize((img.shape[1], img.shape[0]))
        _, rows = self._net.detect(img)
        faces = []
        for r in (rows if rows is not None else []):
            x, y, w, h = (int(round(v)) for v in r[:4])
            faces.append(Face((x, y, w, h), r[4:14].reshape(5, 2), float(r[14]), r))
        return sorted(faces, key=lambda f: f.box[0])         # left to right


def draw_face(img: np.ndarray, face: Face, label: str, colour=(0, 220, 0), landmarks: bool = False) -> None:
    x, y, w, h = face.box
    cv2.rectangle(img, (x, y), (x + w, y + h), colour, 2, cv2.LINE_AA)
    if landmarks:
        for px, py in face.landmarks:
            cv2.circle(img, (int(px), int(py)), 2, (0, 255, 255), -1, cv2.LINE_AA)
    if label:
        font, scale = cv2.FONT_HERSHEY_SIMPLEX, max(0.45, w / 220)
        (tw, th), _ = cv2.getTextSize(label, font, scale, 1)
        ty = y - 6 if y - th - 10 > 0 else y + h + th + 6
        cv2.rectangle(img, (x, ty - th - 4), (x + tw + 6, ty + 4), colour, -1)
        cv2.putText(img, label, (x + 3, ty), font, scale, (0, 0, 0), 1, cv2.LINE_AA)
