"""
Topic 11 - Real-time Emotion Recognition (Happy, Sad, Angry, Surprised, ...).

  1. Faces found with the YuNet CNN detector (core.faces)
  2. Each face cropped to a square, made gray and 64 x 64
  3. FER+ CNN (Microsoft, trained on the FER+ re-labelled FER-2013 faces) gives scores for
     8 emotions; the face and its mirror image are both scored and averaged (test-time augmentation)
  4. Calibration (optional, on by default): FER+ answers "neutral" far too often for real angry / sad /
     fearful faces. A logistic regression on the 8 scores, fitted on 1000 FER-2013 TRAINING faces,
     turns them into 7 calibrated emotions (FER-2013 test: 50 % -> 62 %, see EVALUATION.md)
  Optional: Gemini (cloud) as a second opinion on each face.

Fit the calibration:  python -m topics.t11_emotion_recognition calibrate <folder with one sub-folder per emotion>

Real-time with a webcam window (press ESC to stop):
    python -m topics.t11_emotion_recognition --webcam
Run on an image:  python -m topics.t11_emotion_recognition samples/face_astronaut.png
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np

from core import cloud, utils
from core.base import DIPModule, ModuleResult, Param
from core.faces import MODELS_DIR, FaceDetector, draw_face
from core.registry import register

FERPLUS = MODELS_DIR / "emotion-ferplus-8.onnx"
CALIBRATION = MODELS_DIR / "fer_calibration.json"
EMOTIONS = ("neutral", "happy", "surprised", "sad", "angry", "disgust", "fear", "contempt")
CALIBRATED = ("angry", "disgust", "fear", "happy", "neutral", "sad", "surprised")      # FER-2013 classes
COLOURS = {"neutral": (200, 200, 200), "happy": (0, 220, 0), "surprised": (0, 220, 255), "sad": (255, 140, 0),
           "angry": (0, 0, 255), "disgust": (0, 140, 70), "fear": (200, 0, 200), "contempt": (120, 120, 255)}


@register
class EmotionRecognizer(DIPModule):
    topic_no = 11
    name = "Real-time Emotion Recognition"
    description = ("Finds faces (YuNet) and tells the emotion of each one with the FER+ CNN, calibrated on "
                   "real faces (happy, sad, angry, surprised, neutral, ...). Webcam mode runs in real time.")
    samples = ("face_astronaut.png", "face_cameraman.png")

    _net = None                                            # loaded once, shared by all instances

    def parameters(self) -> list[Param]:
        return [
            Param("method", "Classifier", "select", "FER+ CNN (offline)",
                  options=("FER+ CNN (offline)", "Gemini (cloud)")),
            Param("calibrated", "Calibrated scores (7 emotions, more accurate)", "checkbox", True,
                  visible_when={"method": "FER+ CNN (offline)"},
                  help="Corrects FER+'s habit of calling sad / angry / fearful faces neutral."),
            Param("min_score", "Face detection confidence", "slider", 0.8, 0.5, 0.95, 0.05),
        ]

    # ----- pipeline -----------------------------------------------------------
    def process(self, image, *, method, min_score, calibrated=True) -> ModuleResult:
        img, _ = utils.resize_max(utils.ensure_bgr(image), 1280)
        faces = FaceDetector.shared().detect(img, min_score)
        if not faces:
            raise ValueError("No face found. Try a clearer, front-facing photo or lower the confidence.")
        found = img.copy()
        for f in faces:
            draw_face(found, f, f"{f.score:.2f}", landmarks=True)
        self._add_step(f"1. Faces found by YuNet ({len(faces)})", found)

        out, rows = img.copy(), []
        for i, face in enumerate(faces, 1):
            crop = face.crop(img, margin=0.1)
            if method.startswith("FER"):
                probs = self.predict(crop, calibrated)
                if i <= 3:
                    gray64 = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), (64, 64))
                    self._add_step(f"2. Face {i}: 64x64 gray input to FER+", utils.enlarge(gray64, 3))
                    self._add_step(f"3. Face {i}: emotion probabilities", self._bars(probs))
            else:
                probs = self.ask_gemini(crop)
            emotion = max(probs, key=probs.get)
            draw_face(out, face, f"{emotion} {probs[emotion]:.0%}", COLOURS[emotion])
            top3 = sorted(probs.items(), key=lambda kv: -kv[1])[:3]
            rows.append({"Face": i, "Emotion": emotion,
                         "Top 3": ", ".join(f"{e} {p:.0%}" for e, p in top3)})
        info = {"Faces": len(faces), "Main emotion": rows[0]["Emotion"], "Per face": rows}
        if method.startswith("FER") and calibrated and self._calibration() is None:
            info["Note"] = f"models/{CALIBRATION.name} not found - raw FER+ scores shown"
        return ModuleResult(out, info=info)

    # ----- classifiers ----------------------------------------------------------
    @classmethod
    def _model(cls):
        if cls._net is None:
            if not FERPLUS.exists():
                raise FileNotFoundError(f"{FERPLUS.name} missing in models/ (from github.com/onnx/models, emotion_ferplus).")
            cls._net = cv2.dnn.readNetFromONNX(str(FERPLUS))
        return cls._net

    _calib = None

    @classmethod
    def _calibration(cls) -> dict | None:
        if cls._calib is None and CALIBRATION.exists():
            cls._calib = json.loads(CALIBRATION.read_text())
        return cls._calib

    def scores(self, face_bgr: np.ndarray) -> np.ndarray:
        """The 8 raw FER+ scores, averaged over the face and its mirror image."""
        gray = cv2.resize(cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY), (64, 64)).astype(np.float32)
        net, out = self._model(), []
        for g in (gray, gray[:, ::-1].copy()):
            net.setInput(g.reshape(1, 1, 64, 64))         # FER+ wants raw 0..255 gray values
            out.append(net.forward().ravel())
        return np.mean(out, axis=0)

    def predict(self, face_bgr: np.ndarray, calibrated: bool = True) -> dict[str, float]:
        scores, calib = self.scores(face_bgr), self._calibration() if calibrated else None
        names = EMOTIONS
        if calib is not None:                              # logistic regression: softmax(W s + b)
            scores = np.array(calib["coef"]) @ scores + np.array(calib["intercept"])
            names = CALIBRATED
        e = np.exp(scores - scores.max())
        return dict(zip(names, (e / e.sum()).tolist()))

    def calibrate(self, folder: Path) -> float:
        """Fit the logistic-regression calibration on labelled face crops (sub-folder per emotion)."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score
        alias = {"surprise": "surprised"}
        X, y = [], []
        for d in sorted(p for p in Path(folder).iterdir() if p.is_dir()):
            label = alias.get(d.name.lower(), d.name.lower())
            if label not in CALIBRATED:
                continue
            for f in sorted(d.iterdir()):
                if f.suffix.lower() in (".jpg", ".jpeg", ".png"):
                    X.append(self.scores(utils.read_image(f)))
                    y.append(label)
        if len(set(y)) < len(CALIBRATED):
            raise ValueError(f"Need one sub-folder per emotion: {', '.join(CALIBRATED)}")
        X, y = np.array(X), np.array(y)
        lr = LogisticRegression(C=1.0, max_iter=3000)
        cv_acc = float(cross_val_score(lr, X, y, cv=5).mean())
        lr.fit(X, y)
        order = [list(lr.classes_).index(c) for c in CALIBRATED]
        CALIBRATION.write_text(json.dumps({"coef": lr.coef_[order].tolist(), "intercept": lr.intercept_[order].tolist(),
                                           "faces": len(y), "cv_accuracy": round(cv_acc, 4)}))
        type(self)._calib = None
        return cv_acc

    @staticmethod
    def ask_gemini(face_bgr: np.ndarray) -> dict[str, float]:
        answer = cloud.gemini(
            "Look at this face. Give the probability (0-1) of each emotion: " + ", ".join(EMOTIONS) +
            ". Reply ONLY with JSON like {\"happy\": 0.8, \"neutral\": 0.1, ...}.", face_bgr)
        match = re.search(r"\{.*\}", answer, re.S)
        if not match:
            raise RuntimeError(f"Unexpected Gemini answer: {answer[:120]}")
        raw = {k.lower(): float(v) for k, v in json.loads(match.group()).items()}
        probs = {e: raw.get(e, 0.0) for e in EMOTIONS}
        total = sum(probs.values()) or 1.0
        return {e: p / total for e, p in probs.items()}

    # ----- drawing / webcam ---------------------------------------------------
    @staticmethod
    def _bars(probs: dict[str, float], width: int = 300) -> np.ndarray:
        canvas = np.full((len(probs) * 22 + 10, width, 3), 30, np.uint8)
        for i, (e, p) in enumerate(probs.items()):
            y = 8 + i * 22
            cv2.putText(canvas, e, (6, y + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (230, 230, 230), 1, cv2.LINE_AA)
            cv2.rectangle(canvas, (90, y), (90 + int(p * (width - 140)), y + 15), COLOURS[e], -1)
            cv2.putText(canvas, f"{p:.0%}", (width - 45, y + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (230, 230, 230), 1, cv2.LINE_AA)
        return canvas

    def run_webcam(self, camera: int = 0) -> None:
        cap = cv2.VideoCapture(camera)
        if not cap.isOpened():
            raise SystemExit("Could not open the webcam.")
        detector = FaceDetector.shared()
        print("Webcam running - press ESC to stop.")
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            for face in detector.detect(frame, 0.8):
                probs = self.predict(face.crop(frame, margin=0.1))
                emotion = max(probs, key=probs.get)
                draw_face(frame, face, f"{emotion} {probs[emotion]:.0%}", COLOURS[emotion])
            cv2.imshow("Emotion recognition (ESC = quit)", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "calibrate":
        print(f"5-fold accuracy: {EmotionRecognizer().calibrate(Path(sys.argv[2])):.1%} -> saved {CALIBRATION}")
    elif "--webcam" in sys.argv:
        EmotionRecognizer().run_webcam()
    else:
        from core.cli import run_cli
        run_cli(EmotionRecognizer)
