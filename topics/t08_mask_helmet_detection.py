"""
Topic 8 - Face Mask & Helmet Detection (security / safety compliance check).

For every face found by YuNet (core.faces, its 5 landmarks tell where eyes, nose and mouth are):
Mask (classical colour rule)
  - learn THIS person's skin colour (Cr, Cb in YCrCb) from the cheeks between the eyes and nose
  - look at the nose-to-chin area: if little of it has that skin colour, it is covered -> mask
    (adaptive to every skin tone, no fixed "skin range")
Helmet (classical colour + texture rule)
  - look at the area above the eyebrows: skin = bare forehead, dark + textured = hair,
    smooth and not skin / not hair = a helmet shell
Trained CNN (most accurate): pretrained ConvNeXt-Tiny features (core.deep) + two SVMs, one trained on
  real face crops with / without masks (Kaggle 12k face-mask set) and one on real head crops with /
  without hard hats (hard-hat detection set). Train:
      python -m topics.t08_mask_helmet_detection train-mask <folder with WithMask/ and WithoutMask/>
      python -m topics.t08_mask_helmet_detection train-helmet <folder with hardhat/ and no-hardhat/>
Gemini (cloud) can be used instead as a zero-shot judge of both questions.
A rule ("mask required", "helmet required", "both") colours every face green (OK) or red.

Run alone:  python -m topics.t08_mask_helmet_detection samples/face_astronaut_mask.png
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
from core.faces import Face, FaceDetector, draw_face
from core.recognition import MODELS_DIR, FeatureClassifier
from core.registry import register

MASK_MODEL, HELMET_MODEL = MODELS_DIR / "mask_svm.joblib", MODELS_DIR / "helmet_svm.joblib"
MASK_YES, HELMET_YES = ("withmask", "with_mask", "mask"), ("hardhat", "helmet")


def _cnn(img: np.ndarray) -> np.ndarray:
    from core.deep import CNNEmbedder
    return CNNEmbedder.shared().embed(img)


def head_crop(img: np.ndarray, face: Face) -> np.ndarray:
    """Face box grown upwards and sideways so a helmet (if any) is inside."""
    x, y, w, h = face.box
    H, W = img.shape[:2]
    return img[max(0, y - int(0.7 * h)):min(H, y + int(0.6 * h)), max(0, x - int(0.3 * w)):min(W, x + int(1.3 * w))]


@register
class MaskHelmetDetector(DIPModule):
    topic_no = 8
    name = "Face Mask & Helmet Detection"
    description = ("Checks every face for a mask and a helmet: trained CNN (ConvNeXt-Tiny + SVM on real photos), "
                   "classical colour / texture rules, or Gemini; flags rule violations.")
    samples = ("face_astronaut_mask.png", "face_astronaut_helmet.png", "face_astronaut.png")
    METHODS = ("Trained CNN", "Classical rules", "Gemini (cloud)")
    _models: dict = {}

    def parameters(self) -> list[Param]:
        return [
            Param("method", "Method", "select", self.METHODS[0], options=self.METHODS),
            Param("rule", "Safety rule", "select", "Mask required",
                  options=("Mask required", "Helmet required", "Both required")),
            Param("min_score", "Face detection confidence", "slider", 0.6, 0.3, 0.95, 0.05,
                  help="Masked faces get lower detector scores - keep this fairly low."),
        ]

    # ----- pipeline -----------------------------------------------------------
    def process(self, image, *, method, rule, min_score) -> ModuleResult:
        img, _ = utils.resize_max(utils.ensure_bgr(image), 1280)
        faces = FaceDetector.shared().detect(img, min_score)
        if not faces:
            raise ValueError("No face found. Lower the detection confidence or use a clearer photo.")

        out, rows = img.copy(), []
        for i, face in enumerate(faces, 1):
            if method.startswith("Trained"):
                mask, helmet, scores = self.trained(img, face, show=(i == 1))
            elif method.startswith("Classical"):
                mask, helmet, scores = self.classical(img, face, show=(i == 1))
            else:
                mask, helmet = self.ask_gemini(img, face)
                scores = {}
            need_mask, need_helmet = rule in ("Mask required", "Both required"), rule in ("Helmet required", "Both required")
            ok = (mask or not need_mask) and (helmet or not need_helmet)
            label = f"{'MASK' if mask else 'no mask'} | {'HELMET' if helmet else 'no helmet'}"
            draw_face(out, face, label, (0, 200, 0) if ok else (0, 0, 255))
            rows.append({"Face": i, "Mask": "yes" if mask else "no", "Helmet": "yes" if helmet else "no",
                         "Rule OK": "yes" if ok else "NO", **scores})
        violations = sum(r["Rule OK"] == "NO" for r in rows)
        return ModuleResult(out, info={"Faces": len(faces), "Violations": violations,
                                       "Status": "ALL OK" if violations == 0 else "ALERT", "Per face": rows})

    # ----- trained CNN ----------------------------------------------------------------
    @classmethod
    def _model(cls, path: Path) -> FeatureClassifier:
        if path not in cls._models:
            if not path.exists():
                raise FileNotFoundError(f"{path.name} is not trained yet - see 'train-mask' / 'train-helmet' at the top "
                                        "of topics/t08_mask_helmet_detection.py, or use 'Classical rules'.")
            cls._models[path] = FeatureClassifier.load(path, _cnn)
        return cls._models[path]

    def trained(self, img: np.ndarray, face: Face, show: bool) -> tuple[bool, bool, dict]:
        face_img, head_img = face.crop(img, margin=0.15), head_crop(img, face)
        m_label, m_conf, _ = self._model(MASK_MODEL).predict(face_img)
        h_label, h_conf, _ = self._model(HELMET_MODEL).predict(head_img)
        if show:
            self._add_step("1. Face crop -> mask classifier", face_img)
            self._add_step("2. Head crop -> helmet classifier", head_img)
        mask, helmet = m_label.lower() in MASK_YES, h_label.lower() in HELMET_YES
        return mask, helmet, {"Mask confidence": f"{m_conf:.0%}", "Helmet confidence": f"{h_conf:.0%}"}

    # ----- classical rules ----------------------------------------------------------
    def classical(self, img: np.ndarray, face: Face, show: bool) -> tuple[bool, bool, dict]:
        x, y, w, h = face.box
        eyes_y = int(face.landmarks[:2, 1].mean())
        nose = face.landmarks[2]
        mouth_y = int(face.landmarks[3:, 1].mean())
        H, W = img.shape[:2]
        ycc = cv2.cvtColor(cv2.GaussianBlur(img, (5, 5), 0), cv2.COLOR_BGR2YCrCb).astype(np.int16)
        clip = lambda x0, y0, x1, y1: (max(0, x0), max(0, y0), min(W, x1), min(H, y1))

        # skin reference: the two cheeks, between eye level and nose level (outside the nose)
        ey, ny = eyes_y + int(0.05 * h), int(nose[1])
        cheeks = [clip(x + int(0.12 * w), ey, x + int(0.32 * w), ny), clip(x + int(0.68 * w), ey, x + int(0.88 * w), ny)]
        ref = np.vstack([ycc[b:d, a:c, 1:].reshape(-1, 2) for a, b, c, d in cheeks])
        cr, cb = np.median(ref, axis=0) if len(ref) else (150, 110)

        def skin_share(a, b, c, d):
            region = ycc[b:d, a:c]
            if region.size == 0:
                return 0.0
            close = (np.abs(region[:, :, 1] - cr) < 10) & (np.abs(region[:, :, 2] - cb) < 10)
            return float(close.mean())

        lower = clip(x + int(0.2 * w), int(nose[1] - 0.05 * h), x + int(0.8 * w), min(y + h, mouth_y + int(0.15 * h)))
        lower_skin = skin_share(*lower)
        mask = lower_skin < 0.35

        # helmet: band above the eyebrows up to the top of the head
        top = clip(x + int(0.1 * w), y - int(0.35 * h), x + int(0.9 * w), eyes_y - int(0.2 * h))
        a, b, c, d = top
        head = img[b:d, a:c]
        helmet, helmet_share = False, 0.0
        if head.size:
            hsv = cv2.cvtColor(head, cv2.COLOR_BGR2HSV)
            gray = cv2.cvtColor(head, cv2.COLOR_BGR2GRAY).astype(np.float32)
            texture = np.abs(cv2.Laplacian(cv2.GaussianBlur(gray, (3, 3), 0), cv2.CV_32F))
            region = ycc[b:d, a:c]
            skin = (np.abs(region[:, :, 1] - cr) < 10) & (np.abs(region[:, :, 2] - cb) < 10)
            hair = (hsv[:, :, 2] < 80)
            smooth = cv2.blur(texture, (7, 7)) < 6
            helmet_share = float((~skin & ~hair & smooth).mean())
            helmet = helmet_share > 0.5

        if show:
            vis = img.copy()
            for (a, b, c, d), colour in [(cheeks[0], (255, 200, 0)), (cheeks[1], (255, 200, 0)),
                                         (lower, (0, 0, 255)), (top, (0, 255, 255))]:
                cv2.rectangle(vis, (a, b), (c, d), colour, 2)
            self._add_step("1. Regions: cheeks = skin reference (blue), nose-to-chin (red), head top (yellow)", vis)
            sk = ((np.abs(ycc[:, :, 1] - cr) < 10) & (np.abs(ycc[:, :, 2] - cb) < 10)).astype(np.uint8) * 255
            self._add_step("2. Pixels with this person's skin colour", sk)
        return mask, helmet, {"Skin below nose": f"{lower_skin:.0%}", "Helmet-like head area": f"{helmet_share:.0%}"}

    # ----- Gemini -------------------------------------------------------------------
    @staticmethod
    def ask_gemini(img: np.ndarray, face: Face) -> tuple[bool, bool]:
        x, y, w, h = face.box
        H, W = img.shape[:2]
        crop = img[max(0, y - int(0.8 * h)):min(H, y + int(1.3 * h)), max(0, x - int(0.5 * w)):min(W, x + int(1.5 * w))]
        answer = cloud.gemini("Is this person WEARING a face mask (covering nose and mouth)? Is this person WEARING "
                              "a helmet or hard hat on the head (holding one does not count)? Reply ONLY with JSON: "
                              "{\"mask\": true/false, \"helmet\": true/false}", crop)
        m = re.search(r"\{.*\}", answer, re.S)
        if not m:
            raise RuntimeError(f"Unexpected Gemini answer: {answer[:120]}")
        data = json.loads(m.group().lower())
        return bool(data.get("mask")), bool(data.get("helmet"))


def train(kind: str, folder: Path) -> None:
    """Train the mask or helmet SVM on a folder with one sub-folder per class."""
    clf = FeatureClassifier(_cnn)
    report = clf.train_folder(folder)
    target = MASK_MODEL if kind == "mask" else HELMET_MODEL
    clf.save(target)
    print(f"{kind}: {report['images']} images, classes {report['classes']}, "
          f"test accuracy {report.get('accuracy', float('nan')):.1%} -> {target.name}")


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] in ("train-mask", "train-helmet"):
        train(sys.argv[1].split("-")[1], Path(sys.argv[2]))
    else:
        from core.cli import run_cli
        run_cli(MaskHelmetDetector)
