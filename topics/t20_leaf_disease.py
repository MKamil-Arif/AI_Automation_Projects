"""
Topic 20 - Plant Leaf Disease Detection (Healthy, Rust, Blight, Scab, ... on 14 crops, 38 classes).

  1. Disease name: MobileNetV2 CNN fine-tuned on PlantVillage (54,000 leaf photos, 38 classes),
     ONNX file (9 MB) from huggingface.co/onnx-community, run with OpenCV's DNN module
  2. How bad is it? (classical DIP) - the leaf is segmented (saturation + Otsu, biggest blob);
     inside it, pixels that are not healthy green (brown / yellow / black spots) are the lesions;
     severity = lesion area / leaf area
  3. Optional treatment advice from a Groq LLM (GROQ_API_KEY in .env), in English or Urdu

Evaluate mode measures the CNN's accuracy on a folder with one sub-folder per class
(datasets/plantvillage_eval = 3 photos of each of the 38 classes from PlantVillage).
Run alone:  python -m topics.t20_leaf_disease samples/leafdisease_tomato_late_blight.jpg
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from core import cloud, utils
from core.base import DIPModule, ModuleResult, Param
from core.registry import register

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models" / "plant_disease"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@register
class LeafDiseaseDetector(DIPModule):
    topic_no = 20
    name = "Plant Leaf Disease Detection"
    description = ("Names the disease of a leaf (MobileNetV2 trained on PlantVillage, 38 classes), measures how "
                   "much of the leaf is affected (classical lesion segmentation) and can give treatment tips.")
    samples = tuple(f"leafdisease_{n}.jpg" for n in (
        "tomato_late_blight", "tomato_healthy", "potato_early_blight", "apple_apple_scab",
        "corn_maize_common_rust", "grape_black_rot", "pepper_bell_bacterial_spot", "strawberry_leaf_scorch"))
    confirm_modes = ("Evaluate on folder",)

    _net, _labels = None, None

    def parameters(self) -> list[Param]:
        return [
            Param("mode", "Mode", "select", "Diagnose", options=("Diagnose", "Evaluate on folder")),
            Param("dataset", "Folder (one sub-folder per class, alphabetical = model order)", "text",
                  "datasets/plantvillage_eval", visible_when={"mode": "Evaluate on folder"}),
            Param("advice", "Treatment advice (Groq)", "select", "Off", options=("Off", "English", "Urdu"),
                  visible_when={"mode": "Diagnose"}),
        ]

    # ----- pipeline -----------------------------------------------------------
    def process(self, image, *, mode, dataset, advice) -> ModuleResult:
        if mode == "Evaluate on folder":
            return self.evaluate(dataset)
        img, _ = utils.resize_max(utils.ensure_bgr(image), 800)
        self._add_step("Original", img)
        probs = self.predict(img)
        order = np.argsort(probs)[::-1]
        label = self._labels[order[0]]

        leaf, lesions = self.lesions(img)
        severity = lesions.sum() / max(leaf.sum(), 1)
        healthy = "healthy" in label.lower()
        out = img.copy()
        contours, _ = cv2.findContours(lesions.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, (0, 0, 255), 1, cv2.LINE_AA)
        utils.put_label(out, f"{label} ({probs[order[0]]:.0%})", (8, 24), (255, 255, 255), 0.6)

        info = {
            "Diagnosis": label,
            "Confidence": f"{probs[order[0]]:.0%}",
            "Affected area": f"{severity:.1%}",
            "Severity": self.grade(severity) if not healthy else "none (healthy)",
            "Top 3": [{"Class": self._labels[i], "Probability": f"{probs[i]:.1%}"} for i in order[:3]],
        }
        if advice != "Off" and not healthy:
            info["Advice"] = cloud.groq(
                f"A farmer's leaf was diagnosed as '{label}' with about {severity:.0%} of the leaf affected. "
                f"Give 3 short, practical, well-known treatment / prevention tips (no brand names). "
                f"Answer in {advice}, max 80 words.", model=cloud.GROQ_MODELS[1])
            info["Note"] = "AI-generated advice - confirm with a local agriculture expert before spraying."
        return ModuleResult(out, info=info)

    # ----- CNN ----------------------------------------------------------------------
    @classmethod
    def _load(cls):
        if cls._net is None:
            model = MODEL_DIR / "model.onnx"
            if not model.exists():
                raise FileNotFoundError("models/plant_disease/model.onnx missing (huggingface.co/onnx-community/"
                                        "mobilenet_v2_1.0_224-plant-disease-identification-ONNX).")
            cls._net = cv2.dnn.readNetFromONNX(str(model))
            labels = json.loads((MODEL_DIR / "config.json").read_text())["id2label"]
            cls._labels = [labels[str(i)] for i in range(len(labels))]
        return cls._net

    def predict(self, img: np.ndarray) -> np.ndarray:
        """Same preprocessing as in training: shortest side 256, centre crop 224, scale to -1..1."""
        net = self._load()
        h, w = img.shape[:2]
        s = 256 / min(h, w)
        resized = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_LINEAR)
        y, x = (resized.shape[0] - 224) // 2, (resized.shape[1] - 224) // 2
        crop = resized[y:y + 224, x:x + 224]
        self._add_step("1. CNN input (224 x 224 centre crop)", crop)
        blob = cv2.dnn.blobFromImage(crop, scalefactor=1 / 127.5, mean=(127.5, 127.5, 127.5), swapRB=True)
        net.setInput(blob)
        logits = net.forward().ravel()
        e = np.exp(logits - logits.max())
        return e / e.sum()

    # ----- classical lesion analysis ---------------------------------------------------
    def lesions(self, img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = (hsv[:, :, i].astype(np.int16) for i in range(3))
        _, leaf = cv2.threshold(cv2.GaussianBlur(hsv[:, :, 1], (5, 5), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        leaf = cv2.morphologyEx(leaf, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        contours, _ = cv2.findContours(leaf, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros_like(leaf)
        if contours:
            cv2.drawContours(mask, [max(contours, key=cv2.contourArea)], -1, 255, -1)   # fills the spots too
        leaf = mask > 0
        self._add_step("2. Leaf mask (saturation + Otsu, holes filled)", mask)

        healthy_green = (h >= 28) & (h <= 90) & (s >= 50) & (v >= 40)
        lesions = leaf & ~healthy_green
        lesions = cv2.morphologyEx(lesions.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
        vis = img.copy()
        vis[lesions] = (0.4 * vis[lesions] + (0, 0, 150)).astype(np.uint8)
        vis[~leaf] = (vis[~leaf] * 0.3).astype(np.uint8)
        self._add_step("3. Lesions = leaf pixels that are not healthy green", vis)
        return leaf, lesions

    @staticmethod
    def grade(fraction: float) -> str:
        return "mild" if fraction < 0.1 else "moderate" if fraction < 0.25 else "severe"

    # ----- evaluation ------------------------------------------------------------------
    def evaluate(self, dataset: str) -> ModuleResult:
        folder = Path(dataset) if Path(dataset).is_absolute() else PROJECT_ROOT / dataset
        # case-sensitive name order (like Keras / torchvision ImageFolder): on Windows, sorting Path
        # objects ignores case and would put "healthy" before "Powdery_mildew"
        classes = sorted((p for p in folder.iterdir() if p.is_dir()), key=lambda p: p.name) if folder.is_dir() else []
        self._load()
        if len(classes) != len(self._labels):
            raise ValueError(f"Expected {len(self._labels)} class folders in {folder}, found {len(classes)}.")
        correct = total = 0
        rows = []
        for idx, cls in enumerate(classes):
            hits = n = 0
            for file in sorted(cls.iterdir()):
                if file.suffix.lower() in IMAGE_EXTS:
                    hits += int(np.argmax(self.predict(utils.read_image(file))) == idx)
                    n += 1
            correct, total = correct + hits, total + n
            rows.append({"Class": self._labels[idx], "Images": n, "Correct": hits})
        canvas = np.full((60, 420, 3), 250, np.uint8)
        cv2.putText(canvas, f"Accuracy {correct / max(total, 1):.1%} ({correct}/{total})", (10, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30, 30, 30), 2, cv2.LINE_AA)
        return ModuleResult(canvas, info={"Accuracy": f"{correct / max(total, 1):.1%}", "Images": total,
                                          "Classes": len(classes), "Per class": rows})


if __name__ == "__main__":
    from core.cli import run_cli
    run_cli(LeafDiseaseDetector)
