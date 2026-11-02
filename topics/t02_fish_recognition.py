"""
Topic 2 - Fish Species Recognition (marine / freshwater study).

Underwater colour correction (classical, optional)
  Water absorbs red light first, so underwater photos look blue-green and hazy.
  1. Red channel compensation (Ancuti et al., 2018): red is boosted where green is strong
  2. Gray-world white balance
  3. CLAHE on the lightness channel (local contrast), light unsharp mask
Species
  Trained model - transfer learning: pretrained ConvNeXt-Tiny features (core.deep) + SVM trained on
                  a folder of labelled fish photos (core.recognition). Sample data: 8 species from
                  the phylo-fish dataset (huggingface.co/datasets/imageomics/phylo-fish) in datasets/fish/.
  Gemini        - zero-shot: names any species, no training (GEMINI_API_KEY in .env)

Run alone:  python -m topics.t02_fish_recognition samples/fish_query_goldfish.jpg
"""
from __future__ import annotations

import json
import re

import cv2
import numpy as np

from core import cloud, utils
from core.base import ModuleResult, Param
from core.deep import CNNEmbedder
from core.recognition import RecognitionModule
from core.registry import register


@register
class FishRecognizer(RecognitionModule):
    topic_no = 2
    name = "Fish Species Recognition"
    description = ("Corrects underwater colour cast (red-channel compensation, white balance, CLAHE) and names "
                   "the species: transfer learning (ConvNeXt-Tiny + SVM) trained on your fish photos, or Gemini.")
    samples = ("fish_underwater_simulated.jpg", "fish_query_goldfish.jpg", "fish_query_common_carp.jpg",
               "fish_query_pickerel.jpg", "fish_query_longnose_gar.jpg", "fish_query_pumpkinseed_sunfish.jpg",
               "fish_query_mosquitofish.jpg", "fish_query_skipjack_herring.jpg", "fish_query_stonecat_catfish.jpg")
    MODEL_FILE = "fish_svm.joblib"
    DEFAULT_DATASET = "datasets/fish"

    def parameters(self) -> list[Param]:
        predict = {"mode": "Predict"}
        return [
            *super().parameters(),                                  # mode + dataset (mode must come first)
            Param("method", "Species classifier", "select", "Trained model (CNN + SVM)",
                  options=("Trained model (CNN + SVM)", "Gemini (cloud, zero-shot)"), visible_when=predict),
            Param("enhance", "Underwater colour correction", "checkbox", False, visible_when=predict),
        ]

    # ----- pipeline -----------------------------------------------------------
    def process(self, image, *, method, enhance, mode, dataset) -> ModuleResult:
        img = utils.ensure_bgr(image)
        if enhance and mode == "Predict":
            img = self.enhance_underwater(img)
        if method.startswith("Gemini") and mode == "Predict":
            return self.ask_gemini(img)
        return super().process(img, mode=mode, dataset=dataset)

    # ----- classical underwater enhancement ---------------------------------------
    def enhance_underwater(self, img: np.ndarray) -> np.ndarray:
        img, _ = utils.resize_max(img, 1000)
        self._add_step("Underwater photo (input)", img)
        f = img.astype(np.float32) / 255
        b, g, r = f[:, :, 0], f[:, :, 1], f[:, :, 2]
        r = r + (g.mean() - r.mean()) * (1 - r) * g                # red channel compensation
        f = np.clip(np.dstack([b, g, r]), 0, 1)
        self._add_step("a. Red channel compensated", (f * 255).astype(np.uint8))
        means = f.reshape(-1, 3).mean(axis=0)
        f = np.clip(f * (means.mean() / (means + 1e-6)), 0, 1)     # gray-world white balance
        out = (f * 255).astype(np.uint8)
        self._add_step("b. Gray-world white balance", out)
        lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = cv2.createCLAHE(2.0, (8, 8)).apply(lab[:, :, 0])
        out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        out = cv2.addWeighted(out, 1.5, cv2.GaussianBlur(out, (0, 0), 2), -0.5, 0)
        self._add_step("c. CLAHE + sharpening (corrected)", out)
        return out

    # ----- trained model hooks ------------------------------------------------------
    def features(self, img: np.ndarray) -> np.ndarray:
        return CNNEmbedder.shared().embed(self.prepare(img))

    def explain(self, img: np.ndarray) -> tuple[np.ndarray, dict]:
        self._add_step("Image given to the classifier", img)
        self._add_step("What the CNN sees (224 x 224 centre crop)", CNNEmbedder.shared().model_input(img))
        return img.copy(), {}

    # ----- Gemini -------------------------------------------------------------------
    def ask_gemini(self, img: np.ndarray) -> ModuleResult:
        answer = cloud.gemini("Identify the fish species in this image. Reply ONLY with JSON: "
                              "{\"common_name\": \"...\", \"scientific_name\": \"...\", \"confidence\": 0.0-1.0, "
                              "\"habitat\": \"freshwater / marine / both\"}", img)
        m = re.search(r"\{.*\}", answer, re.S)
        if not m:
            raise RuntimeError(f"Unexpected Gemini answer: {answer[:120]}")
        data = json.loads(m.group())
        out = img.copy()
        utils.put_label(out, f"{data.get('common_name', '?')} ({float(data.get('confidence', 0)):.0%})",
                        (10, 28), (255, 255, 255), 0.7)
        return ModuleResult(out, info={"Prediction": data.get("common_name", "?"),
                                       "Scientific name": data.get("scientific_name", "?"),
                                       "Confidence": f"{float(data.get('confidence', 0)):.0%}",
                                       "Habitat": data.get("habitat", "?"), "Model": "Gemini (zero-shot)"})


if __name__ == "__main__":
    from core.cli import run_cli
    run_cli(FishRecognizer)
