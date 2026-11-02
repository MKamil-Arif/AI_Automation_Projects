"""
Topic 14 - Waste Segregation: Recyclable vs Non-Recyclable (plastic, paper, metal, glass, ...).

Transfer learning
  1. ConvNeXt-Tiny (pretrained on ImageNet, core.deep) turns the photo into a 768-number feature
     vector - it already "knows" edges, textures, shiny surfaces, transparency, ...
  2. An RBF-SVM learns our waste classes from those vectors (core.recognition)
  3. The class decides the bin: recyclable (paper, cardboard, plastic, metal, glass) or general waste

Dataset: TrashNet (Stanford, 2527 photos, 6 classes: cardboard, glass, metal, paper, plastic, trash)
from github.com/garythung/trashnet -> datasets/trashnet/. TrashNet has no organic class; add a
folder "organic" with your own photos and train again to get it.
Run alone:  python -m topics.t14_waste_segregation photo.jpg   ("mode=Train from folder" to train)
"""
from __future__ import annotations

import cv2
import numpy as np

from core.deep import CNNEmbedder
from core.recognition import RecognitionModule
from core.registry import register

# class -> (recyclable?, bin advice)
BINS = {
    "cardboard": (True, "Blue bin - paper & cardboard"),
    "paper": (True, "Blue bin - paper & cardboard"),
    "plastic": (True, "Yellow bin - plastic & metal"),
    "metal": (True, "Yellow bin - plastic & metal"),
    "glass": (True, "Green bin - glass"),
    "organic": (False, "Brown bin - organic / compost"),
    "trash": (False, "Black bin - general waste"),
}


@register
class WasteSegregation(RecognitionModule):
    topic_no = 14
    name = "Waste Segregation"
    description = ("Classifies a waste item (cardboard, glass, metal, paper, plastic, trash) with transfer "
                   "learning - pretrained ConvNeXt-Tiny features + SVM trained on TrashNet - and suggests the bin.")
    samples = ("waste_query_plastic.jpg", "waste_query_glass.jpg", "waste_query_paper.jpg",
               "waste_query_metal.jpg", "waste_query_cardboard.jpg", "waste_query_trash.jpg")
    MODEL_FILE = "waste_svm.joblib"
    DEFAULT_DATASET = "datasets/trashnet"
    WORK_SIZE = 384

    def features(self, img: np.ndarray) -> np.ndarray:
        return CNNEmbedder.shared().embed(self.prepare(img))

    def explain(self, img: np.ndarray) -> tuple[np.ndarray, dict]:
        self._add_step("Original", img)
        crop = CNNEmbedder.shared().model_input(img)
        self._add_step("1. What the CNN sees (224 x 224 centre crop)", crop)
        edges = cv2.Canny(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), 60, 160)
        self._add_step("2. Edges (the CNN's first layers respond to these)", edges)
        return img.copy(), {}

    def process(self, image, *, mode, dataset):
        result = super().process(image, mode=mode, dataset=dataset)
        label = result.info.get("Prediction")
        if label is not None:                                    # add the bin advice to a prediction
            recyclable, advice = BINS.get(label, (False, "Check local rules"))
            result.info = {"Prediction": label, "Recyclable": "YES" if recyclable else "NO",
                           "Bin": advice, **{k: v for k, v in result.info.items() if k != "Prediction"}}
        return result


if __name__ == "__main__":
    from core.cli import run_cli
    run_cli(WasteSegregation)
