"""
A pretrained CNN used as a feature extractor ("transfer learning").

The network was trained on ImageNet (1.2 million photos, 1000 classes). We cut off its last
classification layer and keep the feature vector it computes for an image: similar-looking
objects get similar vectors, so a small classifier (SVM) trained on a few hundred of our own
photos is enough. Weights are downloaded once into models/torch/.

Backbones
  mobilenet_v3_large  - 960 numbers, fast (ImageNet top-1 75.3 %, 21 MB)
  convnext_tiny       - 768 numbers, stronger (ImageNet top-1 82.5 %, 109 MB)
"""
from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np

TORCH_HOME = Path(__file__).resolve().parent.parent / "models" / "torch"
DEFAULT_BACKBONE = "convnext_tiny"


class CNNEmbedder:
    _shared: dict[str, "CNNEmbedder"] = {}

    def __init__(self, backbone: str = DEFAULT_BACKBONE) -> None:
        os.environ.setdefault("TORCH_HOME", str(TORCH_HOME))      # where torchvision keeps the weights
        import torch
        from torchvision import models

        if backbone == "mobilenet_v3_large":
            weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V2
            model = models.mobilenet_v3_large(weights=weights)
            model.classifier = torch.nn.Identity()               # keep the 960-d pooled features
        elif backbone == "convnext_tiny":
            weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1
            model = models.convnext_tiny(weights=weights)
            model.classifier[2] = torch.nn.Identity()            # keep the 768-d normalised features
        else:
            raise ValueError(f"Unknown backbone {backbone}")
        self.backbone = backbone
        self.model = model.eval()
        self.transform = weights.transforms()                    # resize, centre crop 224, normalise
        torch.set_num_threads(max(1, os.cpu_count() or 1))

    @classmethod
    def shared(cls, backbone: str = DEFAULT_BACKBONE) -> "CNNEmbedder":
        if backbone not in cls._shared:
            cls._shared[backbone] = cls(backbone)
        return cls._shared[backbone]

    def embed(self, img_bgr: np.ndarray) -> np.ndarray:
        import torch
        from PIL import Image

        pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        with torch.no_grad():
            vec = self.model(self.transform(pil)[None])[0].numpy().astype(np.float64)
        return vec / (np.linalg.norm(vec) + 1e-9)                # unit length: only the "direction" matters

    def model_input(self, img_bgr: np.ndarray) -> np.ndarray:
        """The 224 x 224 crop the CNN actually sees (for 'Behind the scenes')."""
        h, w = img_bgr.shape[:2]
        s = 232 / min(h, w)
        small = cv2.resize(img_bgr, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
        y, x = (small.shape[0] - 224) // 2, (small.shape[1] - 224) // 2
        return small[y:y + 224, x:x + 224]
