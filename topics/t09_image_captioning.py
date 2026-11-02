"""
Topic 9 - Image Caption Generator ("A dog is sitting on grass").

Captioning models
  BLIP (offline)  - Salesforce BLIP-base: a Vision Transformer encodes the image, a text decoder
                    writes the sentence word by word (beam search). Runs on the CPU, no internet
                    after the first download (~1 GB into models/hf/).
  Gemini (cloud)  - Google's multimodal model through the API key in .env (short caption or a
                    detailed description).
Urdu: the English caption is translated by a Groq LLM (GROQ_API_KEY in .env).
Classical DIP part: the main colours of the picture (k-means clustering in Lab colour space).

Run alone:  python -m topics.t09_image_captioning samples/flower_query_sunflower.jpg
"""
from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

from core import cloud, utils
from core.base import DIPModule, ModuleResult, Param
from core.registry import register

HF_CACHE = Path(__file__).resolve().parent.parent / "models" / "hf"
BLIP_NAME = "Salesforce/blip-image-captioning-base"

# a few named colours (Lab is close to how people see colour differences)
NAMED_COLOURS = {"black": (20, 20, 20), "white": (245, 245, 245), "gray": (128, 128, 128),
                 "dark gray": (65, 65, 65), "red": (40, 40, 200),
                 "orange": (30, 140, 245), "yellow": (40, 220, 240), "green": (50, 160, 60), "dark green": (30, 80, 35),
                 "blue": (200, 110, 40), "sky blue": (235, 200, 135), "purple": (150, 60, 130), "pink": (190, 150, 240),
                 "brown": (40, 75, 120), "beige": (180, 210, 230)}


@register
class ImageCaptioner(DIPModule):
    topic_no = 9
    name = "Image Caption Generator"
    description = ("Writes a sentence describing the image: BLIP (offline vision transformer) or Gemini "
                   "(cloud), optional Urdu translation via Groq, plus the main colours (k-means).")
    samples = ("flower_query_sunflower.jpg", "orchard_aerial_synthetic.png", "portrait_room_synthetic.png",
               "road_synthetic.png", "face_astronaut.png")

    _blip = None                                           # (processor, model), loaded once

    def parameters(self) -> list[Param]:
        return [
            Param("backend", "Caption model", "select", "BLIP (offline)", options=("BLIP (offline)", "Gemini (cloud)")),
            Param("style", "Style", "select", "Short caption", options=("Short caption", "Detailed description")),
            Param("urdu", "Also in Urdu (Groq)", "checkbox", False),
        ]

    # ----- pipeline -----------------------------------------------------------
    def process(self, image, *, backend, style, urdu) -> ModuleResult:
        img, _ = utils.resize_max(utils.ensure_bgr(image), 1024)
        self._add_step("Input", img)
        colours, palette = self.main_colours(img)
        self._add_step("Main colours (k-means, k = 5, Lab space)", palette)

        if backend.startswith("BLIP"):
            self._load_blip()                                      # first time ~40 s; not part of the timing
        start = time.time()
        if backend.startswith("BLIP"):
            caption = self.caption_blip(img, detailed=style.startswith("Detailed"))
        else:
            prompt = ("Write ONE short caption (max 15 words) for this image." if style.startswith("Short")
                      else "Describe this image in 3-4 sentences: the main objects, the scene, colours and mood.")
            caption = cloud.gemini(prompt, img)
        seconds = time.time() - start

        info = {"Caption": caption}
        if urdu:
            info["Urdu"] = cloud.groq("Translate this image caption to Urdu. Reply with the Urdu text only:\n" + caption)
        info.update({"Model": backend, "Time": f"{seconds:.1f} s", "Main colours": ", ".join(colours)})
        return ModuleResult(self._with_caption(img, caption), info=info)

    # ----- BLIP -------------------------------------------------------------------
    @classmethod
    def _load_blip(cls):
        if cls._blip is None:
            from transformers import BlipForConditionalGeneration, BlipProcessor
            processor = BlipProcessor.from_pretrained(BLIP_NAME, cache_dir=HF_CACHE)
            model = BlipForConditionalGeneration.from_pretrained(BLIP_NAME, cache_dir=HF_CACHE).eval()
            cls._blip = (processor, model)
        return cls._blip

    def caption_blip(self, img: np.ndarray, detailed: bool) -> str:
        import torch

        processor, model = self._load_blip()
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        inputs = processor(images=rgb, return_tensors="pt")
        self._add_step("BLIP input (384 x 384, normalised)", cv2.resize(img, (384, 384)))
        with torch.no_grad():
            ids = model.generate(**inputs, num_beams=5, max_new_tokens=40 if detailed else 20,
                                 min_new_tokens=15 if detailed else 3, repetition_penalty=1.3)
        text = processor.decode(ids[0], skip_special_tokens=True).strip()
        return text[:1].upper() + text[1:] + ("" if text.endswith(".") else ".")

    # ----- classical part: dominant colours ------------------------------------
    @staticmethod
    def main_colours(img: np.ndarray, k: int = 5) -> tuple[list[str], np.ndarray]:
        small, _ = utils.resize_max(img, 160)
        lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
        cv2.setRNGSeed(0)
        _, labels, centres = cv2.kmeans(lab, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
        share = np.bincount(labels.ravel(), minlength=k) / len(labels)
        order = np.argsort(share)[::-1]
        bgr = cv2.cvtColor(centres[order].reshape(1, -1, 3).astype(np.uint8), cv2.COLOR_LAB2BGR).reshape(-1, 3)
        names_lab = {n: cv2.cvtColor(np.uint8([[c]]), cv2.COLOR_BGR2LAB)[0, 0].astype(np.float32)
                     for n, c in NAMED_COLOURS.items()}
        names = []
        for c in centres[order]:
            name = min(names_lab, key=lambda n: np.linalg.norm(names_lab[n] - c))
            if name not in names:
                names.append(name)
        palette = np.zeros((60, 400, 3), np.uint8)
        x = 0
        for colour, s in zip(bgr, share[order]):
            w = int(round(s * 400))
            palette[:, x:x + w] = colour
            x += w
        palette[:, x:] = bgr[-1]
        return names[:3], palette

    @staticmethod
    def _with_caption(img: np.ndarray, caption: str) -> np.ndarray:
        """Image with the (English) caption written in a band underneath, word-wrapped."""
        w = img.shape[1]
        font, scale = cv2.FONT_HERSHEY_SIMPLEX, max(0.55, w / 1100)
        lines, line = [], ""
        for word in caption.encode("ascii", "replace").decode().split():
            trial = f"{line} {word}".strip()
            if cv2.getTextSize(trial, font, scale, 1)[0][0] > w - 24 and line:
                lines.append(line)
                line = word
            else:
                line = trial
        lines.append(line)
        line_h = int(34 * scale) + 6
        band = np.full((line_h * len(lines) + 16, w, 3), 25, np.uint8)
        for i, text in enumerate(lines):
            cv2.putText(band, text, (12, 8 + line_h * (i + 1) - 8), font, scale, (255, 255, 255), 1, cv2.LINE_AA)
        return np.vstack([img, band])


if __name__ == "__main__":
    from core.cli import run_cli
    run_cli(ImageCaptioner)
