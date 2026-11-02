"""
Topic 15 - Smart Auto Photo Editor (a mini Snapseed).

Auto mode : gray-world white balance -> percentile contrast stretch -> gamma auto-brightness
            -> histogram equalization (global / CLAHE) -> denoise -> unsharp-mask sharpening
Manual    : brightness, contrast, saturation, gamma, sharpness sliders
Filters   : grayscale, sepia, pencil sketch, cartoon, vignette

Run alone:  python -m topics.t15_smart_photo_editor <image> mode=Manual brightness=30
"""
from __future__ import annotations

import cv2
import numpy as np

from core import utils
from core.base import DIPModule, ModuleResult, Param
from core.registry import register


@register
class SmartPhotoEditor(DIPModule):
    topic_no = 15
    name = "Smart Auto Photo Editor"
    description = ("One-click auto enhance (white balance, contrast, brightness, histogram "
                   "equalization, sharpening) plus manual sliders and creative filters.")

    FILTERS = ("None", "Grayscale", "Sepia", "Pencil Sketch", "Cartoon", "Vignette")

    def parameters(self) -> list[Param]:
        auto = {"mode": "Auto Enhance"}
        manual = {"mode": "Manual"}
        return [
            Param("mode", "Mode", "select", "Auto Enhance", options=("Auto Enhance", "Manual")),
            Param("white_balance", "Auto white balance", "checkbox", True, visible_when=auto,
                  help="Gray-world assumption: average colour of a scene should be gray."),
            Param("equalization", "Histogram equalization", "select", "CLAHE",
                  options=("None", "Global", "CLAHE"), visible_when=auto,
                  help="CLAHE = local equalization, avoids the over-done look of global HE."),
            Param("denoise", "Denoise", "checkbox", False, visible_when=auto),
            Param("sharpen", "Sharpen", "checkbox", True, visible_when=auto),
            Param("brightness", "Brightness", "slider", 0, -100, 100, 1, visible_when=manual),
            Param("contrast", "Contrast", "slider", 1.0, 0.5, 2.0, 0.05, visible_when=manual),
            Param("saturation", "Saturation", "slider", 1.0, 0.0, 2.0, 0.05, visible_when=manual),
            Param("gamma", "Gamma (>1 brightens shadows)", "slider", 1.0, 0.3, 3.0, 0.05,
                  visible_when=manual),
            Param("sharpness", "Sharpness", "slider", 0.0, 0.0, 3.0, 0.1, visible_when=manual),
            Param("effect", "Creative filter", "select", "None", options=self.FILTERS),
        ]

    # ----- pipeline -----------------------------------------------------------
    def process(self, image, *, mode, white_balance, equalization, denoise, sharpen,
                brightness, contrast, saturation, gamma, sharpness, effect) -> ModuleResult:
        original = utils.ensure_bgr(image)
        img = original.copy()
        applied: list[str] = []
        self._add_step("Original", img)
        self._add_step("Original histogram", utils.histogram_image(img))

        if mode == "Auto Enhance":
            if white_balance:
                img = self.white_balance(img)
                applied.append("White balance")
                self._add_step("1. Gray-world white balance", img)
            img = self.auto_contrast(img)
            applied.append("Contrast stretch")
            self._add_step("2. Contrast stretch (1%-99% of L channel)", img)
            img, g = self.auto_brightness(img)
            if abs(g - 1) > 0.01:
                applied.append(f"Gamma {g:.2f}")
            self._add_step(f"3. Auto brightness (gamma = {g:.2f})", img)
            if equalization == "Global":
                img = self.equalize(img)
                applied.append("Global HE")
                self._add_step("4. Global histogram equalization", img)
            elif equalization == "CLAHE":
                img = self.clahe(img)
                applied.append("CLAHE")
                self._add_step("4. CLAHE (local equalization)", img)
            if denoise:
                img = self.denoise(img)
                applied.append("Denoise")
                self._add_step("5. Non-local means denoising", img)
            if sharpen:
                img = self.unsharp_mask(img, amount=0.8)
                applied.append("Sharpen")
                self._add_step("6. Unsharp-mask sharpening", img)
        else:
            if brightness != 0 or contrast != 1.0:
                img = self.brightness_contrast(img, brightness, contrast)
                applied.append(f"Brightness {brightness:+d}, contrast x{contrast:.2f}")
                self._add_step("Brightness / contrast", img)
            if gamma != 1.0:
                img = self.gamma_correction(img, gamma)
                applied.append(f"Gamma {gamma:.2f}")
                self._add_step(f"Gamma {gamma:.2f}", img)
            if saturation != 1.0:
                img = self.adjust_saturation(img, saturation)
                applied.append(f"Saturation x{saturation:.2f}")
                self._add_step("Saturation", img)
            if sharpness > 0:
                img = self.unsharp_mask(img, amount=sharpness)
                applied.append(f"Sharpness {sharpness:.1f}")
                self._add_step("Sharpness (unsharp mask)", img)

        if effect != "None":
            img = self.apply_filter(img, effect)
            applied.append(f"{effect} filter")
            self._add_step(f"Filter: {effect}", img)

        self._add_step("Final histogram", utils.histogram_image(img))

        before, after = self._luma_stats(original), self._luma_stats(img)
        return ModuleResult(img, info={
            "Brightness": f"{before[0]:.0f} -> {after[0]:.0f}",
            "Contrast (std)": f"{before[1]:.0f} -> {after[1]:.0f}",
            "Operations": ", ".join(applied) or "None",
        })

    # ----- auto tools (each can be used on its own) ---------------------------
    @staticmethod
    def white_balance(img: np.ndarray) -> np.ndarray:
        """Gray world: scale each channel so its mean equals the overall mean.

        Means are taken over near-neutral pixels only (low saturation), otherwise a photo
        full of green leaves would be "corrected" towards magenta. Gains are capped too.
        """
        f = img.astype(np.float32)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        neutral = (hsv[:, :, 1] < 60) & (hsv[:, :, 2] > 30) & (hsv[:, :, 2] < 245)
        pixels = f[neutral] if neutral.mean() > 0.05 else f.reshape(-1, 3)
        means = pixels.mean(axis=0)
        gains = np.clip(means.mean() / (means + 1e-6), 0.75, 1.33)
        return np.clip(f * gains, 0, 255).astype(np.uint8)

    @staticmethod
    def auto_contrast(img: np.ndarray, clip_percent: float = 1.0) -> np.ndarray:
        """Stretch the lightness channel so 1%..99% of pixels cover the full 0..255 range."""
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        L = lab[:, :, 0].astype(np.float32)
        lo, hi = np.percentile(L, (clip_percent, 100 - clip_percent))
        if hi - lo > 10:
            lab[:, :, 0] = np.clip((L - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    @staticmethod
    def auto_brightness(img: np.ndarray) -> tuple[np.ndarray, float]:
        """Pick a gamma that pulls a too-dark / too-bright image towards mid-gray.
        Dark photos may be brightened a lot; bright ones (e.g. white backgrounds) only a little."""
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        mean = np.clip(lab[:, :, 0].mean() / 255.0, 0.02, 0.98)
        target = np.clip(mean, 0.40, 0.65)            # only correct if outside this band
        g = float(np.clip(np.log(target) / np.log(mean), 0.4, 1.2))
        lut = (np.linspace(0, 1, 256) ** g * 255).astype(np.uint8)
        lab[:, :, 0] = cv2.LUT(lab[:, :, 0], lut)
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR), g

    @staticmethod
    def equalize(img: np.ndarray) -> np.ndarray:
        """Global histogram equalization on luminance only (keeps colours natural)."""
        ycc = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        ycc[:, :, 0] = cv2.equalizeHist(ycc[:, :, 0])
        return cv2.cvtColor(ycc, cv2.COLOR_YCrCb2BGR)

    @staticmethod
    def clahe(img: np.ndarray, clip_limit: float = 2.0, grid: int = 8) -> np.ndarray:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = cv2.createCLAHE(clip_limit, (grid, grid)).apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    @staticmethod
    def denoise(img: np.ndarray, strength: int = 5) -> np.ndarray:
        return cv2.fastNlMeansDenoisingColored(img, None, strength, strength, 7, 21)

    @staticmethod
    def unsharp_mask(img: np.ndarray, amount: float = 1.0, sigma: float = 2.0) -> np.ndarray:
        """sharp = original + amount * (original - blurred)"""
        blurred = cv2.GaussianBlur(img, (0, 0), sigma)
        return cv2.addWeighted(img, 1 + amount, blurred, -amount, 0)

    # ----- manual tools -------------------------------------------------------
    @staticmethod
    def brightness_contrast(img: np.ndarray, brightness: int, contrast: float) -> np.ndarray:
        out = (img.astype(np.float32) - 128) * contrast + 128 + brightness
        return np.clip(out, 0, 255).astype(np.uint8)

    @staticmethod
    def gamma_correction(img: np.ndarray, gamma: float) -> np.ndarray:
        lut = (np.linspace(0, 1, 256) ** (1.0 / gamma) * 255).astype(np.uint8)
        return cv2.LUT(img, lut)

    @staticmethod
    def adjust_saturation(img: np.ndarray, factor: float) -> np.ndarray:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # ----- creative filters ---------------------------------------------------
    def apply_filter(self, img: np.ndarray, name: str) -> np.ndarray:
        filters = {
            "Grayscale": lambda x: cv2.cvtColor(cv2.cvtColor(x, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR),
            "Sepia": self.sepia,
            "Pencil Sketch": self.pencil_sketch,
            "Cartoon": self.cartoon,
            "Vignette": self.vignette,
        }
        if name not in filters:
            raise ValueError(f"Unknown filter '{name}'")
        return filters[name](img)

    @staticmethod
    def sepia(img: np.ndarray) -> np.ndarray:
        kernel = np.array([[0.131, 0.534, 0.272],      # rows = output B, G, R
                           [0.168, 0.686, 0.349],      # cols = input  B, G, R
                           [0.189, 0.769, 0.393]])
        return np.clip(cv2.transform(img.astype(np.float32), kernel), 0, 255).astype(np.uint8)

    @staticmethod
    def pencil_sketch(img: np.ndarray) -> np.ndarray:
        """Colour-dodge blend of the gray image with its blurred negative."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(255 - gray, (21, 21), 0)
        sketch = cv2.divide(gray, 255 - blur, scale=256)
        return cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def cartoon(img: np.ndarray) -> np.ndarray:
        """Flat colours (bilateral filter + posterize) and bold edges (adaptive threshold)."""
        color = img
        for _ in range(2):
            color = cv2.bilateralFilter(color, 9, 75, 75)
        color = (color // 32) * 32 + 16
        gray = cv2.medianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 7)
        edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 3)
        return cv2.bitwise_and(color, color, mask=edges)

    @staticmethod
    def vignette(img: np.ndarray, strength: float = 0.6) -> np.ndarray:
        h, w = img.shape[:2]
        mask = cv2.getGaussianKernel(h, h * 0.5) @ cv2.getGaussianKernel(w, w * 0.5).T
        mask = (1 - strength) + strength * mask / mask.max()
        return np.clip(img * mask[:, :, None], 0, 255).astype(np.uint8)

    # ----- private ------------------------------------------------------------
    @staticmethod
    def _luma_stats(img: np.ndarray) -> tuple[float, float]:
        L = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[:, :, 0]
        return float(L.mean()), float(L.std())


if __name__ == "__main__":
    from core.cli import run_cli
    run_cli(SmartPhotoEditor)
