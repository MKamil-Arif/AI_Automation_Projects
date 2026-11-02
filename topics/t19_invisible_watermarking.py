"""
Topic 19 - Invisible Watermarking System (DWT + QIM, blind extraction).

Embed
  1. BGR -> YCrCb, work on the luminance (Y) channel
  2. 1-level Haar DWT of Y -> LL, LH, HL, HH sub-bands
  3. Watermark (text rendered to a small binary image, or a logo) is copied many times over
     the chosen sub-band; every copy is scrambled and XOR-ed with patterns from the secret key
  4. Each coefficient is quantized to an "even" or "odd" grid point (Quantization Index
     Modulation) to store one bit
  5. Inverse DWT -> watermarked image (looks identical: PSNR ~37 dB at the default strength 36;
     on 20 real photos 36 survives JPEG-50 in 95 % of cases, strength 20 (PSNR 42 dB) in 0 %)
Extract (blind - original image NOT needed)
  Read every coefficient's grid (even/odd), undo XOR + scrambling of each copy, majority-vote.

Known trade-off (good viva point): LL survives JPEG / blur / resize but NOT a global
brightness shift (it moves every LL coefficient off its grid); HL / LH survive brightness
changes but not JPEG / blur, because those remove high frequencies.

Run alone:  python -m topics.t19_invisible_watermarking <image> mode="Robustness Test"
"""
from __future__ import annotations

import cv2
import numpy as np
import pywt

from core import utils
from core.base import DIPModule, ModuleResult, Param
from core.registry import register


@register
class InvisibleWatermark(DIPModule):
    topic_no = 19
    name = "Invisible Watermarking (DWT)"
    description = ("Hides a text or logo watermark inside the wavelet (DWT) coefficients of an "
                   "image and extracts it later without the original - for copyright protection.")

    TEXT_SHAPE = (32, 128)          # rows x cols of the binary text watermark
    LOGO_SHAPE = (64, 64)
    WAVELET = "haar"

    def parameters(self) -> list[Param]:
        embedding = ("Embed", "Robustness Test")
        return [
            Param("mode", "Mode", "select", "Embed", options=("Embed", "Extract", "Robustness Test"),
                  help="Extract expects an image that was watermarked here (saved as PNG)."),
            Param("wm_type", "Watermark type", "select", "Text", options=("Text", "Logo")),
            Param("text", "Watermark text", "text", "(c) DIP 2026",
                  visible_when={"wm_type": "Text", "mode": embedding}),
            Param("logo", "Logo image", "image", None,
                  visible_when={"wm_type": "Logo", "mode": embedding}),
            Param("strength", "Strength (quantization step)", "slider", 36, 8, 64, 2,
                  help="Higher = survives more attacks but is less invisible. Use the same value to extract."),
            Param("band", "DWT sub-band", "select", "LL", options=("LL", "HL", "LH"),
                  help="LL = most robust, HL/LH = most invisible. Use the same band to extract."),
            Param("key", "Secret key", "number", 1234, 0, 999999, 1,
                  help="Scrambles the watermark. Without the right key it cannot be read."),
        ]

    # ----- pipeline -----------------------------------------------------------
    def process(self, image, *, mode, wm_type, text, logo, strength, band, key) -> ModuleResult:
        img = utils.ensure_bgr(image)
        shape = self.TEXT_SHAPE if wm_type == "Text" else self.LOGO_SHAPE

        if mode == "Extract":
            bits, confidence = self.extract(img, shape, strength, band, key)
            self._add_step("Watermarked image (input)", img)
            out = utils.enlarge(bits * 255, 4)
            self._add_step("Extracted watermark", out)
            return ModuleResult(out, info={
                "Confidence": f"{confidence:.0%}",
                "Verdict": "Watermark found" if confidence > 0.5 else "No / damaged watermark",
            })

        if wm_type == "Text":
            bits = self.text_to_bits(text)
        else:
            if logo is None:
                raise ValueError("Please upload a logo image for the watermark.")
            bits = self.logo_to_bits(logo)

        marked, tiles = self.embed(img, bits, strength, band, key)
        diff = cv2.absdiff(img, marked)
        self._add_step("Original", img)
        self._add_step("Watermark bits", utils.enlarge(bits * 255, 4))
        self._add_step("Difference x20 (what was changed)", np.clip(diff.astype(np.int32) * 20, 0, 255).astype(np.uint8))
        self._add_step("Watermarked image", marked)

        info = {
            "PSNR": f"{utils.psnr(img, marked):.1f} dB",
            "SSIM": f"{utils.ssim(img, marked):.4f}",
            "Copies embedded": tiles,
        }
        if mode == "Robustness Test":
            info["Attack results"] = self._robustness_table(marked, bits, strength, band, key)
        else:
            check, _ = self.extract(marked, bits.shape, strength, band, key)
            self._add_step("Extracted back (check)", utils.enlarge(check * 255, 4))
            info["Bit error rate"] = f"{self.bit_error_rate(bits, check):.2%}"
        return ModuleResult(marked, info=info)

    # ----- watermark preparation ----------------------------------------------
    def text_to_bits(self, text: str) -> np.ndarray:
        """Render text into a small black/white image -> 0/1 array."""
        if not text.strip():
            raise ValueError("Watermark text is empty.")
        h, w = self.TEXT_SHAPE
        canvas = np.zeros((h, w), np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), _ = cv2.getTextSize(text, font, 1.0, 1)
        scale = min((w - 6) / tw, (h - 8) / th)
        (tw, th), _ = cv2.getTextSize(text, font, scale, 1)
        cv2.putText(canvas, text, ((w - tw) // 2, (h + th) // 2), font, scale, 255, 1, cv2.LINE_AA)
        return (canvas > 100).astype(np.uint8)

    def logo_to_bits(self, logo: np.ndarray) -> np.ndarray:
        gray = cv2.resize(utils.ensure_gray(logo), self.LOGO_SHAPE[::-1], interpolation=cv2.INTER_AREA)
        _, bw = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return bw.astype(np.uint8)

    # ----- embed / extract ----------------------------------------------------
    def embed(self, img: np.ndarray, bits: np.ndarray, strength: float, band: str, key: int):
        """Returns (watermarked image, number of watermark copies stored)."""
        ycc = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        y = ycc[:, :, 0].astype(np.float64)
        coeffs, region = self._decompose(y)
        c = coeffs[band]
        reps = self._repetitions(c.shape, bits.shape)
        h, w = bits.shape

        # every copy (tile) is scrambled + XOR-ed with its own key-derived pattern
        tile_keys = iter(self._tile_keys(bits.size, reps[0] * reps[1], key))
        tiled = np.zeros((h * reps[0], w * reps[1]), np.uint8)
        for ty in range(reps[0]):
            for tx in range(reps[1]):
                perm, noise = next(tile_keys)
                tiled[ty * h:(ty + 1) * h, tx * w:(tx + 1) * w] = (bits.ravel()[perm] ^ noise).reshape(h, w)
        th, tw = tiled.shape
        c[:th, :tw] = self._qim_embed(c[:th, :tw], tiled, strength)

        y[region] = pywt.idwt2((coeffs["LL"], (coeffs["LH"], coeffs["HL"], coeffs["HH"])),
                               self.WAVELET, mode="periodization")
        ycc[:, :, 0] = np.clip(np.round(y), 0, 255).astype(np.uint8)
        return cv2.cvtColor(ycc, cv2.COLOR_YCrCb2BGR), reps[0] * reps[1]

    def extract(self, img: np.ndarray, shape: tuple[int, int], strength: float, band: str, key: int):
        """Blind extraction. Returns (bits, confidence 0..1)."""
        y = cv2.cvtColor(utils.ensure_bgr(img), cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(np.float64)
        coeffs, _ = self._decompose(y)
        c = coeffs[band]
        reps = self._repetitions(c.shape, shape)
        h, w = shape
        raw = self._qim_extract(c[:h * reps[0], :w * reps[1]], strength)

        # undo XOR + scrambling of every copy, then majority-vote
        tile_keys = iter(self._tile_keys(h * w, reps[0] * reps[1], key))
        votes = np.zeros(h * w)
        for ty in range(reps[0]):
            for tx in range(reps[1]):
                perm, noise = next(tile_keys)
                votes[perm] += raw[ty * h:(ty + 1) * h, tx * w:(tx + 1) * w].ravel() ^ noise
        votes /= reps[0] * reps[1]

        bits = (votes > 0.5).astype(np.uint8).reshape(shape)
        confidence = float(np.mean(np.abs(votes - 0.5) * 2))   # copies agree -> 1, random -> ~0
        return bits, confidence

    # ----- robustness ---------------------------------------------------------
    ATTACKS = {
        "No attack": lambda im: im,
        "JPEG quality 75": lambda im: InvisibleWatermark._jpeg(im, 75),
        "JPEG quality 50": lambda im: InvisibleWatermark._jpeg(im, 50),
        "Gaussian noise (sigma 5)": lambda im: np.clip(
            im + np.random.default_rng(0).normal(0, 5, im.shape), 0, 255).astype(np.uint8),
        "Brightness +20": lambda im: cv2.add(im, np.full_like(im, 20)),
        "Gaussian blur 3x3": lambda im: cv2.GaussianBlur(im, (3, 3), 0),
        "Median filter 3x3": lambda im: cv2.medianBlur(im, 3),
        "Resize 50% and back": lambda im: cv2.resize(
            cv2.resize(im, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA),
            im.shape[1::-1], interpolation=cv2.INTER_LINEAR),
    }

    def _robustness_table(self, marked, bits, strength, band, key) -> list[dict]:
        rows = []
        for name, attack in self.ATTACKS.items():
            attacked = attack(marked)
            got, conf = self.extract(attacked, bits.shape, strength, band, key)
            nc, ber = self.normalized_correlation(bits, got), self.bit_error_rate(bits, got)
            self._add_step(f"{name}: NC={nc:.2f}", utils.enlarge(got * 255, 4))
            psnr = utils.psnr(marked, attacked)
            rows.append({"Attack": name, "NC": round(nc, 3), "BER": f"{ber:.1%}",
                         "PSNR vs marked": "∞" if np.isinf(psnr) else f"{psnr:.1f} dB",
                         "Survived": "Yes" if nc >= 0.85 else "No"})
        return rows

    @staticmethod
    def _jpeg(img: np.ndarray, quality: int) -> np.ndarray:
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)

    # ----- metrics ------------------------------------------------------------
    @staticmethod
    def normalized_correlation(a: np.ndarray, b: np.ndarray) -> float:
        """NC on bipolar (+1/-1) bits: 1.0 = identical, ~0 = random."""
        x, y = a.astype(np.float64) * 2 - 1, b.astype(np.float64) * 2 - 1
        return float(np.sum(x * y) / np.sqrt(np.sum(x * x) * np.sum(y * y)))

    @staticmethod
    def bit_error_rate(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.mean(a != b))

    # ----- private helpers ----------------------------------------------------
    def _decompose(self, y: np.ndarray):
        """DWT of the even-sized part of Y (odd last row/col is left untouched)."""
        h, w = (y.shape[0] // 2) * 2, (y.shape[1] // 2) * 2
        region = (slice(0, h), slice(0, w))
        LL, (LH, HL, HH) = pywt.dwt2(y[region], self.WAVELET, mode="periodization")
        return {"LL": LL, "LH": LH, "HL": HL, "HH": HH}, region

    @staticmethod
    def _repetitions(band_shape, wm_shape) -> tuple[int, int]:
        reps = (band_shape[0] // wm_shape[0], band_shape[1] // wm_shape[1])
        if min(reps) < 1:
            need = (wm_shape[0] * 2, wm_shape[1] * 2)
            raise ValueError(f"Image too small: needs at least {need[1]}x{need[0]} pixels.")
        return reps

    @staticmethod
    def _tile_keys(n: int, count: int, key: int) -> list[tuple[np.ndarray, np.ndarray]]:
        """Per copy: a permutation (scrambles positions) + random bits (balances 0s and 1s,
        so a wrong key / wrong band gives ~50% random votes = low confidence)."""
        rng = np.random.default_rng(int(key))
        return [(rng.permutation(n), rng.integers(0, 2, n, dtype=np.uint8)) for _ in range(count)]

    @staticmethod
    def _qim_embed(c: np.ndarray, bits: np.ndarray, delta: float) -> np.ndarray:
        """bit 0 -> nearest multiple of delta, bit 1 -> nearest multiple + delta/2."""
        offset = bits * (delta / 2)
        return np.round((c - offset) / delta) * delta + offset

    @staticmethod
    def _qim_extract(c: np.ndarray, delta: float) -> np.ndarray:
        d0 = np.abs(c - np.round(c / delta) * delta)
        d1 = np.abs(c - (np.round((c - delta / 2) / delta) * delta + delta / 2))
        return (d1 < d0).astype(np.uint8)


if __name__ == "__main__":
    from core.cli import run_cli
    run_cli(InvisibleWatermark)
