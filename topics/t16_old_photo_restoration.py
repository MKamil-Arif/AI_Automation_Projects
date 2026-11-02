"""
Topic 16 - Old Photo Restoration (scratches, dust, noise, fading, colorization).

  1. Yellow / sepia tint removed (work on the gray / luminance channel), contrast stretched
  2. Damage mask: white top-hat finds thin BRIGHT things (scratches, dust), black top-hat finds
     thin DARK ones - both narrower than the structuring element. Optional hand-drawn mask added.
  3. Inpainting fills the masked pixels from their surroundings (Telea fast marching or
     Navier-Stokes fluid flow)
  4. Non-local means denoising removes film grain
  5. Fading fixed: percentile contrast stretch + CLAHE
  6. Colorization (classical, no deep learning):
       Welsh et al. 2002 "Transferring color to greyscale images" - every gray pixel takes the
       colour of the reference pixel with the most similar brightness + local texture (std)
     or a simple sepia tone.

Run alone:  python -m topics.t16_old_photo_restoration samples/old_photo_synthetic.png
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree

from core import utils
from core.base import DIPModule, ModuleResult, Param
from core.registry import register

DEFAULT_REFERENCE = Path(__file__).resolve().parent.parent / "samples" / "colour_reference_landscape.png"


@register
class OldPhotoRestorer(DIPModule):
    topic_no = 16
    name = "Old Photo Restoration"
    description = ("Removes scratches and dust (morphological detection + inpainting), film grain and "
                   "fading, then optionally colorizes the photo from a colour reference image.")
    samples = ("old_photo_synthetic.png",)

    WORK_SIZE = 1200

    def parameters(self) -> list[Param]:
        return [
            Param("black_white", "Black & white photo (remove yellow tint)", "checkbox", True),
            Param("defects", "Damage to remove", "select", "Bright scratches & dust",
                  options=("Bright scratches & dust", "Dark scratches", "Both", "None")),
            Param("sensitivity", "Damage threshold (lower = finds more)", "slider", 25, 5, 80, 1),
            Param("max_width", "Max scratch width (px)", "slider", 7, 3, 21, 2,
                  help="Structuring element size. Real details thinner than this may be removed too."),
            Param("user_mask", "Extra damage mask (optional, white = damaged)", "image", None),
            Param("inpaint", "Inpainting method", "select", "Telea", options=("Telea", "Navier-Stokes")),
            Param("denoise", "Grain removal strength (0 = off)", "slider", 10, 0, 30, 1),
            Param("colorize", "Colorize", "select", "None",
                  options=("None", "Reference image (Welsh)", "Sepia tone")),
            Param("reference", "Colour reference image (similar scene)", "image", None,
                  visible_when={"colorize": "Reference image (Welsh)"},
                  help="If empty, samples/colour_reference_landscape.png is used."),
        ]

    # ----- pipeline -----------------------------------------------------------
    def process(self, image, *, black_white, defects, sensitivity, max_width, user_mask, inpaint,
                denoise, colorize, reference) -> ModuleResult:
        img, _ = utils.resize_max(utils.ensure_bgr(image), self.WORK_SIZE)
        self._add_step("Original", img)

        if black_white:
            lum, chroma = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), None
            self._add_step("1. Gray (sepia tint removed)", lum)
        else:
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            lum, chroma = lab[:, :, 0], lab[:, :, 1:]
            self._add_step("1. Lightness channel (L of Lab)", lum)

        contrast_before = lum.std()
        # stretch first: fading squeezes the scratches' contrast below any fixed threshold
        lum = self.stretch(lum)
        self._add_step("2. Contrast stretched", lum)
        mask = self.damage_mask(lum, defects, sensitivity, max_width)
        if user_mask is not None:
            extra = cv2.resize(utils.ensure_gray(user_mask), lum.shape[::-1], interpolation=cv2.INTER_NEAREST)
            mask |= (extra > 127).astype(np.uint8) * 255
        self._add_step("3. Damage mask (top-hat)", mask)

        flag = cv2.INPAINT_TELEA if inpaint == "Telea" else cv2.INPAINT_NS
        restored = cv2.inpaint(lum, mask, 3, flag) if mask.any() else lum.copy()
        self._add_step(f"4. Inpainted ({inpaint})", restored)

        if denoise > 0:
            restored = cv2.fastNlMeansDenoising(restored, None, denoise, 7, 21)
            self._add_step("5. Non-local means denoising", restored)

        restored = self.fix_fading(restored)
        self._add_step("6. Local contrast (stretch + CLAHE)", restored)

        if chroma is not None:                                     # colour photo: put colour back
            # inpaint accepts 1 or 3 channels only, so a and b are filled one at a time
            a, b = (cv2.inpaint(np.ascontiguousarray(chroma[:, :, i]), mask, 3, flag) for i in (0, 1))
            output = cv2.cvtColor(np.dstack([restored, a, b]), cv2.COLOR_LAB2BGR)
        elif colorize == "Reference image (Welsh)":
            ref = reference if reference is not None else self._default_reference()
            output = self.colorize_welsh(restored, utils.ensure_bgr(ref))
            self._add_step("Colour reference", utils.resize_max(utils.ensure_bgr(ref), 600)[0])
            self._add_step("7. Colorized (Welsh colour transfer)", output)
        elif colorize == "Sepia tone":
            output = self.sepia(restored)
        else:
            output = cv2.cvtColor(restored, cv2.COLOR_GRAY2BGR)

        return ModuleResult(output, info={
            "Damaged pixels": f"{(mask > 0).mean():.2%}",
            "Damage spots": int(cv2.connectedComponents(mask)[0] - 1),
            "Contrast (std)": f"{contrast_before:.0f} -> {restored.std():.0f}",
            "Colour": "kept" if chroma is not None else colorize,
        })

    # ----- steps ----------------------------------------------------------------
    @staticmethod
    def damage_mask(lum: np.ndarray, defects: str, sensitivity: int, max_width: int) -> np.ndarray:
        """Top-hat transforms keep only structures thinner than the structuring element."""
        if defects == "None":
            return np.zeros_like(lum)
        smooth = cv2.GaussianBlur(lum, (3, 3), 0)                  # grain would look like dust
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max_width, max_width))
        response = np.zeros_like(lum)
        if defects in ("Bright scratches & dust", "Both"):
            response = cv2.max(response, cv2.morphologyEx(smooth, cv2.MORPH_TOPHAT, kernel))
        if defects in ("Dark scratches", "Both"):
            response = cv2.max(response, cv2.morphologyEx(smooth, cv2.MORPH_BLACKHAT, kernel))
        mask = (response > sensitivity).astype(np.uint8) * 255

        # Shape filter: keep scratches (long + thin) and dust (tiny specks). Compact blobs are
        # usually real details (a person, a stone, texture) that only happen to be small.
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        bw, bh, area = stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_HEIGHT], stats[:, cv2.CC_STAT_AREA]
        elongated = (np.maximum(bw, bh) >= 3 * np.minimum(bw, bh)) | (area < 0.3 * bw * bh)
        dust = (area >= 3) & (area <= 40)
        keep = (elongated & (np.maximum(bw, bh) >= 15)) | dust
        keep[0] = False                                            # label 0 = background
        mask = (keep[labels] * 255).astype(np.uint8)
        return cv2.dilate(mask, np.ones((3, 3), np.uint8))         # cover the soft scratch edges too

    @staticmethod
    def stretch(gray: np.ndarray) -> np.ndarray:
        """Percentile contrast stretch: 0.5% darkest -> 0, 0.5% brightest -> 255."""
        lo, hi = np.percentile(gray, (0.5, 99.5))
        return np.clip((gray.astype(np.float32) - lo) * 255 / max(hi - lo, 1), 0, 255).astype(np.uint8)

    def fix_fading(self, gray: np.ndarray) -> np.ndarray:
        return cv2.createCLAHE(1.5, (8, 8)).apply(self.stretch(gray))

    # ----- colorization ---------------------------------------------------------
    @staticmethod
    def _local_std(x: np.ndarray, k: int = 5) -> np.ndarray:
        mean = cv2.blur(x, (k, k))
        return np.sqrt(np.maximum(cv2.blur(x * x, (k, k)) - mean * mean, 0))

    def colorize_welsh(self, gray: np.ndarray, reference: np.ndarray, n_samples: int = 400) -> np.ndarray:
        ref, _ = utils.resize_max(reference, 600)
        ref_lab = cv2.cvtColor(ref, cv2.COLOR_BGR2LAB).astype(np.float32)
        target_l = cv2.cvtColor(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float32)

        # 1. luminance remapping: give the reference the same brightness statistics as the target
        ref_l = (ref_lab[:, :, 0] - ref_lab[:, :, 0].mean()) * (target_l.std() / (ref_lab[:, :, 0].std() + 1e-6)) \
            + target_l.mean()
        # 2. a few hundred reference "swatches" on a jittered grid
        rng = np.random.default_rng(0)
        side = int(np.sqrt(n_samples))
        h, w = ref_l.shape
        ys = np.clip((np.repeat(np.arange(side), side) + rng.random(side * side)) * h / side, 0, h - 1).astype(int)
        xs = np.clip((np.tile(np.arange(side), side) + rng.random(side * side)) * w / side, 0, w - 1).astype(int)
        ref_feat = np.column_stack([ref_l[ys, xs], self._local_std(ref_l)[ys, xs]])
        scale = ref_feat.std(axis=0) + 1e-6                        # equal weight to brightness and texture
        # 3. every target pixel takes the a, b colour of the most similar swatch
        target_feat = np.column_stack([target_l.ravel(), self._local_std(target_l).ravel()])
        _, idx = cKDTree(ref_feat / scale).query(target_feat / scale)
        ab = ref_lab[ys, xs, 1:][idx].reshape(*target_l.shape, 2)
        ab = cv2.medianBlur(ab.astype(np.uint8), 5)                # remove speckled colour noise
        ab = cv2.GaussianBlur(ab, (0, 0), 2)
        out = np.dstack([target_l.astype(np.uint8), ab])
        return cv2.cvtColor(out, cv2.COLOR_LAB2BGR)

    @staticmethod
    def sepia(gray: np.ndarray) -> np.ndarray:
        g = gray.astype(np.float32) / 255
        return (np.dstack([g * 0.72, g * 0.88, g]) * 255).clip(0, 255).astype(np.uint8)

    @staticmethod
    def _default_reference() -> np.ndarray:
        if not DEFAULT_REFERENCE.exists():
            raise ValueError("Upload a colour reference image (a colour photo of a similar scene).")
        return utils.read_image(DEFAULT_REFERENCE)


if __name__ == "__main__":
    from core.cli import run_cli
    run_cli(OldPhotoRestorer)
