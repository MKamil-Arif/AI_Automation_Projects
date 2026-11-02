"""
Topic 7 - Fake Currency Detection (compare a photo of a note with a genuine reference scan).

  1. Align: ORB keypoints + ratio test + RANSAC homography warp the photo onto the reference
     (any rotation, scale, perspective). Uneven lighting is divided out.
  2. Checks
       Colour        - mean colour difference (Delta E in Lab a,b) against the genuine note
       Print sharpness - band-pass (DoG) energy of the guilloche area and the microtext band vs.
                       the reference: scanners / printers blur the 1-pixel lines of real printing.
                       (A blurry photo of a genuine note also fails this -> retake the photo.)
       Security thread - a dark (dashed) vertical line must exist in its column
       Watermark window - a plain bright area must exist in its place
  3. Verdict: thread or watermark missing -> FAKE, 2+ other failures -> FAKE, 1 -> SUSPICIOUS

The sample note in samples/ is a FICTIONAL design ("DIP STUDIO DEMO NOTE - SPECIMEN"). For a real
note, use a genuine scan of the same denomination as reference and the "Generic" template (or add
a NoteTemplate with that note's feature positions).
Run alone:  python -m topics.t07_fake_currency samples/note_photo_fake_print.jpg
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from core import utils
from core.base import DIPModule, ModuleResult, Param
from core.registry import register

DEFAULT_REFERENCE = Path(__file__).resolve().parent.parent / "samples" / "note_reference_genuine.png"
Box = tuple[float, float, float, float]          # x0, y0, x1, y1 as fractions of width / height


@dataclass(frozen=True)
class NoteTemplate:
    """Where a note's security features are (fractions of the reference note's size)."""
    thread: Box | None = None
    watermark: Box | None = None
    microtext: Box | None = None
    fine_print: Box | None = None


TEMPLATES = {
    "Demo note (samples)": NoteTemplate(thread=(0.60, 0.02, 0.64, 0.98), watermark=(0.74, 0.2, 0.9, 0.8),
                                        microtext=(0.05, 0.83, 0.58, 0.91), fine_print=(0.1, 0.17, 0.41, 0.83)),
    "Generic (colour + sharpness only)": NoteTemplate(),
}


@register
class FakeCurrencyDetector(DIPModule):
    topic_no = 7
    name = "Fake Currency Detection"
    description = ("Aligns a photo of a bank note with a genuine reference (ORB + homography), then checks "
                   "colour, fine-print sharpness, microtext, security thread and watermark window.")
    samples = ("note_photo_genuine.jpg", "note_photo_fake_print.jpg", "note_photo_fake_no_thread.jpg",
               "note_photo_fake_colour.jpg")

    # pass limits (tuned on the sample note: genuine photos vs. three kinds of fakes)
    MAX_COLOUR_DIFF = 6.0
    MIN_SHARPNESS = 0.55        # genuine photos ~65 %, good copies ~45 %, prints ~15 %
    MIN_THREAD_DEPTH = 35
    MIN_WINDOW_BRIGHTNESS = 15

    def parameters(self) -> list[Param]:
        return [
            Param("reference", "Genuine reference note (scan)", "image", None,
                  help="Empty = the demo note in samples/."),
            Param("template", "Security feature positions", "select", next(iter(TEMPLATES)),
                  options=tuple(TEMPLATES)),
        ]

    # ----- pipeline -----------------------------------------------------------
    def process(self, image, *, reference, template) -> ModuleResult:
        ref = utils.ensure_bgr(reference) if reference is not None else utils.read_image(DEFAULT_REFERENCE)
        ref, _ = utils.resize_max(ref, 1000)
        photo, _ = utils.resize_max(utils.ensure_bgr(image), 1400)
        self._add_step("Reference (genuine)", ref)

        aligned, inliers = self.align(photo, ref)
        aligned = self.match_lighting(aligned, ref)
        self._add_step(f"2. Photo warped onto the reference ({inliers} matching points)", aligned)
        diff = cv2.absdiff(cv2.GaussianBlur(aligned, (5, 5), 0), cv2.GaussianBlur(ref, (5, 5), 0))
        self._add_step("3. Difference to reference (x3)", np.clip(diff.astype(np.int32) * 3, 0, 255).astype(np.uint8))

        tpl = TEMPLATES[template]
        checks = self.run_checks(aligned, ref, tpl)
        verdict = self.verdict(checks)
        output = self._annotate(aligned, tpl, checks, verdict)
        return ModuleResult(output, info={
            "Verdict": verdict,
            "Checks failed": f"{sum(not c['pass'] for c in checks)} / {len(checks)}",
            "SSIM vs reference": f"{utils.ssim(aligned, ref):.2f}",
            "Checks": [{"Check": c["name"], "Value": c["value"], "Needed": c["needed"],
                        "Result": "PASS" if c["pass"] else "FAIL"} for c in checks],
        })

    # ----- alignment --------------------------------------------------------------
    def align(self, photo: np.ndarray, ref: np.ndarray) -> tuple[np.ndarray, int]:
        orb = cv2.ORB_create(5000)
        kp1, des1 = orb.detectAndCompute(cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY), None)
        kp2, des2 = orb.detectAndCompute(cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY), None)
        if des1 is None or des2 is None:
            raise ValueError("No features found - is this a photo of a bank note?")
        good = [m for m, n in (p for p in cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(des1, des2, k=2) if len(p) == 2)
                if m.distance < 0.75 * n.distance]
        if len(good) < 12:
            raise ValueError("Could not line the photo up with the reference note (different note type?).")
        src = np.float32([kp1[m.queryIdx].pt for m in good])
        dst = np.float32([kp2[m.trainIdx].pt for m in good])
        H, inl = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
        if H is None or inl.sum() < 12:
            raise ValueError("Could not line the photo up with the reference note (different note type?).")
        inlier_matches = [g for g, ok in zip(good, inl.ravel()) if ok][:60]
        self._add_step("1. ORB matches used for alignment",
                       cv2.drawMatches(photo, kp1, ref, kp2, inlier_matches, None, matchColor=(0, 255, 0),
                                       flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS))
        return cv2.warpPerspective(photo, H, ref.shape[1::-1]), int(inl.sum())

    @staticmethod
    def match_lighting(img: np.ndarray, ref: np.ndarray) -> np.ndarray:
        """Divide out slow brightness changes (uneven light) using the reference as target.
        The same factor is applied to B, G and R, so colour errors of a fake are kept."""
        g_img = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32), (0, 0), 25)
        g_ref = cv2.GaussianBlur(cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY).astype(np.float32), (0, 0), 25)
        gain = (g_ref + 1) / (g_img + 1)
        return np.clip(img.astype(np.float32) * gain[:, :, None], 0, 255).astype(np.uint8)

    # ----- checks -------------------------------------------------------------------
    @staticmethod
    def _crop(img: np.ndarray, box: Box | None) -> np.ndarray:
        if box is None:
            return img
        h, w = img.shape[:2]
        return img[int(box[1] * h):int(box[3] * h), int(box[0] * w):int(box[2] * w)]

    @staticmethod
    def _fine_detail(img: np.ndarray) -> float:
        """Energy of 1-2 pixel wide detail: a band-pass (Difference of Gaussians, sigma 1 and 2).
        Less sensitive to slight camera blur and noise than the Laplacian, but a printer's blur kills it."""
        g = utils.ensure_gray(img).astype(np.float32)
        return float((cv2.GaussianBlur(g, (0, 0), 1) - cv2.GaussianBlur(g, (0, 0), 2)).var())

    def run_checks(self, aligned: np.ndarray, ref: np.ndarray, tpl: NoteTemplate) -> list[dict]:
        checks = []
        # colour: compare blurred a,b channels so tiny misalignments do not count
        lab_a = cv2.cvtColor(cv2.GaussianBlur(aligned, (9, 9), 0), cv2.COLOR_BGR2LAB).astype(np.float32)
        lab_r = cv2.cvtColor(cv2.GaussianBlur(ref, (9, 9), 0), cv2.COLOR_BGR2LAB).astype(np.float32)
        delta_e = float(np.mean(np.linalg.norm(lab_a[:, :, 1:] - lab_r[:, :, 1:], axis=2)))
        checks.append({"name": "Colour", "value": f"Delta E {delta_e:.1f}", "needed": f"< {self.MAX_COLOUR_DIFF}",
                       "pass": delta_e < self.MAX_COLOUR_DIFF, "box": None})

        # fine print + microtext measure the same thing (print quality) -> one check, two areas
        ratios = []
        for name, box in (("Fine print", tpl.fine_print), ("Microtext", tpl.microtext)):
            if name == "Microtext" and box is None:
                continue
            ratios.append(self._fine_detail(self._crop(aligned, box)) / (self._fine_detail(self._crop(ref, box)) + 1e-9))
            if box is not None:
                self._add_step(f"{name}: photo (left) vs genuine (right)",
                               np.hstack([self._crop(aligned, box), self._crop(ref, box)]))
        checks.append({"name": "Print sharpness", "value": " / ".join(f"{r:.0%}" for r in ratios) + " of genuine",
                       "needed": f">= {self.MIN_SHARPNESS:.0%}", "pass": min(ratios) >= self.MIN_SHARPNESS,
                       "box": tpl.fine_print})

        if tpl.thread is not None:
            strip = cv2.cvtColor(self._crop(aligned, tpl.thread), cv2.COLOR_BGR2GRAY).astype(np.float32)
            profile = strip.mean(axis=0)
            col = int(np.argmin(profile))
            dark_rows = strip[:, max(col - 1, 0):col + 2].min(axis=1) < np.median(strip) - self.MIN_THREAD_DEPTH
            depth = float(np.median(strip) - strip[:, max(col - 1, 0):col + 2].min(axis=1)[dark_rows].mean()) \
                if dark_rows.any() else 0.0
            present = dark_rows.mean() > 0.4
            checks.append({"name": "Security thread",
                           "value": f"dark in {dark_rows.mean():.0%} of rows (depth {depth:.0f})",
                           "needed": "> 40% of rows", "pass": bool(present), "box": tpl.thread, "critical": True})

        if tpl.watermark is not None:
            gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY).astype(np.float32)
            lift = float(self._crop(gray, tpl.watermark).mean() - gray.mean())
            checks.append({"name": "Watermark window", "value": f"+{lift:.0f} brighter",
                           "needed": f">= +{self.MIN_WINDOW_BRIGHTNESS}", "pass": lift >= self.MIN_WINDOW_BRIGHTNESS,
                           "box": tpl.watermark, "critical": True})
        return checks

    @staticmethod
    def verdict(checks: list[dict]) -> str:
        if any(c.get("critical") and not c["pass"] for c in checks):
            return "FAKE"
        failed = sum(not c["pass"] for c in checks)
        return "FAKE" if failed >= 2 else "SUSPICIOUS" if failed == 1 else "GENUINE"

    # ----- drawing ------------------------------------------------------------------
    @staticmethod
    def _annotate(aligned, tpl, checks, verdict) -> np.ndarray:
        out = aligned.copy()
        h, w = out.shape[:2]
        for c in checks:
            if c["box"] is None:
                continue
            x0, y0, x1, y1 = c["box"]
            colour = (0, 200, 0) if c["pass"] else (0, 0, 255)
            cv2.rectangle(out, (int(x0 * w), int(y0 * h)), (int(x1 * w), int(y1 * h)), colour, 2)
            utils.put_label(out, c["name"], (int(x0 * w) + 3, int(y0 * h) + 16), colour, 0.45)
        colour = {"GENUINE": (80, 220, 80), "SUSPICIOUS": (0, 200, 255), "FAKE": (60, 60, 255)}[verdict]
        utils.put_label(out, verdict, (w - 150, h - 14), colour, 0.9)
        return out


if __name__ == "__main__":
    from core.cli import run_cli
    run_cli(FakeCurrencyDetector)
