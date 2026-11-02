"""
Topic 17 - Background Remover & Changer (like remove.bg).

Segmentation (foreground mask)
  U2-Net (deep, default)    : U2-Net-p salient-object CNN (Qin et al. 2020, 4.7 MB ONNX, offline, run
                              with cv2.dnn) gives a "main subject" probability map -> threshold 0.5.
                              DUTS-TE (100 photos): mean IoU 0.76, GrabCut only 0.38 - 0.44.
  GrabCut (rectangle)       : everything outside a rectangle is background; GrabCut models
                              foreground/background colours with Gaussian mixtures and cuts the
                              graph between them (min-cut). The rectangle reaches the bottom edge
                              because portraits usually touch it.
  GrabCut (saliency seeds)  : no rectangle - a frequency-tuned saliency map (Achanta et al. 2009,
                              colour distance from the image's mean colour x centre prior) marks
                              likely foreground; the top/left/right border is sure background.
  Chroma key (green screen) : HSV threshold on green + "despill" (removes green fringes)
Refinement
  Only the main subject is kept, small holes filled, then a guided filter (He et al. 2010) in a
  thin band around the edge snaps the mask to the real image edges -> a smooth alpha matte
  instead of a jagged cut-out.
Background
  Transparent PNG, white, any colour, portrait-mode blur, or a custom image.

Run alone:  python -m topics.t17_background_remover samples/portrait_room_synthetic.png background=White
"""
from __future__ import annotations

import cv2
import numpy as np

from core import utils
from core.base import DIPModule, ModuleResult, Param
from core.recognition import MODELS_DIR
from core.registry import register

U2NET = MODELS_DIR / "u2netp.onnx"
U2NET_URL = "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2netp.onnx"
DEEP_METHOD = "U2-Net (deep, offline)"

GRABCUT_METHODS = ("GrabCut (rectangle)", "GrabCut (saliency seeds)")


def guided_filter(guide: np.ndarray, src: np.ndarray, radius: int, eps: float) -> np.ndarray:
    """Edge-preserving smoothing of `src` that follows the edges of `guide` (both float 0..1)."""
    box = lambda x: cv2.boxFilter(x, -1, (2 * radius + 1, 2 * radius + 1))
    mean_i, mean_p = box(guide), box(src)
    var_i = box(guide * guide) - mean_i * mean_i
    cov_ip = box(guide * src) - mean_i * mean_p
    a = cov_ip / (var_i + eps)                 # local linear model: q = a * I + b
    b = mean_p - a * mean_i
    return box(a) * guide + box(b)


@register
class BackgroundRemover(DIPModule):
    topic_no = 17
    name = "Background Remover & Changer"
    description = ("Separates the subject from the background (U2-Net CNN, GrabCut, saliency seeds or "
                   "green-screen chroma key), smooths the edge with a guided filter and puts in a new background.")
    samples = ("portrait_room_synthetic.png", "portrait_greenscreen_synthetic.png")

    WORK_SIZE = 1200
    GRABCUT_SIZE = 400          # GrabCut is slow, so it runs on a small copy
    _net = None

    def parameters(self) -> list[Param]:
        return [
            Param("method", "Segmentation", "select", DEEP_METHOD,
                  options=(DEEP_METHOD, *GRABCUT_METHODS, "Chroma key (green screen)")),
            Param("margin", "Rectangle margin (% of width/height)", "slider", 5, 1, 30, 1,
                  visible_when={"method": GRABCUT_METHODS[0]},
                  help="The subject must be fully inside the rectangle (it may touch the bottom edge)."),
            Param("iterations", "GrabCut iterations", "slider", 5, 1, 10, 1,
                  visible_when={"method": GRABCUT_METHODS}),
            Param("refine", "Smooth edges (guided filter)", "checkbox", True),
            Param("background", "New background", "select", "Transparent",
                  options=("Transparent", "White", "Solid colour", "Blur (portrait mode)", "Custom image")),
            Param("colour", "Background colour", "color", "#1E90FF",
                  visible_when={"background": "Solid colour"}),
            Param("blur", "Blur strength", "slider", 15, 3, 40, 1,
                  visible_when={"background": "Blur (portrait mode)"}),
            Param("bg_image", "Background image", "image", None,
                  visible_when={"background": "Custom image"}),
        ]

    # ----- pipeline -----------------------------------------------------------
    def process(self, image, *, method, margin, iterations, refine, background, colour, blur,
                bg_image) -> ModuleResult:
        img, _ = utils.resize_max(utils.ensure_bgr(image), self.WORK_SIZE)
        self._add_step("Original", img)

        note = None
        if method == DEEP_METHOD and not U2NET.exists():
            method, note = GRABCUT_METHODS[1], f"models/{U2NET.name} not found ({U2NET_URL}) - used GrabCut"
        if method.startswith("Chroma"):
            mask = self.chroma_key(img)
        elif method == DEEP_METHOD:
            mask = self.u2net(img)
        else:
            mask = self.grabcut(img, method, margin, iterations)
        mask = self.clean(mask)
        self._add_step("3. Cleaned mask (small blobs removed, holes filled)", mask)

        alpha = self.refine_alpha(img, mask) if refine else mask.astype(np.float32) / 255
        self._add_step("4. Alpha matte" + (" (guided filter)" if refine else ""), (alpha * 255).astype(np.uint8))

        fg = self.despill(img, alpha) if method.startswith("Chroma") else img
        output = self.composite(fg, alpha, background, colour, blur, bg_image)
        info = {
            "Foreground": f"{alpha.mean():.1%}",
            "Background": background,
            "Edges": "guided filter" if refine else "hard cut",
            "Size": f"{img.shape[1]} x {img.shape[0]}",
        }
        if note:
            info["Note"] = note
        return ModuleResult(output, info=info)

    # ----- segmentation -------------------------------------------------------
    def u2net(self, img: np.ndarray) -> np.ndarray:
        """U2-Net-p on a 320 x 320 copy (ImageNet normalisation, like its training) -> mask at full size."""
        if BackgroundRemover._net is None:
            BackgroundRemover._net = cv2.dnn.readNetFromONNX(str(U2NET))
        rgb = cv2.cvtColor(cv2.resize(img, (320, 320), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        x = rgb.astype(np.float32) / max(float(rgb.max()), 1.0)
        x = (x - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)
        BackgroundRemover._net.setInput(x.transpose(2, 0, 1)[None].astype(np.float32))
        sal = BackgroundRemover._net.forward()[0, 0]
        sal = (sal - sal.min()) / (np.ptp(sal) + 1e-9)
        sal = cv2.resize(sal, img.shape[1::-1], interpolation=cv2.INTER_LINEAR)
        self._add_step("1. U2-Net: probability of 'main subject'", (sal * 255).astype(np.uint8))
        return (sal > 0.5).astype(np.uint8) * 255

    def grabcut(self, img: np.ndarray, method: str, margin: int, iterations: int) -> np.ndarray:
        cv2.setRNGSeed(0)                                          # GrabCut's k-means init is random
        small, _ = utils.resize_max(img, self.GRABCUT_SIZE)
        h, w = small.shape[:2]
        mask = np.zeros((h, w), np.uint8)
        bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)

        if method == GRABCUT_METHODS[0]:
            mx, my = int(w * margin / 100), int(h * margin / 100)
            rect = (mx, my, w - 2 * mx, h - my)                    # reaches the bottom edge
            seeds = small.copy()
            cv2.rectangle(seeds, rect[:2], (rect[0] + rect[2] - 1, h - 1), (0, 255, 0), 2)
            self._add_step("1. GrabCut rectangle (outside = background)", seeds)
            try:
                cv2.grabCut(small, mask, rect, bgd, fgd, iterations, cv2.GC_INIT_WITH_RECT)
            except cv2.error:
                raise ValueError("Could not tell subject and background apart (the image is almost one colour).") from None
        else:
            saliency = self.saliency(small)
            self._add_step("1a. Saliency map (colour contrast x centre prior)", (saliency * 255).astype(np.uint8))
            mask[:] = cv2.GC_PR_BGD
            otsu, _ = cv2.threshold((saliency * 255).astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            mask[saliency * 255 > otsu] = cv2.GC_PR_FGD
            mask[saliency > np.percentile(saliency, 97)] = cv2.GC_FGD
            b = max(2, int(0.03 * min(h, w)))                      # sure background frame
            mask[:b, :], mask[:, :b], mask[:, -b:] = cv2.GC_BGD, cv2.GC_BGD, cv2.GC_BGD
            if not np.isin(mask, (cv2.GC_FGD, cv2.GC_PR_FGD)).any():  # flat image: nothing salient at all
                cv2.ellipse(mask, (w // 2, int(0.6 * h)), (w // 4, h // 3), 0, 0, 360, int(cv2.GC_PR_FGD), -1)
            seeds = small.copy()
            seeds[mask == cv2.GC_BGD] = (0, 0, 200)
            seeds[mask == cv2.GC_PR_FGD] = (0.5 * seeds[mask == cv2.GC_PR_FGD] + (0, 100, 0)).astype(np.uint8)
            seeds[mask == cv2.GC_FGD] = (0, 255, 0)
            self._add_step("1b. Seeds (red = sure bg, green = sure fg, tint = probable fg)", seeds)
            try:
                cv2.grabCut(small, mask, None, bgd, fgd, iterations, cv2.GC_INIT_WITH_MASK)
            except cv2.error:                                      # e.g. a one-colour image: no two colour models
                raise ValueError("Could not tell subject and background apart (the image is almost one colour).") from None

        fg = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
        self._add_step("2. GrabCut result (low resolution)", fg)
        return cv2.resize(fg, img.shape[1::-1], interpolation=cv2.INTER_LINEAR)

    @staticmethod
    def saliency(img: np.ndarray) -> np.ndarray:
        """Frequency-tuned saliency: how far each (blurred) pixel's Lab colour is from the mean colour."""
        lab = cv2.cvtColor(cv2.GaussianBlur(img, (5, 5), 0), cv2.COLOR_BGR2LAB).astype(np.float32)
        sal = np.linalg.norm(lab - lab.reshape(-1, 3).mean(axis=0), axis=2)
        h, w = sal.shape
        ys, xs = np.mgrid[0:h, 0:w]
        centre = np.exp(-(((xs - w / 2) / (0.35 * w)) ** 2 + ((ys - 0.6 * h) / (0.45 * h)) ** 2))
        sal = sal * centre
        return sal / (sal.max() + 1e-9)

    def chroma_key(self, img: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        green = cv2.inRange(hsv, (35, 60, 40), (85, 255, 255))
        self._add_step("1. Green pixels (HSV threshold)", green)
        fg = cv2.bitwise_not(green)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        self._add_step("2. Foreground = not green", fg)
        return fg

    @staticmethod
    def clean(mask: np.ndarray) -> np.ndarray:
        binary = (mask > 127).astype(np.uint8)
        # opening removes thin slivers (e.g. along a wall / floor edge) glued to the subject
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        if n <= 1:
            return binary * 255
        areas = stats[1:, cv2.CC_STAT_AREA]
        keep = np.zeros(n, bool)
        # keep the main subject (and a second one of similar size), drop look-alike objects
        keep[1:] = areas >= 0.5 * areas.max()
        binary = keep[labels].astype(np.uint8)
        holes = (1 - binary).astype(np.uint8)                       # fill small holes only
        hn, hlabels, hstats, _ = cv2.connectedComponentsWithStats(holes, connectivity=4)
        h, w = mask.shape
        for i in range(1, hn):
            x, y, bw, bh, area = hstats[i]
            if area < 0.01 * h * w and x > 0 and y > 0 and x + bw < w and y + bh < h:
                binary[hlabels == i] = 1
        return binary * 255

    def refine_alpha(self, img: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Guided filter in a thin band around the edge only.

        The guide is the image projected on the Lab colour direction that best separates the
        subject's mean colour from the background's (a gray guide fails when, e.g., a blue shirt
        and a brown floor have the same brightness).
        """
        hard = mask.astype(np.float32) / 255
        fg, bg = mask > 127, mask <= 127
        if not fg.any() or not bg.any():
            return hard
        radius = max(4, min(img.shape[:2]) // 80)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
        direction = lab[fg].mean(axis=0) - lab[bg].mean(axis=0)
        guide = lab @ (direction / (np.linalg.norm(direction) + 1e-6))
        guide = (guide - guide.min()) / (np.ptp(guide) + 1e-6)
        self._add_step("Guide image (colour direction subject vs background)", (guide * 255).astype(np.uint8))
        soft = np.clip(guided_filter(guide, hard, radius, 1e-3), 0, 1)
        kernel = np.ones((2 * radius + 1, 2 * radius + 1), np.uint8)
        band = cv2.dilate(mask, kernel) != cv2.erode(mask, kernel)
        return np.where(band, soft, hard)

    # ----- compositing --------------------------------------------------------
    @staticmethod
    def despill(img: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        """Green light reflected onto the subject: cap green at max(blue, red) near the edge."""
        out = img.copy()
        edge = (alpha > 0.02) & (alpha < 0.98)
        edge = cv2.dilate(edge.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        b, g, r = out[:, :, 0], out[:, :, 1], out[:, :, 2]
        g[edge] = np.minimum(g[edge], np.maximum(b[edge], r[edge]))
        return out

    def composite(self, fg, alpha, background, colour, blur, bg_image) -> np.ndarray:
        if background == "Transparent":
            return np.dstack([fg, (alpha * 255).astype(np.uint8)])          # BGRA
        h, w = fg.shape[:2]
        if background == "White":
            bg = np.full_like(fg, 255)
        elif background == "Solid colour":
            hex_ = colour.lstrip("#")
            r, g, b = (int(hex_[i:i + 2], 16) for i in (0, 2, 4))
            bg = np.full_like(fg, (b, g, r))
        elif background == "Blur (portrait mode)":
            bg = cv2.GaussianBlur(fg, (0, 0), blur)
        else:
            if bg_image is None:
                raise ValueError("Please upload a background image.")
            bg = self._cover(utils.ensure_bgr(bg_image), w, h)
        a = alpha[:, :, None]
        return (a * fg + (1 - a) * bg).astype(np.uint8)

    @staticmethod
    def _cover(img: np.ndarray, w: int, h: int) -> np.ndarray:
        """Scale and centre-crop so the image fills w x h (like CSS background-size: cover)."""
        s = max(w / img.shape[1], h / img.shape[0])
        big = cv2.resize(img, (int(np.ceil(img.shape[1] * s)), int(np.ceil(img.shape[0] * s))))
        y, x = (big.shape[0] - h) // 2, (big.shape[1] - w) // 2
        return big[y:y + h, x:x + w]

    # ----- evaluation helper (for the report) ---------------------------------
    @staticmethod
    def mask_iou(predicted: np.ndarray, ground_truth: np.ndarray) -> float:
        """Intersection over Union of two masks (alpha > 0.5 counts as foreground)."""
        if predicted.shape[:2] != ground_truth.shape[:2]:
            predicted = cv2.resize(predicted, ground_truth.shape[1::-1], interpolation=cv2.INTER_LINEAR)
        p, g = predicted > 127, ground_truth > 127
        return float(np.logical_and(p, g).sum() / max(np.logical_or(p, g).sum(), 1))


if __name__ == "__main__":
    from core.cli import run_cli
    run_cli(BackgroundRemover)
