"""
Topic 1 - Tree Detection & Counting (aerial / drone images of orchards and forests).

  1. Greenness map: Excess Green index ExG = 2g - r - b (tree shadows and soil are not green).
     Otsu splits greener / less green; trees are the SMALLER part (greener on soil, less green
     than grass for silvery olive trees)
  2. Tree mask: Otsu threshold, opening, hole filling
  3. Count:
       Watershed     - distance transform peaks = crown centres, watershed splits touching crowns
       LoG blobs     - Laplacian-of-Gaussian blob detector on the greenness map, each blob = one
                       crown (its sigma gives the crown radius: r = sigma * sqrt(2))
  4. Crown size statistics; with a ground resolution (metres per pixel) also in metres

Run alone:  python -m topics.t01_tree_counting samples/orchard_aerial_synthetic.png
"""
from __future__ import annotations

import cv2
import numpy as np
from skimage.feature import blob_log
from skimage.segmentation import watershed

from core import utils
from core.base import DIPModule, ModuleResult, Param
from core.registry import register


@register
class TreeCounter(DIPModule):
    topic_no = 1
    name = "Tree Detection & Counting"
    description = ("Counts tree crowns in aerial / drone images: Excess Green segmentation, then "
                   "watershed or Laplacian-of-Gaussian blobs to separate touching crowns.")
    samples = ("orchard_aerial_synthetic.png",)

    WORK_SIZE = 1200
    MIN_AREA = 0.75        # smallest crown kept, as a fraction of pi * (min_crown / 2)^2

    def parameters(self) -> list[Param]:
        return [
            Param("method", "Counting method", "select", "Watershed", options=("Watershed", "LoG blobs")),
            Param("min_crown", "Smallest crown diameter (px, 0 = auto)", "slider", 0, 0, 150, 2,
                  help="0 = measured automatically (granulometry). Smaller blobs are ignored; also sets how "
                       "close two tree centres may be."),
            Param("gsd", "Ground resolution (metres per pixel, 0 = unknown)", "slider", 0.0, 0.0, 1.0, 0.01),
        ]

    # ----- pipeline -----------------------------------------------------------
    def process(self, image, *, method, min_crown, gsd) -> ModuleResult:
        img, scale = utils.resize_max(utils.ensure_bgr(image), self.WORK_SIZE)
        self._add_step("Original", img)

        green, mask, min_crown = self.tree_mask(img, min_crown * scale)
        if method == "Watershed":
            crowns = self.count_watershed(mask, min_crown)
        else:
            crowns = self.count_log(green, mask, min_crown)

        diameters = np.array([2 * r for _, _, r in crowns]) if crowns else np.zeros(1)
        info = {
            "Trees": len(crowns),
            "Canopy cover": f"{(mask > 0).mean():.1%}",
            "Mean crown": f"{diameters.mean():.0f} px",
        }
        if gsd > 0:
            info["Mean crown (m)"] = f"{diameters.mean() * gsd / scale:.1f} m"
            info["Trees per hectare"] = round(len(crowns) / (mask.size * (gsd / scale) ** 2 / 10_000))
        return ModuleResult(self._annotate(img, crowns), info=info)

    # ----- steps ----------------------------------------------------------------
    def tree_mask(self, img: np.ndarray, min_crown: float) -> tuple[np.ndarray, np.ndarray, float]:
        """Returns (map where trees are bright, tree mask, smallest crown diameter in px)."""
        f = img.astype(np.float32)
        total = f.sum(axis=2) + 1e-6
        exg = (2 * f[:, :, 1] - f[:, :, 0] - f[:, :, 2]) / total          # chromatic ExG
        green = np.clip((exg + 1) / 3 * 255, 0, 255).astype(np.uint8)
        green = cv2.GaussianBlur(green, (5, 5), 0)
        self._add_step("1. Excess Green index", green)

        _, mask = cv2.threshold(green, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Otsu splits "greener" from "less green". Trees cover the SMALLER part of an orchard: on
        # bare soil they are the greener part, on green grass (e.g. silvery olive trees) the less green.
        if (mask > 0).mean() > 0.5:
            mask = cv2.bitwise_not(mask)
            green = 255 - green
            self._add_step("1b. Ground is greener than the trees -> trees = the less-green part", green)

        if min_crown <= 0:                                      # auto: typical crown size by granulometry
            radius, _, spectrum = utils.granulometry(mask)
            min_crown = 1.2 * radius                            # accept crowns down to ~60 % of typical
            self._add_step(f"Granulometry: typical crown radius {radius:.0f} px", utils.plot_spectrum(spectrum))
        min_crown = max(4.0, min_crown)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, utils.disk(min_crown / 8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, utils.disk(2))
        mask = utils.fill_small_holes(mask, np.pi * (min_crown / 2) ** 2)   # gaps between leaf clumps
        self._add_step("2. Tree mask (Otsu + opening + hole filling)", mask)
        return green, mask, min_crown

    def count_watershed(self, mask: np.ndarray, min_crown: float) -> list[tuple[float, float, float]]:
        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        self._add_step("3. Distance transform (bright = crown centre)", utils.normalize_to_uint8(dist))
        r_min = min_crown / 2
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(r_min * 1.4) | 1,) * 2)
        peaks = ((dist >= cv2.dilate(dist, kernel)) & (dist > 0.6 * r_min)).astype(np.uint8)
        n, seeds = cv2.connectedComponents(cv2.dilate(peaks, np.ones((3, 3), np.uint8)))
        labels = watershed(-dist, seeds, mask=mask > 0)

        vis = np.zeros((*mask.shape, 3), np.uint8)
        rng = np.random.default_rng(1)
        crowns = []
        for i in range(1, n):
            region = labels == i
            area = region.sum()
            if area < self.MIN_AREA * np.pi * r_min ** 2:     # smaller than the smallest accepted crown
                continue
            ys, xs = np.nonzero(region)
            vis[region] = rng.integers(60, 255, 3)
            crowns.append((float(xs.mean()), float(ys.mean()), float(np.sqrt(area / np.pi))))
        self._add_step("4. Watershed: one colour per crown", vis)
        return crowns

    def count_log(self, green: np.ndarray, mask: np.ndarray, min_crown: float) -> list[tuple[float, float, float]]:
        smooth = cv2.GaussianBlur(green, (0, 0), 2).astype(np.float32) / 255 * (mask > 0)
        f = 0.5 if min(smooth.shape) > 400 else 1.0             # half size: 4x faster, same crowns
        small = cv2.resize(smooth, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
        r_min = min_crown / 2 * f
        blobs = blob_log(small, min_sigma=r_min / np.sqrt(2), max_sigma=4 * r_min / np.sqrt(2),
                         num_sigma=12, threshold=0.05, overlap=0.5)
        crowns = [(float(x / f), float(y / f), float(s * np.sqrt(2) / f)) for y, x, s in blobs
                  if mask[min(int(y / f), mask.shape[0] - 1), min(int(x / f), mask.shape[1] - 1)]]
        vis = cv2.cvtColor(utils.normalize_to_uint8(smooth), cv2.COLOR_GRAY2BGR)
        for x, y, r in crowns:
            cv2.circle(vis, (int(x), int(y)), int(r), (0, 200, 255), 1, cv2.LINE_AA)
        self._add_step("3. LoG blobs on the greenness map", vis)
        return crowns

    @staticmethod
    def _annotate(img, crowns) -> np.ndarray:
        out = img.copy()
        for i, (x, y, r) in enumerate(crowns, 1):
            cv2.circle(out, (int(x), int(y)), int(r), (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(out, str(i), (int(x) - 7, int(y) + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (255, 255, 255), 1, cv2.LINE_AA)
        utils.put_label(out, f"Trees: {len(crowns)}", (10, 28), (255, 255, 255), 0.8)
        return out


if __name__ == "__main__":
    from core.cli import run_cli
    run_cli(TreeCounter)
