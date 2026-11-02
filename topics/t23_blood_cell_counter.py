"""
Topic 23 - Blood Cell Counter (RBC / WBC / Platelets) from a stained blood-smear image.

Pipeline
  1. Purple stain mask (HSV)        -> WBC nuclei + platelets
  2. Cell mask (Otsu, green channel)-> every cell; pale RBC centres filled with hole filling
  3. Typical RBC radius             -> median area of round, isolated cells
  4. Large purple blobs = WBC, tiny purple blobs = platelets
  5. RBC count                      -> distance transform + watershed (splits touching cells)
                                       or Hough circle transform

Real dataset to try: BCCD (Blood Cell Count and Detection).
Run alone:  python -m topics.t23_blood_cell_counter samples/blood_smear_synthetic.png
"""
from __future__ import annotations

import cv2
import numpy as np
from skimage.segmentation import watershed

from core import utils
from core.base import DIPModule, ModuleResult, Param
from core.registry import register


@register
class BloodCellCounter(DIPModule):
    topic_no = 23
    name = "Blood Cell Counter"
    description = ("Counts red cells (RBC), white cells (WBC) and platelets in a stained "
                   "blood-smear image using colour segmentation, morphology and watershed.")
    samples = ("blood_smear_synthetic.png",)

    PURPLE_HUE = (115, 160)        # OpenCV hue scale 0-179: stained nuclei look violet
    WBC_MIN_AREA = 0.30            # purple blob bigger than 30% of an RBC -> WBC nucleus
    PLATELET_MIN_AREA = 0.01       # smaller purple blobs (but not noise) -> platelet

    def parameters(self) -> list[Param]:
        return [
            Param("method", "RBC counting method", "select", "Watershed",
                  options=("Watershed", "Hough Circles"),
                  help="Watershed splits touching cells; Hough searches for circle shapes."),
            Param("cell_radius", "RBC radius in pixels (0 = auto)", "slider", 0, 0, 80, 1),
            Param("purple_saturation", "WBC stain sensitivity (min saturation)", "slider",
                  90, 30, 200, 5, help="Lower = also detects paler purple stain."),
            Param("show_labels", "Number every RBC", "checkbox", False),
        ]

    # ----- pipeline -----------------------------------------------------------
    def process(self, image, *, method, cell_radius, purple_saturation, show_labels) -> ModuleResult:
        # microscope photos can be 12+ megapixels; 1600 px keeps every cell large enough and the run fast
        img, scale = utils.resize_max(utils.ensure_bgr(image), 1600)
        cell_radius = cell_radius * scale
        self._add_step("Original", img)

        purple = self.purple_mask(img, purple_saturation)
        self._add_step("1. Purple stain mask (HSV threshold)", purple)

        raw_cells = self.cell_mask(img)
        radius = cell_radius or self.estimate_radius(raw_cells)
        rbc_area = np.pi * radius ** 2
        cells = self.fill_holes(raw_cells, max_area=0.6 * rbc_area)
        self._add_step("2. All cells (Otsu on green channel + hole filling)", cells)

        wbcs, platelets = self.split_purple(purple, radius)
        wbc_zone = np.zeros_like(cells)
        cv2.drawContours(wbc_zone, wbcs, -1, 255, -1)
        wbc_zone = cv2.dilate(wbc_zone, utils.disk(radius))          # nucleus + its cytoplasm
        rbc_mask = cv2.bitwise_and(cells, cv2.bitwise_not(wbc_zone))
        self._add_step("3. RBC mask (cells minus WBC regions)", rbc_mask)

        if method == "Watershed":
            rbcs = self.count_watershed(img, rbc_mask, radius)
        else:
            rbcs = self.count_hough(img, rbc_mask, radius)

        output = self._annotate(img, rbcs, wbcs, platelets, radius, show_labels)
        n_rbc, n_wbc = len(rbcs), len(wbcs)
        return ModuleResult(output, info={
            "RBC": n_rbc,
            "WBC": n_wbc,
            "Platelets": len(platelets),
            "RBC radius (px)": round(float(radius / scale), 1),          # in the input image's pixels
            "RBC : WBC": f"{n_rbc / n_wbc:.0f} : 1" if n_wbc else "-",
        })

    # ----- segmentation -------------------------------------------------------
    def purple_mask(self, img: np.ndarray, min_saturation: int) -> np.ndarray:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lo = np.array([self.PURPLE_HUE[0], min_saturation, 20], np.uint8)
        hi = np.array([self.PURPLE_HUE[1], 255, 235], np.uint8)
        mask = cv2.inRange(hsv, lo, hi)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, utils.disk(1))
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, utils.disk(2))

    @staticmethod
    def cell_mask(img: np.ndarray) -> np.ndarray:
        """Cells absorb green light, so they are darker than the background in the G channel."""
        green = cv2.GaussianBlur(img[:, :, 1], (5, 5), 0)
        _, mask = cv2.threshold(green, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return cv2.morphologyEx(mask, cv2.MORPH_OPEN, utils.disk(2))

    @staticmethod
    def fill_holes(mask: np.ndarray, max_area: float) -> np.ndarray:
        """Fill small enclosed holes (pale RBC centres), leave big background gaps alone."""
        return utils.fill_small_holes(mask, max_area)

    def estimate_radius(self, mask: np.ndarray) -> float:
        """Granulometry (pattern spectrum, core.utils): the opening radius at which the most cell
        area disappears is the typical cell radius - touching cells and specks do not fool it."""
        radius, _, spectrum = utils.granulometry(mask)
        self._add_step(f"Granulometry: area lost per opening radius (peak = cell radius {radius:.0f} px)",
                       utils.plot_spectrum(spectrum))
        return radius

    def split_purple(self, purple: np.ndarray, radius: float):
        """Join the lobes of one nucleus, then sort purple blobs by size."""
        merged = cv2.morphologyEx(purple, cv2.MORPH_CLOSE, utils.disk(radius * 0.3))
        contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rbc_area = np.pi * radius ** 2
        wbcs, platelets = [], []
        for c in contours:
            area = cv2.contourArea(c)
            if area >= self.WBC_MIN_AREA * rbc_area:
                wbcs.append(c)
            elif area >= max(4.0, self.PLATELET_MIN_AREA * rbc_area):
                platelets.append(c)
        return wbcs, platelets

    # ----- RBC counting -------------------------------------------------------
    def count_watershed(self, img: np.ndarray, rbc_mask: np.ndarray, radius: float) -> list[np.ndarray]:
        """Each cell centre is a peak of the distance transform -> one watershed seed per cell."""
        dist = cv2.distanceTransform(rbc_mask, cv2.DIST_L2, 5)
        self._add_step("4. Distance transform", utils.normalize_to_uint8(dist))

        local_max = dist >= cv2.dilate(dist, utils.disk(radius * 0.7))
        peaks = (local_max & (dist > 0.45 * radius)).astype(np.uint8)
        peaks = cv2.dilate(peaks, utils.disk(2))                      # merge flat-topped maxima
        n, seeds = cv2.connectedComponents(peaks)               # seeds = 1..n-1

        # Flood the inverted distance map (not the photo: the pale RBC centre is an edge in
        # the photo and would trap the seed). skimage floods by height, which is what we need.
        markers = watershed(-dist, seeds, mask=rbc_mask > 0)

        vis = np.zeros_like(img)
        rng = np.random.default_rng(0)
        cells = []
        for label in range(1, n):
            region = (markers == label).astype(np.uint8)
            if region.sum() < 0.25 * np.pi * radius ** 2:
                continue
            vis[region > 0] = rng.integers(60, 255, 3)
            contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cells.append(max(contours, key=cv2.contourArea))
        self._add_step("5. Watershed regions (one colour per cell)", vis)
        return cells

    def count_hough(self, img: np.ndarray, rbc_mask: np.ndarray, radius: float) -> list[np.ndarray]:
        gray = cv2.GaussianBlur(img[:, :, 1], (5, 5), 1.5)
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=radius * 1.2,
                                   param1=60, param2=16,
                                   minRadius=int(radius * 0.7), maxRadius=int(radius * 1.3) + 1)
        cells = []
        vis = cv2.cvtColor(rbc_mask, cv2.COLOR_GRAY2BGR)
        for x, y, r in (np.round(circles[0]).astype(int) if circles is not None else []):
            if 0 <= y < rbc_mask.shape[0] and 0 <= x < rbc_mask.shape[1] and rbc_mask[y, x]:
                cv2.circle(vis, (x, y), r, (0, 200, 0), 2)
                cells.append(cv2.ellipse2Poly((int(x), int(y)), (int(r), int(r)), 0, 0, 360, 10))
        self._add_step("4. Hough circles inside RBC mask", vis)
        return cells

    # ----- drawing ------------------------------------------------------------
    @staticmethod
    def _annotate(img, rbcs, wbcs, platelets, radius, show_labels) -> np.ndarray:
        out = img.copy()
        cv2.drawContours(out, rbcs, -1, (0, 0, 220), 1, cv2.LINE_AA)
        if show_labels:
            for i, c in enumerate(rbcs, 1):
                x, y, w, h = cv2.boundingRect(c)
                cv2.putText(out, str(i), (x + w // 2 - 6, y + h // 2 + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1, cv2.LINE_AA)
        for c in wbcs:
            (x, y), r = cv2.minEnclosingCircle(c)
            cv2.circle(out, (int(x), int(y)), int(r + radius * 0.6), (255, 80, 0), 2, cv2.LINE_AA)
            utils.put_label(out, "WBC", (int(x - r), int(y - r - radius * 0.6 - 4)), (255, 200, 120), 0.5)
        for c in platelets:
            (x, y), r = cv2.minEnclosingCircle(c)
            cv2.circle(out, (int(x), int(y)), int(r + 4), (0, 220, 255), 1, cv2.LINE_AA)
        utils.put_label(out, f"RBC: {len(rbcs)}  WBC: {len(wbcs)}  Platelets: {len(platelets)}",
                        (10, 25), (255, 255, 255), 0.6)
        return out


if __name__ == "__main__":
    from core.cli import run_cli
    run_cli(BloodCellCounter)
