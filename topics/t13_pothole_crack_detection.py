"""
Topic 13 - Road Pothole & Crack Detection (for municipality road surveys).

Potholes, learned (default): YOLOv8n fine-tuned in this project on real road photos
  (evaluation/train_detectors.py) -> boxes + confidence. Works on normal street-level photos.
Potholes, classical = dark blobs (best for top-down views of asphalt)
  1. Gray -> CLAHE -> median blur (removes asphalt grain)
  2. Local background = a very large median blur of the road; darkness = background - image
     (works even if one side of the road is in light and the other in shade)
  3. Otsu threshold on darkness (never below a minimum), morphology, contours
  4. Keep blob-like shapes only (solid, thick enough), grade severity by size
Cracks = thin dark lines
  1. Sato ridge filter (black ridges, sigma 1-3) responds to thin dark lines
  2. Threshold, ignore pothole areas, skeletonize -> crack length in pixels
  3. Keep long, thin pieces only

Real datasets to try: Kaggle "Pothole Detection" images, RDD2022 (road damage).
Run alone:  python -m topics.t13_pothole_crack_detection samples/road_synthetic.png
"""
from __future__ import annotations

import cv2
import numpy as np
from skimage.filters import sato
from skimage.morphology import skeletonize

from core import utils
from core.base import DIPModule, ModuleResult, Param
from core.registry import register
from core.yolo import YoloDetector

YOLO_WEIGHTS = "pothole_yolov8n.pt"


@register
class PotholeCrackDetector(DIPModule):
    topic_no = 13
    name = "Road Pothole & Crack Detection"
    description = ("Finds potholes (YOLOv8n trained on real roads, or dark blobs darker than the road) and "
                   "cracks (thin dark lines, ridge filter + skeleton), grades severity and suggests a repair priority.")
    samples = ("road_synthetic.png",)

    WORK_SIZE = 1000
    POTHOLE_COLOUR, CRACK_COLOUR = (0, 0, 255), (0, 220, 255)

    def parameters(self) -> list[Param]:
        return [
            Param("detect", "Detect", "select", "Potholes + cracks",
                  options=("Potholes + cracks", "Potholes only", "Cracks only")),
            Param("detector", "Pothole detector", "select", "YOLOv8 (trained)",
                  options=("YOLOv8 (trained)", "Classical (darkness)"),
                  help="YOLO: learned from real road photos. Classical: dark-blob rule for top-down asphalt."),
            Param("confidence", "YOLO confidence threshold", "slider", 0.25, 0.05, 0.9, 0.05,
                  visible_when={"detector": "YOLOv8 (trained)"}),
            Param("sensitivity", "Sensitivity", "slider", 5, 1, 10, 1,
                  help="Higher = also marks slightly dark areas (more detections, more false alarms)."),
            Param("min_pothole", "Min pothole size (% of image)", "slider", 0.2, 0.05, 3.0, 0.05),
            Param("min_crack", "Min crack length (px)", "slider", 60, 10, 400, 10),
        ]

    # ----- pipeline -----------------------------------------------------------
    def process(self, image, *, detect, detector, confidence, sensitivity, min_pothole, min_crack) -> ModuleResult:
        img, scale = utils.resize_max(utils.ensure_bgr(image), self.WORK_SIZE)
        self._add_step("Original", img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        equalized = cv2.createCLAHE(2.0, (8, 8)).apply(gray)
        smooth = cv2.medianBlur(equalized, 5)                     # for potholes: grain removed
        fine = cv2.GaussianBlur(equalized, (0, 0), 1.0)           # for cracks: keeps 1-2 px lines
        self._add_step("1. CLAHE + median blur", smooth)

        # local road brightness: a median far wider than any pothole or crack
        background = cv2.medianBlur(smooth, (min(gray.shape) // 4) | 1)

        potholes, pothole_mask, note = [], np.zeros(gray.shape, np.uint8), None
        if detect != "Cracks only":
            yolo = YoloDetector.get(YOLO_WEIGHTS) if detector.startswith("YOLO") else None
            if yolo is not None:
                potholes, pothole_mask = self.find_potholes_yolo(img, yolo, confidence)
            else:
                if detector.startswith("YOLO"):
                    note = f"models/{YOLO_WEIGHTS} not found - used the classical detector"
                potholes, pothole_mask = self.find_potholes(smooth, background, sensitivity, min_pothole)
        crack_skeleton = np.zeros(gray.shape, bool)
        if detect != "Potholes only":
            crack_skeleton = self.find_cracks(fine, background, pothole_mask, sensitivity, min_crack)

        output = self._annotate(img, potholes, crack_skeleton)
        grades = [p["severity"] for p in potholes]
        crack_len = int(crack_skeleton.sum())
        damage = (pothole_mask > 0).mean() + crack_len * 3 / gray.size     # cracks ~3 px wide
        info = {
            "Potholes": len(potholes),
            "Large / Med / Small": f"{grades.count('large')} / {grades.count('medium')} / {grades.count('small')}",
            "Crack length": f"{crack_len} px",
            "Damaged area": f"{damage:.1%}",
            "Repair priority": self.priority(grades, crack_len, damage),
        }
        if note:
            info["Note"] = note
        if potholes:
            info["Pothole details"] = [{"#": i + 1, "Severity": p["severity"],
                                        "Area (% of image)": round(100 * p["area"] / gray.size, 2),
                                        "Score": p["score"],
                                        # box in the ORIGINAL image's pixels (x, y, w, h)
                                        "Box": [round(v / scale) for v in cv2.boundingRect(p["contour"])]}
                                       for i, p in enumerate(potholes)]
        return ModuleResult(output, info=info)

    # ----- potholes: learned detector -------------------------------------------
    def find_potholes_yolo(self, img: np.ndarray, yolo: YoloDetector, confidence: float):
        h, w = img.shape[:2]
        potholes, mask = [], np.zeros((h, w), np.uint8)
        vis = img.copy()
        for x, y, bw, bh, conf in yolo.detect(img, conf=confidence, imgsz=512):
            contour = np.array([[[x, y]], [[x + bw, y]], [[x + bw, y + bh]], [[x, y + bh]]], np.int32)
            diameter = 2 * np.sqrt(bw * bh / np.pi) / w
            severity = "small" if diameter < 0.08 else "medium" if diameter < 0.15 else "large"
            potholes.append({"contour": contour, "area": bw * bh, "severity": severity, "score": f"{conf:.0%} conf"})
            cv2.rectangle(mask, (x, y), (x + bw, y + bh), 255, -1)
            cv2.rectangle(vis, (x, y), (x + bw, y + bh), (0, 0, 255), 2)
            utils.put_label(vis, f"{conf:.2f}", (x, max(14, y - 4)), (255, 255, 255), 0.45)
        self._add_step(f"2. YOLOv8 detections (confidence >= {confidence:.2f})", vis)
        return potholes, mask

    # ----- potholes -----------------------------------------------------------
    def find_potholes(self, smooth: np.ndarray, background: np.ndarray, sensitivity: int, min_pothole: float):
        h, w = smooth.shape
        darkness = cv2.subtract(background, smooth)               # >0 where darker than surroundings
        self._add_step("2. Darkness = local background - image", utils.normalize_to_uint8(darkness))

        otsu, _ = cv2.threshold(darkness, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        floor = 50 - 4 * sensitivity                              # sensitivity 5 -> at least 30 levels
        mask = (darkness > max(otsu, floor)).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, utils.disk(3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, utils.disk(7))
        self._add_step(f"3. Dark regions (threshold {max(otsu, floor):.0f})", mask)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        potholes, kept = [], np.zeros_like(mask)
        for c in contours:
            area = cv2.contourArea(c)
            if area < min_pothole / 100 * h * w:
                continue
            solidity = area / max(cv2.contourArea(cv2.convexHull(c)), 1)
            blob = np.zeros_like(mask)
            cv2.drawContours(blob, [c], -1, 255, -1)
            thickness = cv2.distanceTransform(blob, cv2.DIST_L2, 5).max()   # radius of biggest inscribed circle
            if solidity < 0.6 or thickness < 6:                    # thin / branchy -> a crack, not a pothole
                continue
            diameter = 2 * np.sqrt(area / np.pi) / w               # relative to road width in the photo
            severity = "small" if diameter < 0.08 else "medium" if diameter < 0.15 else "large"
            potholes.append({"contour": c, "area": area, "severity": severity,
                             "score": f"darkness {int(darkness[blob > 0].mean())}"})
            kept |= blob
        return potholes, kept

    # ----- cracks -------------------------------------------------------------
    def find_cracks(self, fine: np.ndarray, background: np.ndarray, pothole_mask: np.ndarray,
                    sensitivity: int, min_crack: int):
        ridge = sato(fine.astype(np.float32) / 255, sigmas=(1, 2, 3), black_ridges=True)
        ridge_u8 = utils.normalize_to_uint8(ridge)
        self._add_step("4. Sato ridge filter (thin dark lines light up)", ridge_u8)

        otsu, _ = cv2.threshold(ridge_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thresh = max(otsu * (1.6 - 0.1 * sensitivity), 20)         # sensitivity 5 -> 1.1 x Otsu
        # a crack must also be darker than the road itself: the ridge filter alone also fires on
        # the normal road right next to bright lane markings
        darker = cv2.subtract(background, fine) > 22 - 2 * sensitivity
        mask = ((ridge_u8 > thresh) & darker).astype(np.uint8)
        mask[cv2.dilate(pothole_mask, utils.disk(9)) > 0] = 0            # pothole rims are not cracks

        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        skeleton = skeletonize(mask > 0)
        length = np.bincount(labels[skeleton], minlength=n)        # skeleton pixels per piece
        keep = np.zeros(n, bool)
        for i in range(1, n):
            width = stats[i, cv2.CC_STAT_AREA] / max(length[i], 1)
            keep[i] = length[i] >= min_crack and width <= 8
        cracks = skeleton & keep[labels]
        self._add_step("5. Crack skeleton (long, thin pieces)", (cracks * 255).astype(np.uint8))
        return cracks

    # ----- decision + drawing -------------------------------------------------
    @staticmethod
    def priority(grades: list[str], crack_len: int, damage: float) -> str:
        if "large" in grades or damage > 0.05:
            return "HIGH"
        if grades or crack_len > 300:
            return "MEDIUM"
        return "LOW" if crack_len else "None"

    def _annotate(self, img, potholes, crack_skeleton) -> np.ndarray:
        out = img.copy()
        cracks = cv2.dilate(crack_skeleton.astype(np.uint8), utils.disk(1)) > 0
        out[cracks] = self.CRACK_COLOUR
        for i, p in enumerate(potholes, 1):
            c = p["contour"]
            overlay = out.copy()
            cv2.drawContours(overlay, [c], -1, self.POTHOLE_COLOUR, -1)
            out = cv2.addWeighted(overlay, 0.3, out, 0.7, 0)
            cv2.drawContours(out, [c], -1, self.POTHOLE_COLOUR, 2, cv2.LINE_AA)
            x, y, _, _ = cv2.boundingRect(c)
            utils.put_label(out, f"#{i} {p['severity']}", (x, max(16, y - 6)), (255, 255, 255), 0.5)
        utils.put_label(out, f"Potholes: {len(potholes)}   Cracks: {int(crack_skeleton.sum())} px",
                        (10, 26), (255, 255, 255), 0.6)
        return out


if __name__ == "__main__":
    from core.cli import run_cli
    run_cli(PotholeCrackDetector)
