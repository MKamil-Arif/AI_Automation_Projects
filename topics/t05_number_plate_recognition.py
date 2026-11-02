"""
Topic 5 - Automatic Number Plate Recognition (ANPR): find the plate, read its text.

Plate localisation, learned (default): YOLOv8n fine-tuned in this project on real car photos
  (evaluation/train_detectors.py) - copes with any angle, distance and plate style.
Plate localisation, classical (no training)
  1. Gray + bilateral filter (smooths, keeps edges)
  2. Black-hat with a wide rectangle: dark characters on a light plate pop out
  3. Sobel in x: characters make many vertical edges side by side
  4. Closing with a wide kernel joins the characters into one blob -> Otsu -> contours
  5. Keep blobs with a text-line shape (width / height 2-8, enough area), best edge density first
Reading
  6. Plate cut out, enlarged, Otsu-binarised; characters = connected components (shown)
  7. OCR: EasyOCR (offline CRNN, downloads ~100 MB once) or Gemini (cloud)
  8. Clean-up for Pakistani style plates (letters then digits): O->0, I->1, S->5, B->8 in the
     digit part and the reverse in the letter part

Run alone:  python -m topics.t05_number_plate_recognition samples/car_plate_LEB_1234.png
"""
from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np

from core import cloud, utils
from core.base import DIPModule, ModuleResult, Param
from core.registry import register
from core.yolo import YoloDetector

EASYOCR_DIR = Path(__file__).resolve().parent.parent / "models" / "easyocr"
YOLO_WEIGHTS = "plate_yolov8n.pt"
TO_DIGIT = str.maketrans({"O": "0", "D": "0", "Q": "0", "I": "1", "L": "1", "Z": "2", "S": "5", "B": "8", "G": "6", "T": "7"})
TO_LETTER = str.maketrans({"0": "O", "1": "I", "2": "Z", "5": "S", "8": "B", "6": "G", "7": "T", "4": "A"})


@register
class NumberPlateReader(DIPModule):
    topic_no = 5
    name = "Number Plate Recognition (ANPR)"
    description = ("Finds the number plate (YOLOv8n trained on real cars, or classical black-hat + Sobel + "
                   "closing) and reads it with EasyOCR (offline) or Gemini (cloud), with plate-format clean-up.")
    samples = ("car_plate_LEB_1234.png", "car_plate_ICT_4521.png", "car_plate_KHI_7788.png")

    _reader = None

    def parameters(self) -> list[Param]:
        return [
            Param("locator", "Plate finder", "select", "YOLOv8 (trained)",
                  options=("YOLOv8 (trained)", "Classical morphology")),
            Param("ocr", "Text reader", "select", "EasyOCR (offline)", options=("EasyOCR (offline)", "Gemini (cloud)")),
            Param("fix_format", "Fix letters/digits (Pakistani format ABC 1234)", "checkbox", True),
        ]

    # ----- pipeline -----------------------------------------------------------
    def process(self, image, *, locator, ocr, fix_format) -> ModuleResult:
        img, _ = utils.resize_max(utils.ensure_bgr(image), 1200)
        self._add_step("Original", img)
        yolo = YoloDetector.get(YOLO_WEIGHTS) if locator.startswith("YOLO") else None
        if yolo is not None:
            candidates = [b[:4] for b in yolo.detect(img, conf=0.25, imgsz=480)]
            vis = img.copy()
            for x, y, w, h in candidates[:3]:
                cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
            self._add_step("1. YOLOv8 plate detections", vis)
            margin_y = 0.1                                         # YOLO boxes cover the whole plate already
        else:
            candidates = self.find_plates(img)
            margin_y = 0.5                                         # the blob hugs the text: add margin
        if not candidates:
            raise ValueError("No number plate found. Try a closer, front-facing photo.")

        out, rows = img.copy(), []
        for rank, (x, y, w, h) in enumerate(candidates[:3], 1):
            px, py = int(0.06 * w), int(margin_y * h)
            plate = img[max(0, y - py):y + h + py, max(0, x - px):x + w + px]
            if rank == 1:
                self._add_step("5. Plate cut out", utils.enlarge(plate, 2))
                self._show_characters(plate)
            raw = self.read_text(plate, ocr)
            text = self.clean(raw, fix_format)
            if len(re.sub(r"[^A-Z0-9]", "", text)) < 3:             # not really a plate
                continue
            rows.append({"Plate": text, "Raw OCR": raw, "Box": f"{x},{y} {w}x{h}"})
            cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 3)
            utils.put_label(out, text, (x, max(20, y - 10)), (0, 255, 0), 0.9)
            break                                                  # the best plate is enough
        if not rows:
            raise ValueError("A plate-shaped region was found but no readable text in it.")
        info = {"Plate number": rows[0]["Plate"], "Plate finder": "YOLOv8" if yolo else "classical",
                "OCR engine": ocr, "Candidates checked": min(len(candidates), 3), "Details": rows}
        if locator.startswith("YOLO") and yolo is None:
            info["Note"] = f"models/{YOLO_WEIGHTS} not found - used classical morphology"
        return ModuleResult(out, info=info)

    # ----- localisation ------------------------------------------------------------
    def find_plates(self, img: np.ndarray) -> list[tuple[int, int, int, int]]:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 9, 50, 50)
        h, w = gray.shape
        k = max(9, w // 60)
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (3 * k, k)))
        self._add_step("1. Black-hat (dark text on a light plate)", blackhat)
        grad = np.abs(cv2.Sobel(blackhat, cv2.CV_32F, 1, 0, ksize=3))
        grad = utils.normalize_to_uint8(grad)
        self._add_step("2. Sobel x (vertical character edges)", grad)
        closed = cv2.morphologyEx(cv2.GaussianBlur(grad, (5, 5), 0), cv2.MORPH_CLOSE,
                                  cv2.getStructuringElement(cv2.MORPH_RECT, (2 * k, max(3, k // 2))))
        _, mask = cv2.threshold(closed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        mask = cv2.dilate(cv2.erode(mask, None, iterations=2), None, iterations=2)
        self._add_step("3. Closing + Otsu: characters merged into blobs", mask)

        boxes = []
        for c in cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
            (cx, cy), (rw, rh), angle = cv2.minAreaRect(c)
            bw, bh = max(rw, rh), min(rw, rh)
            # the blob is the TEXT line (thinner than the plate itself), so allow up to 8 : 1
            if bh == 0 or not (2.0 <= bw / bh <= 8.0) or bw * bh < 0.002 * h * w or bw > 0.6 * w:
                continue
            x, y, ww, hh = cv2.boundingRect(c)
            density = float((grad[y:y + hh, x:x + ww] > 60).mean())      # plates are full of edges
            boxes.append((density, (x, y, ww, hh)))
        boxes.sort(key=lambda b: -b[0])
        vis = img.copy()
        for i, (_, (x, y, ww, hh)) in enumerate(boxes[:5]):
            cv2.rectangle(vis, (x, y), (x + ww, y + hh), (0, 255, 0) if i == 0 else (0, 165, 255), 2)
        self._add_step("4. Plate-shaped candidates (green = best)", vis)
        return [b for _, b in boxes]

    def _show_characters(self, plate: np.ndarray) -> None:
        big = cv2.resize(plate, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        n, _, stats, _ = cv2.connectedComponentsWithStats(binary)
        vis = cv2.cvtColor(255 - binary, cv2.COLOR_GRAY2BGR)
        H = binary.shape[0]
        for x, y, w, h, _area in stats[1:]:
            if 0.3 * H < h < 0.95 * H and w < 0.3 * binary.shape[1]:  # character-sized blobs only
                cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 0, 255), 2)
        self._add_step("6. Otsu binarisation + character blobs", vis)

    # ----- reading ------------------------------------------------------------------
    @classmethod
    def _easyocr(cls):
        if cls._reader is None:
            import easyocr
            EASYOCR_DIR.mkdir(parents=True, exist_ok=True)
            cls._reader = easyocr.Reader(["en"], gpu=False, model_storage_directory=str(EASYOCR_DIR), verbose=False)
        return cls._reader

    def read_text(self, plate: np.ndarray, ocr: str) -> str:
        if ocr.startswith("Gemini"):
            big = cv2.resize(plate, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            return cloud.gemini("Read the vehicle number plate in this image. Reply with the plate text only.", big)
        # EasyOCR reads small plates better at a fixed height of 128 px, gray + CLAHE
        # (tuned on the validation split: character accuracy 61.7 % -> 65.3 %)
        f = 128 / max(plate.shape[0], 1)
        gray = cv2.cvtColor(cv2.resize(plate, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC), cv2.COLOR_BGR2GRAY)
        big = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(gray)
        results = [r for r in self._easyocr().readtext(big, allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789- ",
                                                         paragraph=False) if r[2] > 0.1]
        if not results:
            return ""
        # reading order: line by line (two-line plates), left to right inside a line
        line_h = max(1.0, float(np.median([abs(r[0][2][1] - r[0][0][1]) for r in results])))
        lines, line_y = [], None
        for r in sorted(results, key=lambda r: (r[0][0][1] + r[0][2][1]) / 2):
            y = (r[0][0][1] + r[0][2][1]) / 2
            if line_y is None or y - line_y > 0.6 * line_h:       # clearly lower: a new text line
                lines.append([])
                line_y = y
            lines[-1].append(r)
        return " ".join(r[1] for line in lines for r in sorted(line, key=lambda r: r[0][0][0]))

    @staticmethod
    def clean(raw: str, fix_format: bool) -> str:
        text = re.sub(r"[^A-Z0-9 ]", " ", raw.upper())
        text = re.sub(r"\s+", " ", text).strip()
        if not fix_format:
            return text
        compact = text.replace(" ", "")
        m = re.match(r"^([A-Z0-9]{2,3}?)([0-9A-Z]{3,4})$", compact)
        if not m:
            return text
        letters, digits = m.group(1).translate(TO_LETTER), m.group(2).translate(TO_DIGIT)
        return f"{letters} {digits}"


if __name__ == "__main__":
    from core.cli import run_cli
    run_cli(NumberPlateReader)
