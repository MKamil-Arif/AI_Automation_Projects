"""
Topic 4 - Face Recognition Attendance System (attendance saved to CSV + Excel).

  1. Faces found with YuNet (core.faces), aligned with the 5 landmarks (eyes level, fixed size)
  2. SFace CNN turns every aligned face into a 128-number embedding ("face fingerprint")
  3. Recognition = cosine similarity between the embedding and every enrolled person's
     embeddings; above the threshold (0.363, SFace's published value) -> that person
  4. Mark attendance: one row per person per day in attendance/attendance_log.csv, and the
     whole log is also written to attendance/attendance.xlsx

Enrolled people live in faces_db/<name>/*.jpg (use mode "Enroll person" with a webcam photo).
Run alone:  python -m topics.t04_face_attendance photo.jpg mode=Recognise
"""
from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import ClassVar

import cv2
import numpy as np
import pandas as pd

from core import utils
from core.base import DIPModule, ModuleResult, Param
from core.faces import MODELS_DIR, Face, FaceDetector, draw_face
from core.registry import register

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SFACE = MODELS_DIR / "face_recognition_sface_2021dec.onnx"
ATTENDANCE_DIR = PROJECT_ROOT / "attendance"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _show(path: Path) -> str:
    """Path relative to the project if it is inside it, else the full path."""
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


class FaceGallery:
    """Embeddings of every enrolled person, rebuilt only when the folder changes."""

    _cache: ClassVar[dict[tuple, "FaceGallery"]] = {}

    def __init__(self, folder: Path, encoder: "FaceAttendance") -> None:
        self.names: list[str] = []
        vectors = []
        for person in sorted(p for p in folder.iterdir() if p.is_dir()):
            for file in sorted(person.iterdir()):
                if file.suffix.lower() not in IMAGE_EXTS:
                    continue
                img = utils.read_image(file)
                faces = FaceDetector.shared().detect(img, 0.7)
                if faces:
                    biggest = max(faces, key=lambda f: f.box[2] * f.box[3])
                    vectors.append(encoder.embed(img, biggest))
                    self.names.append(person.name)
        self.vectors = np.array(vectors).reshape(-1, 128)

    @classmethod
    def get(cls, folder: Path, encoder: "FaceAttendance") -> "FaceGallery":
        files = [p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTS]
        key = (str(folder), len(files), max((p.stat().st_mtime for p in files), default=0))
        if key not in cls._cache:
            cls._cache[key] = cls(folder, encoder)
        return cls._cache[key]

    @property
    def people(self) -> list[str]:
        return sorted(set(self.names))

    def identify(self, vector: np.ndarray, threshold: float) -> tuple[str, float]:
        if not len(self.vectors):
            return "Unknown", 0.0
        sims = self.vectors @ vector                       # rows are unit length -> cosine similarity
        best = int(np.argmax(sims))
        return (self.names[best] if sims[best] >= threshold else "Unknown"), float(sims[best])


@register
class FaceAttendance(DIPModule):
    topic_no = 4
    name = "Face Recognition Attendance"
    description = ("Recognises enrolled people (YuNet detection + SFace embeddings + cosine similarity) "
                   "and marks their attendance once per day in a CSV / Excel sheet.")
    samples = ("face_astronaut.png", "face_cameraman.png")
    confirm_modes = ("Mark attendance", "Enroll person")   # these write files: only on a button press

    _recognizer = None

    def parameters(self) -> list[Param]:
        return [
            Param("mode", "Mode", "select", "Recognise",
                  options=("Recognise", "Mark attendance", "Enroll person", "Attendance sheet")),
            Param("person", "Name of the person to enroll", "text", "",
                  visible_when={"mode": "Enroll person"}),
            Param("threshold", "Match threshold (cosine similarity)", "slider", 0.36, 0.2, 0.7, 0.01,
                  help="SFace's recommended value is 0.363. Higher = stricter."),
            Param("database", "Enrolled faces folder", "text", "faces_db"),
        ]

    # ----- pipeline -----------------------------------------------------------
    def process(self, image, *, mode, person, threshold, database) -> ModuleResult:
        folder = Path(database) if Path(database).is_absolute() else PROJECT_ROOT / database
        folder.mkdir(parents=True, exist_ok=True)
        img, _ = utils.resize_max(utils.ensure_bgr(image), 1280)

        if mode == "Attendance sheet":
            log = self.read_log()
            today = log[log["Date"] == datetime.now().strftime("%Y-%m-%d")] if len(log) else log
            return ModuleResult(img, info={"Present today": len(today), "Total records": len(log),
                                           "Today": today.to_dict("records"),
                                           "File": _show(ATTENDANCE_DIR / "attendance.xlsx")})

        faces = FaceDetector.shared().detect(img, 0.7)
        if not faces:
            raise ValueError("No face found in the image.")
        found = img.copy()
        for f in faces:
            draw_face(found, f, "", landmarks=True)
        self._add_step(f"1. Faces found ({len(faces)})", found)

        if mode == "Enroll person":
            return self.enroll(img, faces, person.strip(), folder)

        gallery = FaceGallery.get(folder, self)
        if not gallery.people:
            raise ValueError("Nobody is enrolled yet. Use mode 'Enroll person' first (one photo per person is enough; "
                             "3-5 photos work better).")
        out, rows = img.copy(), []
        for i, face in enumerate(faces, 1):
            aligned = self.align(img, face)
            if i <= 3:
                self._add_step(f"2. Face {i} aligned for SFace (112x112)", aligned)
            name, sim = gallery.identify(self.embed(img, face), threshold)
            draw_face(out, face, f"{name} {sim:.2f}", (0, 220, 0) if name != "Unknown" else (0, 0, 255))
            rows.append({"Face": i, "Name": name, "Similarity": round(sim, 3)})

        info = {"Faces": len(faces), "Recognised": sum(r["Name"] != "Unknown" for r in rows),
                "Enrolled people": len(gallery.people), "Results": rows}
        if mode == "Mark attendance":
            marked, already = self.mark([r for r in rows if r["Name"] != "Unknown"])
            info["Marked now"] = ", ".join(marked) or "-"
            info["Already marked today"] = ", ".join(already) or "-"
        return ModuleResult(out, info=info)

    # ----- face embedding -------------------------------------------------------
    @classmethod
    def _model(cls):
        if cls._recognizer is None:
            if not SFACE.exists():
                raise FileNotFoundError(f"{SFACE.name} missing in models/ (github.com/opencv/opencv_zoo, face_recognition_sface).")
            cls._recognizer = cv2.FaceRecognizerSF.create(str(SFACE), "")
        return cls._recognizer

    def align(self, img: np.ndarray, face: Face) -> np.ndarray:
        """Rotate / scale so the eyes are level and the face fills a 112x112 square."""
        return self._model().alignCrop(img, face.raw)

    def embed(self, img: np.ndarray, face: Face) -> np.ndarray:
        vec = self._model().feature(self.align(img, face)).ravel().astype(np.float64)
        return vec / (np.linalg.norm(vec) + 1e-9)

    # ----- enrollment / attendance ----------------------------------------------
    def enroll(self, img, faces, person, folder) -> ModuleResult:
        if not person:
            raise ValueError("Type the person's name first.")
        if len(faces) != 1:
            raise ValueError(f"Enrollment photo must show exactly one face (found {len(faces)}).")
        safe = "".join(ch for ch in person if ch.isalnum() or ch in " _-").strip()
        target = folder / safe
        target.mkdir(parents=True, exist_ok=True)
        path = target / f"{datetime.now():%Y%m%d_%H%M%S}.jpg"
        utils.write_image(path, img)
        out = img.copy()
        draw_face(out, faces[0], f"enrolled: {safe}")
        count = len([p for p in target.iterdir() if p.suffix.lower() in IMAGE_EXTS])
        return ModuleResult(out, info={"Enrolled": safe, "Photos of this person": count,
                                       "Saved to": _show(path)})

    @staticmethod
    def read_log() -> pd.DataFrame:
        path = ATTENDANCE_DIR / "attendance_log.csv"
        return pd.read_csv(path) if path.exists() else pd.DataFrame(columns=["Date", "Time", "Name", "Similarity"])

    def mark(self, recognised: list[dict]) -> tuple[list[str], list[str]]:
        ATTENDANCE_DIR.mkdir(exist_ok=True)
        log = self.read_log()
        today = datetime.now().strftime("%Y-%m-%d")
        present = set(log.loc[log["Date"] == today, "Name"]) if len(log) else set()
        marked, already = [], []
        path = ATTENDANCE_DIR / "attendance_log.csv"
        new_file = not path.exists()
        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if new_file:
                writer.writerow(["Date", "Time", "Name", "Similarity"])
            for r in recognised:
                if r["Name"] in present:
                    already.append(r["Name"])
                    continue
                writer.writerow([today, datetime.now().strftime("%H:%M:%S"), r["Name"], r["Similarity"]])
                present.add(r["Name"])
                marked.append(r["Name"])
        self.read_log().to_excel(ATTENDANCE_DIR / "attendance.xlsx", index=False)
        return marked, already


if __name__ == "__main__":
    from core.cli import run_cli
    run_cli(FaceAttendance)
