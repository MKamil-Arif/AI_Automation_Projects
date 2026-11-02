"""
Fine-tunes YOLOv8n (official COCO-pretrained weights from Ultralytics) as a learned detector for
potholes (topic 13) and number plates (topic 5). Training uses only the train images; the
reported numbers come from images the model never saw.

    python evaluation/train_detectors.py potholes
    python evaluation/train_detectors.py plates

Weights are copied to models/pothole_yolov8n.pt and models/plate_yolov8n.pt.
"""
from __future__ import annotations

import json
import os
import random
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REAL = ROOT / "datasets" / "real"
WORK = ROOT / "datasets" / "yolo"


def to_yolo(items: list[tuple[Path, dict]], split_dir: Path) -> None:
    """Copy images and write YOLO label files: 'class cx cy w h' (all 0..1)."""
    (split_dir / "images").mkdir(parents=True, exist_ok=True)
    (split_dir / "labels").mkdir(parents=True, exist_ok=True)
    for img_path, ann in items:
        name = f"{img_path.parent.name}_{img_path.stem}"
        shutil.copy(img_path, split_dir / "images" / f"{name}.jpg")
        W, H = ann["width"], ann["height"]
        lines = [f"0 {(x + w / 2) / W:.6f} {(y + h / 2) / H:.6f} {w / W:.6f} {h / H:.6f}"
                 for x, y, w, h in ann["boxes"] if w > 1 and h > 1]
        (split_dir / "labels" / f"{name}.txt").write_text("\n".join(lines))


def collect(folders: list[Path]) -> list[tuple[Path, dict]]:
    items = []
    for folder in folders:
        for ann in json.loads((folder / "annotations.json").read_text()):
            items.append((folder / ann["file"], ann))
    return items


def train(name: str, train_items, val_items, imgsz: int, epochs: int, target: Path) -> dict:
    from ultralytics import YOLO

    data_dir = WORK / name
    if data_dir.exists():
        shutil.rmtree(data_dir)
    to_yolo(train_items, data_dir / "train")
    to_yolo(val_items, data_dir / "val")
    yaml = data_dir / "data.yaml"
    yaml.write_text(f"path: {data_dir.as_posix()}\ntrain: train/images\nval: val/images\nnames:\n  0: {name[:-1]}\n")

    os.chdir(WORK)                                         # yolov8n.pt is downloaded / runs/ written here
    model = YOLO("yolov8n.pt")
    model.train(data=str(yaml), imgsz=imgsz, epochs=epochs, batch=8, device="cpu", workers=0, patience=15,
                project=str(WORK / "runs"), name=name, exist_ok=True, plots=False, verbose=False, seed=0)
    best = WORK / "runs" / name / "weights" / "best.pt"
    shutil.copy(best, target)
    metrics = YOLO(str(target)).val(data=str(yaml), imgsz=imgsz, device="cpu", workers=0, plots=False, verbose=False)
    return {"train_images": len(train_items), "test_images": len(val_items), "imgsz": imgsz, "epochs": epochs,
            "mAP50": round(float(metrics.box.map50), 4), "mAP50-95": round(float(metrics.box.map), 4),
            "precision": round(float(metrics.box.mp), 4), "recall": round(float(metrics.box.mr), 4)}


def potholes() -> dict:
    items = collect([REAL / "potholes" / s for s in ("train", "validation", "test")])
    random.Random(0).shuffle(items)
    return train("potholes", items[:65], items[65:], 512, 60, ROOT / "models" / "pothole_yolov8n.pt")


def plates() -> dict:
    return train("plates", collect([REAL / "plates_train"]), collect([REAL / "plates_test"]), 480, 25,
                 ROOT / "models" / "plate_yolov8n.pt")


if __name__ == "__main__":
    results_file = Path(__file__).resolve().parent / "results.json"
    for key in sys.argv[1:]:
        result = {"potholes": potholes, "plates": plates}[key]()
        print(key, result, flush=True)
        results = json.loads(results_file.read_text()) if results_file.exists() else {}
        results[f"{key}_yolo"] = {"dataset": "keremberke (Roboflow) - model never saw the test images", **result}
        results_file.write_text(json.dumps(results, indent=1, default=str))
