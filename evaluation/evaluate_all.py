"""
Accuracy of every topic on REAL data (download first: python evaluation/fetch_datasets.py).

    python evaluation/evaluate_all.py               # all topics that have data
    python evaluation/evaluate_all.py fer corel     # only some

Results are merged into evaluation/results.json (used for EVALUATION.md).
"""
from __future__ import annotations

import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core import utils  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
REAL = ROOT / "datasets" / "real"
RESULTS = Path(__file__).resolve().parent / "results.json"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def images_in(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def iou(a, b) -> float:
    """Boxes as x, y, w, h."""
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw = max(0, min(ax2, bx2) - max(a[0], b[0]))
    ih = max(0, min(ay2, by2) - max(a[1], b[1]))
    inter = iw * ih
    return inter / max(a[2] * a[3] + b[2] * b[3] - inter, 1e-9)


def match_boxes(pred, truth, thr) -> tuple[int, int, int]:
    """Greedy one-to-one matching -> (true positives, false positives, false negatives)."""
    used, tp = set(), 0
    for p in pred:
        best = max(((iou(p, t), j) for j, t in enumerate(truth) if j not in used), default=(0, None))
        if best[0] >= thr:
            used.add(best[1])
            tp += 1
    return tp, len(pred) - tp, len(truth) - tp


# ===== topic evaluators =====================================================================
def eval_emotion() -> dict:
    """Topic 11: FER+ CNN on FER-2013 test faces (48x48, already cropped), raw and calibrated.
    The calibration is fitted on 1000 FER-2013 TRAINING faces (fer2013_val), never on the test faces."""
    from topics.t11_emotion_recognition import EmotionRecognizer
    m = EmotionRecognizer()
    cv_acc = m.calibrate(REAL / "fer2013_val")
    to_fer = {"surprised": "surprise", "contempt": "disgust"}
    out = {"dataset": "FER-2013 test (100 faces per emotion)", "calibration": "logistic regression on FER+ scores, "
           f"fitted on {len(images_in_tree(REAL / 'fer2013_val'))} FER-2013 training faces (5-fold {cv_acc:.1%})"}
    for name, calibrated in (("raw FER+", False), ("calibrated", True)):
        correct = total = 0
        per = Counter()
        for cls_dir in sorted((REAL / "fer2013_test").iterdir()):
            for f in images_in(cls_dir):
                probs = m.predict(utils.read_image(f), calibrated)
                best = max(probs, key=probs.get)
                hit = to_fer.get(best, best) == cls_dir.name
                correct += hit
                total += 1
                per[cls_dir.name] += hit
        out["images"] = total
        out[f"accuracy {name}"] = round(correct / total, 4)
        out[f"per_class {name}"] = {k: round(v / 100, 2) for k, v in sorted(per.items())}
    out["accuracy"] = out["accuracy calibrated"]
    return out


def images_in_tree(folder: Path) -> list[Path]:
    return [f for d in sorted(p for p in folder.iterdir() if p.is_dir()) for f in images_in(d)]


def eval_corel() -> dict:
    """Topic 12: every feature x distance on Corel-1K (leave-one-out P@10 and mAP)."""
    from topics.t12_cbir_search_engine import DISTANCES, EXTRACTORS, ImageIndex
    out = {"dataset": "Corel-1K (Wang), 10 classes x 100", "images": 1000, "results": []}
    for name, ext in EXTRACTORS.items():
        index = ImageIndex.get(REAL / "corel1k", ext)
        for metric in DISTANCES:
            r = index.evaluate(metric, 10)
            out["results"].append({"feature": name, "distance": metric,
                                   "P@10": round(r["precision"], 4), "mAP": round(r["map"], 4)})
    out["best"] = max(out["results"], key=lambda r: r["mAP"])
    return out


def train_eval(module_cls, dataset: str, note: str, **params) -> dict:
    m = module_cls()
    start = time.time()
    r = m.run(np.zeros((64, 64, 3), np.uint8), mode="Train from folder", dataset=dataset, **params)
    return {"dataset": note, "images": r.info["Images"], "test_accuracy": r.info.get("Test accuracy"),
            "per_class": r.info.get("Per class (test split)"), "seconds": round(time.time() - start)}


def compare_features(module_cls, folder: Path, note: str) -> dict:
    """Classical vs deep vs both on ONE shared 75/25 split (features extracted once per image)."""
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
    from core.recognition import FeatureClassifier

    m = module_cls()
    C, D, y = [], [], []
    for cls_dir in sorted(p for p in folder.iterdir() if p.is_dir()):
        for f in images_in(cls_dir):
            img = utils.read_image(f)
            C.append(m.classical_features(img))
            D.append(m.deep_features(img))
            y.append(cls_dir.name)
    C, D = np.array(C), np.array(D)
    Cn = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-9)
    sets = {"Classical": C, "Deep (CNN features)": D, "Classical + Deep": np.hstack([Cn, D])}
    idx_tr, idx_te = train_test_split(np.arange(len(y)), test_size=0.25, stratify=y, random_state=42)
    y = np.array(y)
    out = {"dataset": note, "images": len(y), "classes": sorted(set(y)), "test_images": len(idx_te), "results": {}}
    for name, X in sets.items():
        model = FeatureClassifier._make_model(list(y[idx_tr])).fit(X[idx_tr], y[idx_tr])
        out["results"][name] = round(float(accuracy_score(y[idx_te], model.predict(X[idx_te]))), 4)
    out["best"] = max(out["results"], key=out["results"].get)
    for name, X in sets.items():                        # the app's models: each feature set, ALL images
        m._feature_set = next(k for k in m.FEATURE_SETS if k.split(" (")[0] == name.split(" (")[0])
        clf = FeatureClassifier(m.features)
        clf._model, clf.labels = FeatureClassifier._make_model(list(y)).fit(X, y), sorted(set(y))
        clf.save(m.model_path)
    return out


def eval_backbones() -> dict:
    """MobileNetV3 vs ConvNeXt-Tiny features + the same SVM on TrashNet (one shared 75/25 split)."""
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
    from core.deep import CNNEmbedder
    from core.recognition import FeatureClassifier

    folder = ROOT / "datasets" / "trashnet"
    files, y = [], []
    for cls_dir in sorted(p for p in folder.iterdir() if p.is_dir()):
        for f in images_in(cls_dir):
            files.append(f)
            y.append(cls_dir.name)
    y = np.array(y)
    idx_tr, idx_te = train_test_split(np.arange(len(y)), test_size=0.25, stratify=y, random_state=42)
    out = {"dataset": "TrashNet, 6 classes, shared 75/25 split", "images": len(y), "results": {}}
    for backbone in ("mobilenet_v3_large", "convnext_tiny"):
        emb = CNNEmbedder.shared(backbone)
        start = time.time()
        X = np.array([emb.embed(utils.resize_max(utils.read_image(f), 384)[0]) for f in files])
        model = FeatureClassifier._make_model(list(y[idx_tr])).fit(X[idx_tr], y[idx_tr])
        out["results"][backbone] = {"test_accuracy": round(float(accuracy_score(y[idx_te], model.predict(X[idx_te]))), 4),
                                    "ms_per_image": round(1000 * (time.time() - start) / len(files), 1)}
    return out


def eval_flowers() -> dict:
    from topics.t03_flower_recognition import FlowerRecognizer
    return compare_features(FlowerRecognizer, REAL / "tf_flowers", "TF-Flowers (Google), 5 classes, 75/25 split")


def eval_leaves() -> dict:
    from topics.t21_leaf_recognition import LeafRecognizer
    return compare_features(LeafRecognizer, REAL / "medicinal_leaves", "Medicinal leaves (6 species incl. Neem), 75/25 split")


def eval_waste(**params) -> dict:
    from topics.t14_waste_segregation import WasteSegregation
    return train_eval(WasteSegregation, str(ROOT / "datasets" / "trashnet"), "TrashNet, 6 classes, 75/25 split", **params)


def eval_fish(**params) -> dict:
    from topics.t02_fish_recognition import FishRecognizer
    return train_eval(FishRecognizer, str(ROOT / "datasets" / "fish"), "phylo-fish, 8 species, 75/25 split",
                      method="Trained model (CNN + SVM)", enhance=False, **params)


def eval_leaf_disease() -> dict:
    from topics.t20_leaf_disease import LeafDiseaseDetector
    r = LeafDiseaseDetector().run(np.zeros((64, 64, 3), np.uint8), mode="Evaluate on folder",
                                  dataset=str(ROOT / "datasets" / "plantvillage_eval"), advice="Off")
    return {"dataset": "PlantVillage (38 classes)", "images": r.info["Images"], "accuracy": r.info["Accuracy"]}


def eval_plates() -> dict:
    """Topic 5: is the best box of each plate finder on the plate (IoU >= 0.5)? 200 unseen test photos."""
    from topics.t05_number_plate_recognition import YOLO_WEIGHTS, NumberPlateReader
    from core.yolo import YoloDetector
    ann = json.loads((REAL / "plates_test" / "annotations.json").read_text())
    yolo = YoloDetector.get(YOLO_WEIGHTS)
    out = {"dataset": "keremberke license-plate test (various countries, never used for training)", "images": len(ann)}
    finders = {"classical": lambda img: NumberPlateReader().find_plates(img)}
    if yolo is not None:
        finders["YOLOv8 (trained)"] = lambda img: [b[:4] for b in yolo.detect(img, conf=0.25, imgsz=480)]
    for name, find in finders.items():
        hits = top3 = 0
        for a in ann:
            cands = find(utils.read_image(REAL / "plates_test" / a["file"]))
            best = [max(iou(c, t) for t in a["boxes"]) for c in cands[:3]]
            hits += bool(best) and best[0] >= 0.5
            top3 += any(b >= 0.5 for b in best)
        out[name] = {"top-1 plate found (IoU>=0.5)": round(hits / len(ann), 4), "within top-3": round(top3 / len(ann), 4)}
    return out


def _yolo_split_annotations(name: str) -> list[tuple[Path, list]]:
    """The held-out images train_detectors.py used for validation, with pixel boxes."""
    folder = ROOT / "datasets" / "yolo" / name / "val"
    items = []
    for img_path in images_in(folder / "images"):
        h, w = cv2.imread(str(img_path)).shape[:2]
        boxes = []
        for line in (folder / "labels" / f"{img_path.stem}.txt").read_text().split("\n"):
            if line.strip():
                _, cx, cy, bw, bh = map(float, line.split())
                boxes.append(((cx - bw / 2) * w, (cy - bh / 2) * h, bw * w, bh * h))
        items.append((img_path, boxes))
    return items


def eval_potholes() -> dict:
    """Topic 13: classical vs YOLO pothole boxes (IoU >= 0.5) on the 25 photos the YOLO never trained on."""
    from topics.t13_pothole_crack_detection import PotholeCrackDetector
    items = _yolo_split_annotations("potholes")
    out = {"dataset": "keremberke pothole-segmentation: 25 held-out road photos", "images": len(items)}
    for detector in ("Classical (darkness)", "YOLOv8 (trained)"):
        tp = fp = fn = 0
        for img_path, truth in items:
            r = PotholeCrackDetector().run(utils.read_image(img_path), detect="Potholes only", detector=detector)
            if detector.startswith("YOLO") and "Note" in r.info:   # no trained weights yet
                break
            pred = [row["Box"] for row in r.info.get("Pothole details", [])]
            t, f_, n_ = match_boxes(pred, truth, 0.5)
            tp, fp, fn = tp + t, fp + f_, fn + n_
        else:
            prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
            out[detector] = {"precision": round(prec, 4), "recall": round(rec, 4),
                             "f1": round(2 * prec * rec / max(prec + rec, 1e-9), 4)}
    return out


def eval_trees() -> dict:
    """Topic 1: tree count vs annotated crowns on olive orchard drone images."""
    from topics.t01_tree_counting import TreeCounter
    m = TreeCounter()
    rows = []
    for a in json.loads((REAL / "olive_trees" / "annotations.json").read_text()):
        img = utils.read_image(REAL / "olive_trees" / a["file"])
        pred = m.run(img, method="Watershed").info["Trees"]
        rows.append((len(a["boxes"]), pred))
    truth, pred = np.array(rows).T
    return {"dataset": "AgML olive tree crowns (drone)", "images": len(rows),
            "MAE": round(float(np.abs(truth - pred).mean()), 2),
            "mean_abs_percent_error": round(float((np.abs(truth - pred) / np.maximum(truth, 1)).mean()), 4),
            "count_accuracy": round(float(1 - (np.abs(truth - pred).sum() / truth.sum())), 4),
            "pairs (truth, predicted)": rows}


def eval_blood() -> dict:
    """Topic 23: RBC / WBC / platelet counts vs BCCD annotations."""
    import xml.etree.ElementTree as ET
    from topics.t23_blood_cell_counter import BloodCellCounter
    m = BloodCellCounter()
    errs = {"RBC": [], "WBC": [], "Platelets": []}
    truths = {"RBC": [], "WBC": [], "Platelets": []}
    wbc_exact = n = 0
    for xml in sorted((REAL / "bccd" / "Annotations").glob("*.xml")):
        img_path = REAL / "bccd" / "JPEGImages" / (xml.stem + ".jpg")
        if not img_path.exists():
            continue
        names = [o.findtext("name") for o in ET.parse(xml).getroot().iter("object")]
        truth = {"RBC": names.count("RBC"), "WBC": names.count("WBC"), "Platelets": names.count("Platelets")}
        info = m.run(utils.read_image(img_path)).info
        for k in errs:
            errs[k].append(abs(info[k] - truth[k]))
            truths[k].append(truth[k])
        wbc_exact += info["WBC"] == truth["WBC"]
        n += 1
    return {"dataset": "BCCD (364 smear images, VOC boxes)", "images": n,
            **{f"MAE {k}": round(float(np.mean(v)), 2) for k, v in errs.items()},
            **{f"count accuracy {k}": round(1 - float(np.sum(errs[k]) / max(np.sum(truths[k]), 1)), 4) for k in errs},
            "WBC exact": round(wbc_exact / max(n, 1), 4),
            "note": "BCCD's RBC boxes are incomplete (many cells unlabelled), so the RBC error is pessimistic."}


def _edit_distance(a: str, b: str) -> int:
    row = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, row[0] = row[0], i
        for j, cb in enumerate(b, 1):
            prev, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, prev + (ca != cb))
    return row[-1]


def eval_plate_ocr() -> dict:
    """Topic 5 end to end: photo in -> plate text out, vs the labelled text (any plate in the photo)."""
    from topics.t05_number_plate_recognition import NumberPlateReader
    m = NumberPlateReader()
    norm = lambda t: re.sub(r"[^A-Z0-9]", "", t.upper())
    out = {"dataset": "sonnetechnology license-plate-text-recognition (test split, plate-like labels only)",
           "note": "Mostly Vietnamese and Indian plates; some labels are themselves wrong (e.g. MH 20.80 20 for MH 20 BQ 20)."}
    for locator in ("YOLOv8 (trained)", "Classical morphology"):
        exact = found = 0
        char_acc = []
        items = json.loads((REAL / "plates_ocr" / "annotations.json").read_text())
        for a in items:
            try:
                r = m.run(utils.read_image(REAL / "plates_ocr" / a["file"]), locator=locator, fix_format=False)
                pred = norm(r.info["Plate number"])
                found += 1
            except ValueError:
                pred = ""
            truths = [norm(t) for t in a["plates"]]
            exact += pred in truths
            char_acc.append(max(1 - _edit_distance(pred, t) / len(t) for t in truths))
        out[locator] = {"images": len(items), "plate read": round(found / len(items), 4),
                        "exact match": round(exact / len(items), 4),
                        "character accuracy": round(float(np.mean(np.maximum(char_acc, 0))), 4)}
    return out


def _fingerprint_scores(folder: Path) -> tuple[np.ndarray, np.ndarray]:
    """FVC protocol: every impression pair of the same finger (genuine) and the first impressions of
    different fingers (impostor). Returns (genuine scores, impostor scores)."""
    from itertools import combinations
    from topics.t06_fingerprint_matching import FingerprintMatcher
    m = FingerprintMatcher()
    prints = {}
    for f in sorted(folder.glob("*.tif")):
        finger, impression = (int(v) for v in f.stem.split("_"))
        prints[(finger, impression)] = m.analyse(utils.read_image(f), 9, show=False).minutiae
    fingers = sorted({k[0] for k in prints})
    score = lambda a, b: m.match(prints[a], prints[b])[0]
    genuine = [score((f, i), (f, j)) for f in fingers for i, j in combinations(range(1, 9), 2)]
    impostor = [score((a, 1), (b, 1)) for a, b in combinations(fingers, 2)]
    return np.array(genuine), np.array(impostor)


def eval_fingerprint() -> dict:
    """Topic 6 on FVC2002: the match threshold is chosen on DB2_B (security first: just above its highest
    impostor score, + 1 % margin = no false matches there) and tested on DB1_B."""
    from sklearn.metrics import roc_auc_score
    fnmr = lambda g, t: float((g < t).mean())
    fmr = lambda i, t: float((i >= t).mean())
    eer_point = lambda g, i: min(np.linspace(0, 0.5, 201), key=lambda t: abs(fnmr(g, t) - fmr(i, t)))
    g2, i2 = _fingerprint_scores(REAL / "fvc2002_db2b")
    t = float(np.ceil(i2.max() * 100) + 1) / 100
    g1, i1 = _fingerprint_scores(REAL / "fvc2002_db1b")
    t1 = eer_point(g1, i1)
    scores, labels = np.concatenate([g1, i1]), np.array([1] * len(g1) + [0] * len(i1))
    return {"dataset": "FVC2002 DB1_B (test) - threshold chosen on DB2_B (other sensor)",
            "genuine pairs": len(g1), "impostor pairs": len(i1),
            "threshold (from DB2)": round(t, 3),
            "accuracy @ threshold": round(float(((scores >= t) == labels).mean()), 4),
            "FNMR @ threshold": round(fnmr(g1, t), 4), "FMR @ threshold": round(fmr(i1, t), 4),
            "EER (DB1)": round((fnmr(g1, t1) + fmr(i1, t1)) / 2, 4),
            "ROC AUC (DB1)": round(float(roc_auc_score(labels, scores)), 4),
            "EER (DB2)": round((fnmr(g2, eer_point(g2, i2)) + fmr(i2, eer_point(g2, i2))) / 2, 4),
            "mean genuine / impostor score (DB1)": f"{g1.mean():.3f} / {i1.mean():.3f}"}


def eval_background() -> dict:
    """Topic 17 on DUTS-TE (salient objects with hand-drawn masks): IoU, F-measure and MAE of the alpha matte."""
    from topics.t17_background_remover import DEEP_METHOD, GRABCUT_METHODS, BackgroundRemover
    m = BackgroundRemover()
    folder = REAL / "duts_test"
    out = {"dataset": "DUTS-TE (100 random test images, salient-object masks; U2-Net was trained on DUTS-TR)"}
    for method in (DEEP_METHOD, *GRABCUT_METHODS):
        ious, fs, maes = [], [], []
        for f in images_in(folder / "images"):
            img = utils.read_image(f)
            truth = cv2.imread(str(folder / "masks" / (f.stem + ".png")), 0)
            try:
                alpha = m.run(img, method=method, background="Transparent").output[:, :, 3]
            except ValueError:
                alpha = np.zeros(img.shape[:2], np.uint8)
            alpha = cv2.resize(alpha, (truth.shape[1], truth.shape[0])).astype(np.float32) / 255
            gt = truth > 127
            pred = alpha > 0.5
            ious.append((pred & gt).sum() / max((pred | gt).sum(), 1))
            prec, rec = (pred & gt).sum() / max(pred.sum(), 1), (pred & gt).sum() / max(gt.sum(), 1)
            fs.append(1.3 * prec * rec / max(0.3 * prec + rec, 1e-9))   # F-beta, beta^2 = 0.3 (saliency papers)
            maes.append(float(np.abs(alpha - truth / 255).mean()))
        out[method] = {"mean IoU": round(float(np.mean(ious)), 4), "F-measure": round(float(np.mean(fs)), 4),
                       "MAE": round(float(np.mean(maes)), 4), "IoU >= 0.8": f"{np.mean(np.array(ious) >= 0.8):.0%}"}
    return out


def eval_fruit() -> dict:
    """Topic 22 on real banana / mango photos: colour rules (no training) vs the SVM trained on a
    separate train part (saved as the app's model)."""
    from topics.t22_fruit_ripeness import FruitRipenessDetector
    m = FruitRipenessDetector()
    report = m.train(REAL / "fruit_train")
    out = {"dataset": "darthraider fruit-ripeness (raw / ripe banana and mango), 600 train + 200 test photos",
           "SVM accuracy on a 25 % split of the train part": round(float(report.get("test_accuracy", 0)), 4)}
    for method in ("Colour rules", "Trained SVM model"):
        correct = total = 0
        per = Counter()
        for cls_dir in sorted((REAL / "fruit_test").iterdir()):
            state, fruit = cls_dir.name.split("_")
            for f in images_in(cls_dir):
                label = m.run(utils.read_image(f), fruit=fruit.title(), method=method).info["Ripeness"]
                hit = label.split("_")[0].lower() == state
                correct += hit
                total += 1
                per[cls_dir.name] += hit
        out[method] = {"accuracy": round(correct / total, 4), **{k: round(v / 50, 2) for k, v in sorted(per.items())}}
    return out


def eval_copymove() -> dict:
    """Topic 18: 30 real Flickr photos, each forged by copying a textured 64-96 px block elsewhere,
    saved as JPEG 90; the untouched photos (also JPEG 90) measure false alarms."""
    from topics.t18_copy_move_forgery import CopyMoveDetector
    m = CopyMoveDetector()
    rng = np.random.default_rng(0)
    jpeg = lambda img: cv2.imdecode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])[1], cv2.IMREAD_COLOR)
    pairs = []
    for f in images_in(REAL / "flickr8k_test")[40:70]:
        img = utils.resize_max(utils.read_image(f), 640)[0]
        h, w = img.shape[:2]
        for _ in range(200):                                             # a textured source, far enough away
            b = int(rng.integers(64, 97))
            y0, x0, y1, x1 = rng.integers(0, h - b), rng.integers(0, w - b), rng.integers(0, h - b), rng.integers(0, w - b)
            if img[y0:y0 + b, x0:x0 + b].std() > 25 and np.hypot(y1 - y0, x1 - x0) > 1.5 * b:
                break
        forged = img.copy()
        forged[y1:y1 + b, x1:x1 + b] = img[y0:y0 + b, x0:x0 + b]
        pairs.append((jpeg(img), jpeg(forged)))
    out = {"dataset": "30 Flickr8k photos, simulated copy-move (textured 64-96 px block) + JPEG 90", "images": len(pairs)}
    for method in ("Block matching (DCT)", "Keypoints (SIFT)"):
        hits = sum(m.run(fg, method=method).info["Verdict"] == "FORGED" for _, fg in pairs)
        alarms = sum(m.run(orig, method=method).info["Verdict"] == "FORGED" for orig, _ in pairs)
        out[method] = {"detection rate": round(hits / len(pairs), 4), "false alarm rate": round(alarms / len(pairs), 4),
                       "accuracy": round((hits + len(pairs) - alarms) / (2 * len(pairs)), 4)}
    return out


def eval_plate_ocr_gemini(n: int = 60) -> dict:
    """Topic 5 with the cloud reader: YOLO finds the plate, Gemini reads it (first 60 test photos, API quota)."""
    from core import cloud
    from topics.t05_number_plate_recognition import NumberPlateReader
    if not cloud.has_key("GEMINI_API_KEY"):
        raise RuntimeError("GEMINI_API_KEY missing in .env")
    m = NumberPlateReader()
    norm = lambda t: re.sub(r"[^A-Z0-9]", "", t.upper())
    items = json.loads((REAL / "plates_ocr" / "annotations.json").read_text())[:n]
    exact, char_acc, failed = 0, [], 0
    for a in items:
        try:
            pred = norm(m.run(utils.read_image(REAL / "plates_ocr" / a["file"]), ocr="Gemini (cloud)",
                              fix_format=False).info["Plate number"])
        except (ValueError, RuntimeError):
            pred, failed = "", failed + 1
        truths = [norm(t) for t in a["plates"]]
        exact += pred in truths
        char_acc.append(max(0.0, max(1 - _edit_distance(pred, t) / len(t) for t in truths)))
    answered = max(len(items) - failed, 1)
    return {"dataset": f"first {len(items)} photos of the plate-text test set", "reader": "YOLOv8 + Gemini (cloud)",
            "exact match": round(exact / len(items), 4), "character accuracy": round(float(np.mean(char_acc)), 4),
            "no answer / error": failed, "exact match (answered only)": round(exact / answered, 4),
            "character accuracy (answered only)": round(float(np.sum(char_acc)) / answered, 4),
            "note": "Failed calls = free-tier quota / server errors, counted as wrong in the first two numbers."}


def eval_cropweed() -> dict:
    """Topic 24 on CWFID test images (never used for training): vegetation mask IoU per index, and
    crop / weed accuracy over the annotated plant pixels for each rule."""
    from topics.t24_crop_weed_detection import CropWeedDetector
    folder = REAL / "cwfid"
    split = (folder / "train_test_split.yaml").read_text()
    ids = {k: [int(v) for v in re.findall(r"\d+", re.search(k + r":\s*\[([^\]]*)\]", split).group(1))] for k in ("train", "test")}
    test = [i for i in ids["test"] if i not in ids["train"]]           # the official split lists image 28 in both
    m = CropWeedDetector()
    iou = {k: [] for k in m.INDICES}
    px = {s: np.zeros(4) for s in m.STRATEGIES}                        # crop hit, crop total, weed hit, weed total
    failed = Counter()
    for i in test:
        img, _ = utils.resize_max(utils.read_image(folder / "images" / f"{i:03d}_image.png"), m.WORK_SIZE)
        size = (img.shape[1], img.shape[0])
        veg = cv2.resize(cv2.imread(str(folder / "masks" / f"{i:03d}_mask.png"), 0), size, interpolation=cv2.INTER_NEAREST) == 0
        ann = cv2.resize(utils.read_image(folder / "annotations" / f"{i:03d}_annotation.png"), size, interpolation=cv2.INTER_NEAREST)
        crop_gt, weed_gt = (ann[:, :, 1] > 0) & veg, (ann[:, :, 2] > 0) & veg
        for k in m.INDICES:
            pred = m.vegetation_mask(img, k)[1] > 0
            iou[k].append((pred & veg).sum() / max((pred | veg).sum(), 1))
        for s in m.STRATEGIES:
            try:
                labels, _, plants, is_crop, _ = m.classify_plants(img, m.INDICES[0], s)
            except ValueError:                                           # e.g. no rows in a one-row close-up
                failed[s] += 1
                px[s] += [0, crop_gt.sum(), 0, weed_gt.sum()]
                continue
            crop_pred = np.isin(labels, plants[is_crop[plants]])
            weed_pred = np.isin(labels, plants[~is_crop[plants]])
            px[s] += [(crop_pred & crop_gt).sum(), crop_gt.sum(), (weed_pred & weed_gt).sum(), weed_gt.sum()]
    rules = {}
    for s, (ch, ct, wh, wt) in px.items():
        rules[s] = {"pixel accuracy": round(float((ch + wh) / (ct + wt)), 4), "crop recall": round(float(ch / ct), 4),
                    "weed recall": round(float(wh / wt), 4), "balanced accuracy": round(float((ch / ct + wh / wt) / 2), 4),
                    "images where the rule failed": failed[s]}
    return {"dataset": "CWFID (Haug & Ostermann 2014), carrot field, official test split", "test_images": len(test),
            "vegetation IoU": {k: round(float(np.mean(v)), 4) for k, v in iou.items()}, "crop / weed rules": rules,
            "note": "Trained rule = SVM on 11 plant shape / colour features, trained on the 40 CWFID train images. "
                    "Crop rows needs 2+ rows; CWFID close-ups show one carrot row."}


def _bleu(candidates: list[str], references: list[list[str]], n: int = 4) -> dict:
    """Corpus BLEU-1..4 (Papineni et al. 2002) with brevity penalty."""
    tok = lambda s: re.findall(r"[a-z0-9]+", s.lower())
    out = {}
    for order in range(1, n + 1):
        logs, c_len, r_len = [], 0, 0
        for k in range(1, order + 1):
            match = total = 0
            for cand, refs in zip(candidates, references):
                c = tok(cand)
                grams = Counter(tuple(c[i:i + k]) for i in range(len(c) - k + 1))
                max_ref = Counter()
                for ref in refs:
                    r = tok(ref)
                    for g, v in Counter(tuple(r[i:i + k]) for i in range(len(r) - k + 1)).items():
                        max_ref[g] = max(max_ref[g], v)
                match += sum(min(v, max_ref[g]) for g, v in grams.items())
                total += max(len(c) - k + 1, 0)
            logs.append(np.log(max(match, 1e-9) / max(total, 1)))
        for cand, refs in zip(candidates, references):
            c = len(tok(cand))
            c_len += c
            r_len += min((len(tok(r)) for r in refs), key=lambda L: (abs(L - c), L))
        bp = 1.0 if c_len > r_len else np.exp(1 - r_len / max(c_len, 1))
        out[f"BLEU-{order}"] = round(float(bp * np.exp(np.mean(logs))), 4)
    return out


def eval_caption() -> dict:
    """Topic 9: BLIP captions vs 5 human captions each (Flickr8k test, 100 images)."""
    from topics.t09_image_captioning import ImageCaptioner
    m = ImageCaptioner()
    data = json.loads((REAL / "flickr8k_test" / "captions.json").read_text())
    cands = [m.caption_blip(utils.read_image(REAL / "flickr8k_test" / d["file"]), detailed=False) for d in data]
    return {"dataset": "Flickr8k test (100 images x 5 human captions)", "images": len(data),
            **_bleu(cands, [d["captions"] for d in data]),
            "examples": [{"model": c, "human": d["captions"][0]} for c, d in list(zip(cands, data))[:5]]}


def eval_sr() -> dict:
    """Topic 10: SRCNN vs bicubic PSNR (Y channel) on Set5 / Set14 / BSD100, x2 / x3 / x4."""
    from topics.t10_super_resolution import SuperResolution, shrink
    m = SuperResolution()
    files = images_in(REAL / "sr_benchmarks")
    out = {"dataset": "Set5 + Set14 + BSD100 (HR images, anti-aliased bicubic downscale like MATLAB)", "results": []}
    for bench in ("Set5", "Set14", "BSD100"):
        group = [f for f in files if bench.lower() in f.name.lower()]
        for s in (2, 3, 4):
            gains, bic = [], []
            for f in group:
                hr = utils.read_image(f)
                h, w = (hr.shape[0] // s) * s, (hr.shape[1] // s) * s
                hr = hr[:h, :w]
                lr = shrink(hr, s)
                b = m._psnr_y(cv2.resize(lr, (w, h), interpolation=cv2.INTER_CUBIC), hr)
                bic.append(b)
                gains.append(m._psnr_y(m.super_resolve(lr, s), hr) - b)
            if group:
                out["results"].append({"benchmark": bench, "scale": s, "images": len(group),
                                       "bicubic PSNR": round(float(np.mean(bic)), 2),
                                       "SRCNN PSNR": round(float(np.mean(bic) + np.mean(gains)), 2),
                                       "gain dB": round(float(np.mean(gains)), 3)})
    return out


def _split_eval_cnn(folder: Path) -> tuple[float, object, list, list]:
    """Train CNN+SVM on 75 %, return (test accuracy, classifier trained on ALL images, test files, test labels)."""
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
    from core.recognition import FeatureClassifier
    from topics.t08_mask_helmet_detection import _cnn

    files, y = [], []
    for cls_dir in sorted(p for p in folder.iterdir() if p.is_dir()):
        for f in images_in(cls_dir):
            files.append(f)
            y.append(cls_dir.name)
    X = np.array([_cnn(utils.read_image(f)) for f in files])
    y = np.array(y)
    tr, te = train_test_split(np.arange(len(y)), test_size=0.25, stratify=y, random_state=42)
    acc = accuracy_score(y[te], FeatureClassifier._make_model(list(y[tr])).fit(X[tr], y[tr]).predict(X[te]))
    clf = FeatureClassifier(_cnn)
    clf._model, clf.labels = FeatureClassifier._make_model(list(y)).fit(X, y), sorted(set(y))
    return float(acc), clf, [files[i] for i in te], list(y[te])


def eval_mask() -> dict:
    """Topic 8 mask: trained CNN vs classical skin rule on real face photos (Kaggle 12k face-mask set)."""
    from topics.t08_mask_helmet_detection import MASK_MODEL, MASK_YES, MaskHelmetDetector
    from core.faces import FaceDetector
    acc, clf, test_files, test_y = _split_eval_cnn(REAL / "face_mask")
    clf.save(MASK_MODEL)
    m, det = MaskHelmetDetector(), FaceDetector.shared()
    found = correct = 0
    for f, label in zip(test_files, test_y):
        img = utils.read_image(f)
        img = cv2.resize(img, None, fx=2, fy=2) if max(img.shape[:2]) < 160 else img    # tiny crops
        faces = det.detect(img, 0.5)
        if faces:
            found += 1
            mask, _, _ = m.classical(img, max(faces, key=lambda fc: fc.box[2] * fc.box[3]), show=False)
            correct += mask == (label.lower() in MASK_YES)
    n = len(test_files)
    return {"dataset": "Face-Mask-Detection (DamarJati / Kaggle 12k), 400 + 400 photos, 75/25 split",
            "test_images": n, "trained CNN accuracy": round(acc, 4),
            "classical: face found": round(found / n, 4),
            "classical accuracy (when a face is found)": round(correct / max(found, 1), 4),
            "classical accuracy (all, no face = wrong)": round(correct / n, 4)}


def eval_helmet() -> dict:
    """Topic 8 helmet: CNN+SVM on head crops cut from the hard-hat detection test photos."""
    from topics.t08_mask_helmet_detection import HELMET_MODEL
    src, crops = REAL / "hardhat_test", REAL / "helmet_crops"
    names = ["hardhat", "no-hardhat"]
    if not crops.exists():
        for a in json.loads((src / "annotations.json").read_text()):
            img = utils.read_image(src / a["file"])
            for k, ((x, y, w, h), cat) in enumerate(zip(a["boxes"], a["category"])):
                if w < 20 or h < 20:
                    continue
                x0, y0 = int(max(0, x - 0.3 * w)), int(max(0, y - 0.3 * h))
                crop = img[y0:int(y + 1.3 * h), x0:int(x + 1.3 * w)]
                utils.write_image(crops / names[cat] / f"{Path(a['file']).stem}_{k}.jpg", crop)
    acc, clf, test_files, _ = _split_eval_cnn(crops)
    clf.save(HELMET_MODEL)
    counts = {n: len(images_in(crops / n)) for n in names if (crops / n).exists()}
    return {"dataset": "keremberke hard-hat detection (test photos -> head crops >= 20 px)", "crops": counts,
            "test_images": len(test_files), "trained CNN accuracy": round(acc, 4)}


def eval_lfw() -> dict:
    """Topic 4: SFace verification on the LFW test pairs (500 same-person + 500 different-person pairs)."""
    from sklearn.datasets import fetch_lfw_pairs
    from sklearn.metrics import roc_auc_score
    from topics.t04_face_attendance import FaceAttendance
    from core.faces import FaceDetector
    pairs = fetch_lfw_pairs(subset="test", color=True, resize=1.0, slice_=None, data_home=str(REAL / "sklearn"))
    m, det = FaceAttendance(), FaceDetector.shared()
    scores, labels, missed = [], [], 0
    for (a, b), same in zip(pairs.pairs, pairs.target):
        vecs = []
        for face_img in (a, b):
            img = cv2.cvtColor((face_img * (255 if face_img.max() <= 1 else 1)).astype(np.uint8), cv2.COLOR_RGB2BGR)
            faces = det.detect(img, 0.5)
            if not faces:
                break
            centre = min(faces, key=lambda f: abs(f.box[0] + f.box[2] / 2 - img.shape[1] / 2))   # LFW: face is central
            vecs.append(m.embed(img, centre))
        if len(vecs) < 2:
            missed += 1
            scores.append(0.0)
        else:
            scores.append(float(vecs[0] @ vecs[1]))
        labels.append(int(same))
    scores, labels = np.array(scores), np.array(labels)
    acc_fixed = float(((scores >= 0.363) == labels).mean())
    best_t = max(np.linspace(0.1, 0.8, 71), key=lambda t: ((scores >= t) == labels).mean())
    return {"dataset": "LFW test pairs (sklearn), 1000 pairs", "pairs": len(labels), "no face found in a pair": missed,
            "accuracy @ 0.363 (SFace default)": round(acc_fixed, 4),
            "best threshold (on this set)": round(float(best_t), 2),
            "accuracy @ best threshold": round(float(((scores >= best_t) == labels).mean()), 4),
            "ROC AUC": round(float(roc_auc_score(labels, scores)), 4)}


def eval_watermark() -> dict:
    """Topic 19: robustness of the default watermark on 20 real Flickr photos (not the 20 photos the
    default strength was chosen on)."""
    from topics.t19_invisible_watermarking import InvisibleWatermark
    m = InvisibleWatermark()
    per_attack: dict[str, list] = {}
    psnrs = []
    for f in images_in(REAL / "flickr8k_test")[70:90]:
        r = m.run(utils.read_image(f), mode="Robustness Test")
        psnrs.append(float(r.info["PSNR"].split()[0]))
        for row in r.info["Attack results"]:
            per_attack.setdefault(row["Attack"], []).append(row["NC"])
    return {"dataset": "20 Flickr8k photos, default settings (LL band, strength 36)",
            "mean PSNR (dB)": round(float(np.mean(psnrs)), 2),
            "attacks": {k: {"mean NC": round(float(np.mean(v)), 3), "survived (NC >= 0.85)": f"{np.mean(np.array(v) >= 0.85):.0%}"}
                        for k, v in per_attack.items()}}


def eval_restoration() -> dict:
    """Topic 16: real photos aged with the same damage model as the sample -> restored -> SSIM / PSNR."""
    sys.path.insert(0, str(ROOT / "samples"))
    import generate_samples as gs
    from topics.t16_old_photo_restoration import OldPhotoRestorer
    m = OldPhotoRestorer()
    before, after = [], []
    for i, f in enumerate(images_in(REAL / "flickr8k_test")[20:40]):
        clean = utils.read_image(f)
        damaged = gs.age_photo(clean, seed=i)
        truth = m.fix_fading(cv2.cvtColor(clean, cv2.COLOR_BGR2GRAY))
        base = m.fix_fading(cv2.cvtColor(damaged, cv2.COLOR_BGR2GRAY))
        out = cv2.cvtColor(m.run(damaged).output, cv2.COLOR_BGR2GRAY)
        before.append((utils.ssim(base, truth), utils.psnr(base, truth)))
        after.append((utils.ssim(out, truth), utils.psnr(out, truth)))
    b, a = np.array(before).mean(axis=0), np.array(after).mean(axis=0)
    return {"dataset": "20 Flickr8k photos with simulated scratches, dust, grain, fading",
            "SSIM damaged": round(float(b[0]), 3), "SSIM restored": round(float(a[0]), 3),
            "PSNR damaged (dB)": round(float(b[1]), 2), "PSNR restored (dB)": round(float(a[1]), 2)}


EVALUATORS = {"backbones": eval_backbones, "mask": eval_mask, "helmet": eval_helmet, "lfw": eval_lfw,
              "watermark": eval_watermark, "restoration": eval_restoration, "fer": eval_emotion, "corel": eval_corel, "flowers": eval_flowers,
              "leaves": eval_leaves,
              "waste": eval_waste, "fish": eval_fish, "leafdisease": eval_leaf_disease, "plates": eval_plates,
              "potholes": eval_potholes, "trees": eval_trees, "blood": eval_blood, "caption": eval_caption,
              "sr": eval_sr, "cropweed": eval_cropweed, "plateocr": eval_plate_ocr, "plateocr_gemini": eval_plate_ocr_gemini, "fingerprint": eval_fingerprint,
              "background": eval_background, "fruit": eval_fruit, "copymove": eval_copymove}


def save_result(key: str, value: dict) -> None:
    """Re-read the file right before writing, so several evaluations running at once do not erase each other."""
    results = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    results[key] = value
    RESULTS.write_text(json.dumps(results, indent=1, default=str))


def main() -> None:
    for key in sys.argv[1:] or EVALUATORS:
        start = time.time()
        try:
            result = EVALUATORS[key]()
            result["evaluated_in_s"] = round(time.time() - start)
            save_result(key, result)
            summary = {k: v for k, v in result.items() if not isinstance(v, (list, dict))}
            print(key, json.dumps(summary), flush=True)
        except Exception as err:                            # noqa: BLE001 - keep going with the others
            import traceback
            print(key, "FAILED:", type(err).__name__, err, traceback.format_exc().splitlines()[-3], flush=True)


if __name__ == "__main__":
    main()
