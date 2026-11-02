"""
Downloads the REAL test datasets used by evaluation/evaluate_all.py into datasets/real/.
Only a part of each dataset is taken (enough for a fair accuracy number, not gigabytes).

    python evaluation/fetch_datasets.py            # everything
    python evaluation/fetch_datasets.py fer flowers # only some

Sources (all public): Hugging Face datasets-server, TensorFlow flower_photos, BCCD on GitHub.
"""
from __future__ import annotations

import ast
import base64
import collections
import io
import json
import random
import re
import sys
import tarfile
import time
import urllib.parse
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "datasets" / "real"
API = "https://datasets-server.huggingface.co"


def get(url: str, raw: bool = False, timeout: int = 120):
    req = urllib.request.Request(url, headers={"User-Agent": "dip-studio/1.0"})
    for attempt in range(7):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = r.read()
            time.sleep(0.3)                                 # be gentle: anonymous users are rate limited
            return data if raw else json.loads(data)
        except urllib.error.HTTPError as err:
            if err.code not in (429, 500, 502, 503, 504) or attempt == 6:
                raise
            time.sleep(15 * (attempt + 1))                  # rate limited / server busy: back off and retry
        except Exception:                                   # noqa: BLE001 - flaky network: retry
            if attempt == 6:
                raise
            time.sleep(3)


def rows(dataset: str, config: str, split: str, start: int = 0, stop: int | None = None,
         pages: int | None = None) -> list[dict]:
    """Rows start..stop, or `pages` random pages of 100 rows from the whole split (for splits sorted by label)."""
    q = f"{API}/rows?dataset={urllib.parse.quote(dataset)}&config={config}&split={split}"
    total = get(f"{q}&offset=0&length=1")["num_rows_total"]
    stop = min(total, stop or total)
    offsets = list(range(start, stop, 100))
    if pages:
        offsets = sorted(random.Random(0).sample(offsets, min(pages, len(offsets))))
    out = []
    for off in offsets:
        out += [r["row"] for r in get(f"{q}&offset={off}&length={min(100, stop - off)}")["rows"]]
    return out


def save_all(jobs: list[tuple[str, Path]]) -> None:
    """Download (url, path) pairs with 3 threads (more get rate limited)."""
    def one(job):
        url, path = job
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(get(url, raw=True))
    with ThreadPoolExecutor(3) as pool:
        list(pool.map(one, jobs))


def classification(dataset, config, split, folder, label_key="label", image_key="image",
                   per_class=None, names=None, seed=0, pages=None) -> None:
    data = rows(dataset, config, split, pages=pages)
    if names is None:
        info = get(f"{API}/info?dataset={urllib.parse.quote(dataset)}")["dataset_info"][config]["features"]
        names = info[label_key]["names"]
    random.Random(seed).shuffle(data)                       # per_class then takes a random subset
    by = collections.defaultdict(list)
    for r in data:
        by[int(r[label_key])].append(r)
    jobs = []
    for label, items in by.items():
        cls = names[label].replace(" ", "_").lower()
        for i, r in enumerate(items[:per_class] if per_class else items):
            path = OUT / folder / cls / f"{i:04d}.jpg"
            img = r[image_key]
            if isinstance(img, dict):
                jobs.append((img["src"], path))
            else:                                           # base64 bytes inside the row (FER)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(base64.b64decode(img))
    save_all(jobs)
    print(folder, {names[k].lower(): min(len(v), per_class or len(v)) for k, v in by.items()})


def detection(dataset, config, split, folder, limit, start=0, normalised=False) -> None:
    data = rows(dataset, config, split, start, start + limit)
    jobs, ann = [], []
    for i, r in enumerate(data):
        objects = r["objects"] if isinstance(r["objects"], dict) else ast.literal_eval(r["objects"])
        name = f"{i:04d}.jpg"
        jobs.append((r["image"]["src"], OUT / folder / name))
        ann.append({"file": name, "width": r["image"]["width"], "height": r["image"]["height"],
                    "boxes": objects["bbox"], "category": objects.get("category"), "normalised": normalised})
    save_all(jobs)
    (OUT / folder / "annotations.json").write_text(json.dumps(ann))
    print(folder, len(ann), "images")


# ----- the datasets -----------------------------------------------------------------------
def fer():
    classification("Jeneral/fer-2013", "default", "test", "fer2013_test", label_key="labels", image_key="img_bytes",
                   per_class=100, names=["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"])


def fer_val():
    """FER-2013 TRAIN faces, only to calibrate the FER+ scores (never used for the test number)."""
    classification("Jeneral/fer-2013", "default", "train", "fer2013_val", label_key="labels", image_key="img_bytes",
                   per_class=150, pages=60, names=["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"])


def leaves():
    classification("Project-AgML/medicinal_leaf_species_classification", "raw", "train", "medicinal_leaves")
    sample_photos("medicinal_leaves", "leaf_photo", ("neem_leaf", "curry_leaf", "mint_leaf", "arjun_leaf"))


def sample_photos(folder: str, prefix: str, classes: tuple[str, ...]) -> None:
    """Copy one real photo per class into samples/ so the app has real examples for the real-data models."""
    for cls in classes:
        files = sorted((OUT / folder / cls).glob("*.jpg"))
        if files:
            (ROOT / "samples" / f"{prefix}_{cls}.jpg").write_bytes(files[len(files) // 2].read_bytes())


def corel():
    classification("maxglo/corel1k", "default", "train", "corel1k")


def mask():
    classification("DamarJati/Face-Mask-Detection", "default", "train", "face_mask", per_class=400)


def cropweed():
    classification("Rishi210904/Crop_Weed_classification", "default", "test", "crop_weed_test", per_class=150)


def hardhat():
    detection("keremberke/hard-hat-detection", "full", "test", "hardhat_test", 300)


def plates():
    detection("keremberke/license-plate-object-detection", "full", "test", "plates_test", 200)


def plates_train():
    detection("keremberke/license-plate-object-detection", "full", "train", "plates_train", 400)


def plates_ocr(split: str = "test", folder_name: str = "plates_ocr", pages: int | None = None):
    """Car photos WITH the plate text (for end-to-end ANPR accuracy). The source mixes in receipts and
    signs, so only rows whose text looks like a plate (letters + digits, 5-10 characters) are kept."""
    folder, ann, jobs = OUT / folder_name, [], []
    for r in rows("sonnetechnology/license-plate-text-recognition-full", "default", split, pages=pages):
        texts = [t for t in r["target"] if re.fullmatch(r"[A-Z0-9][A-Z0-9 .\-]{3,12}", t)]
        texts = [t for t in texts if 5 <= len(re.sub(r"[^A-Z0-9]", "", t)) <= 10
                 and len(re.findall(r"\d", t)) >= 2 and re.search(r"[A-Z]", t)]
        if texts:
            name = f"{len(ann):04d}.jpg"
            jobs.append((r["image"]["src"], folder / name))
            ann.append({"file": name, "plates": texts})
    save_all(jobs)
    (folder / "annotations.json").write_text(json.dumps(ann))
    print(folder_name, len(ann), "images")


def plates_ocr_val():
    """The same source's VALIDATION split: only used to tune the OCR settings of topic 5."""
    plates_ocr("validation", "plates_ocr_val", pages=8)


def potholes():
    for split in ("train", "validation", "test"):
        detection("keremberke/pothole-segmentation", "full", split, f"potholes/{split}", 100)


def olive():
    detection("Project-AgML/olive_tree_crown_detection", "default", "train", "olive_trees", 12, normalised=True)


def flickr():
    data = rows("jxie/flickr8k", "default", "test", 0, 100)
    jobs, caps = [], []
    for i, r in enumerate(data):
        jobs.append((r["image"]["src"], OUT / "flickr8k_test" / f"{i:04d}.jpg"))
        caps.append({"file": f"{i:04d}.jpg", "captions": [r[f"caption_{k}"] for k in range(5)]})
    save_all(jobs)
    (OUT / "flickr8k_test" / "captions.json").write_text(json.dumps(caps, indent=1))
    print("flickr8k_test", len(caps))


def sr_benchmarks():
    """Set5, Set14, BSD100 zips from the repository; only the high-resolution (HR) images are kept."""
    target = OUT / "sr_benchmarks"
    for bench in ("Set5", "Set14", "BSD100"):
        data = get(f"https://huggingface.co/datasets/keanteng/bsd100-set5-set14/resolve/main/{bench}.zip", raw=True,
                   timeout=600)
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            for name in z.namelist():                       # e.g. Set5/image_SRF_2/img_001_SRF_2_HR.png
                if ".." not in name and name.endswith("_SRF_2_HR.png"):   # each HR image once
                    out = target / f"{bench}__{Path(name).name}"
                    out.parent.mkdir(parents=True, exist_ok=True)
                    out.write_bytes(z.read(name))
    print("sr_benchmarks", dict(collections.Counter(p.name.split("__")[0] for p in target.iterdir())))


def flowers():
    target = OUT / "tf_flowers"
    if target.exists():
        return print("tf_flowers already there")
    data = get("https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz", raw=True, timeout=600)
    with tarfile.open(fileobj=io.BytesIO(data)) as tar:
        for m in tar.getmembers():
            if m.isfile() and m.name.lower().endswith(".jpg") and ".." not in m.name:
                cls = Path(m.name).parent.name
                out = target / cls / Path(m.name).name
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_bytes(tar.extractfile(m).read())
    print("tf_flowers", {p.name: len(list(p.iterdir())) for p in target.iterdir()})
    sample_photos("tf_flowers", "flower_photo", ("daisy", "roses", "sunflowers", "tulips"))


def bccd():
    target = OUT / "bccd"
    if target.exists():
        return print("bccd already there")
    data = get("https://github.com/Shenggan/BCCD_Dataset/archive/refs/heads/master.zip", raw=True, timeout=600)
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        for name in z.namelist():
            if ".." in name or name.endswith("/"):
                continue
            if "/JPEGImages/" in name or "/Annotations/" in name:
                out = target / Path(name).parent.name / Path(name).name
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_bytes(z.read(name))
    print("bccd", {p.name: len(list(p.iterdir())) for p in target.iterdir()})


def bsds_train():
    """BSDS500 train + test photos to TRAIN the SRCNN (topic 10). Its 'val' part is BSD100, a test
    benchmark, so it is left out, and so is 302003 (the 'woman' image of Set5)."""
    target = OUT / "bsds_train"
    if target.exists():
        return print("bsds_train already there")
    data = get("https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz", raw=True, timeout=1800)
    target.mkdir(parents=True)
    with tarfile.open(fileobj=io.BytesIO(data)) as tar:
        for m in tar.getmembers():
            parts = Path(m.name).parts
            if m.isfile() and m.name.endswith(".jpg") and ".." not in m.name and "images" in parts \
                    and parts[-2] in ("train", "test") and parts[-1] != "302003.jpg":
                (target / f"{parts[-2]}_{parts[-1]}").write_bytes(tar.extractfile(m).read())
    print("bsds_train", len(list(target.iterdir())), "images")


def fvc2002():
    """FVC2002 DB1_B and DB2_B: 10 fingers x 8 impressions each (two optical sensors), the public part of
    the FVC2002 contest. DB2 is used to choose the match threshold, DB1 to test it."""
    for db in ("DB1_B", "DB2_B"):
        target = OUT / f"fvc2002_{db.lower().replace('_', '')}"
        if target.exists():
            print(target.name, "already there")
            continue
        data = get(f"http://bias.csr.unibo.it/fvc2002/Downloads/{db}.zip", raw=True, timeout=600)
        target.mkdir(parents=True)
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            for name in z.namelist():
                if name.lower().endswith(".tif") and ".." not in name:
                    (target / Path(name).name).write_bytes(z.read(name))
        print(target.name, len(list(target.iterdir())), "images")


def duts():
    """DUTS-TE salient-object images with hand-made masks (background remover, topic 17)."""
    folder = OUT / "duts_test"
    jobs = []
    for i, r in enumerate(rows("nobg/DUTS", "default", "test", pages=1)):
        jobs += [(r["image"]["src"], folder / "images" / f"{i:04d}.jpg"), (r["mask"]["src"], folder / "masks" / f"{i:04d}.png")]
    save_all(jobs)
    print("duts_test", len(jobs) // 2, "images")


def fruit():
    """Raw / ripe banana and mango photos (topic 22): a train part for the SVM and a separate test part."""
    classification("darthraider/fruit-ripeness-detection-dataset", "default", "train", "fruit_train", per_class=150)
    classification("darthraider/fruit-ripeness-detection-dataset", "default", "test", "fruit_test", per_class=50)
    sample_photos("fruit_test", "fruit_photo", ("raw_banana", "ripe_banana", "raw_mango", "ripe_mango"))


def cwfid():
    """Crop/Weed Field Image Dataset (Haug & Ostermann 2014): 60 top-view carrot field images with
    vegetation masks and crop / weed annotations. Non-commercial research use, cite the paper."""
    target = OUT / "cwfid"
    if target.exists():
        return print("cwfid already there")
    data = get("https://github.com/cwfid/dataset/archive/refs/heads/master.zip", raw=True, timeout=900)
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        for name in z.namelist():
            parts = Path(name).parts                        # dataset-master/<folder>/<file>
            if ".." in name or name.endswith("/") or len(parts) < 2:
                continue
            if parts[1] in ("images", "masks", "annotations", "train_test_split.yaml"):
                out = target / Path(*parts[1:])
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_bytes(z.read(name))
    print("cwfid", {p.name: len(list(p.iterdir())) for p in target.iterdir() if p.is_dir()})


ALL = {"fer": fer, "fer_val": fer_val, "leaves": leaves, "corel": corel, "mask": mask, "cropweed": cropweed, "cwfid": cwfid, "fvc2002": fvc2002, "duts": duts, "fruit": fruit, "bsds_train": bsds_train, "hardhat": hardhat, "plates": plates, "plates_ocr": plates_ocr, "plates_ocr_val": plates_ocr_val,
       "plates_train": plates_train,
       "potholes": potholes, "olive": olive, "flickr": flickr, "sr": sr_benchmarks, "flowers": flowers, "bccd": bccd}

if __name__ == "__main__":
    for key in sys.argv[1:] or ALL:
        try:
            ALL[key]()
        except Exception as err:                            # noqa: BLE001 - report and go on
            print(key, "FAILED:", type(err).__name__, str(err)[:200])
