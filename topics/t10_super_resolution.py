"""
Topic 10 - Image Super Resolution with SRCNN (Dong et al., 2014).

SRCNN improves a bicubic-upscaled image with 3 convolution layers on the luminance (Y) channel:
    9x9 conv, 64 filters  -> patch extraction ("what local pattern is here?")
    1x1 conv, 32 filters  -> non-linear mapping (low-res pattern -> high-res pattern)
    5x5 conv, 1 filter    -> reconstruction of the sharp pixel
One change from the paper: residual learning (from VDSR, Kim et al. 2016) - the layers only
predict the missing detail, which is added to the bicubic image. The original SRCNN needs ~10^8
training steps; with the residual, 30 epochs on a laptop CPU give +0.53 to +1.04 dB over bicubic on
Set5 / Set14 / BSD100 (x2 - x4, EVALUATION.md).
Colour (Cr, Cb) is only upscaled with bicubic - the eye is far less sensitive to colour detail.

Modes
  Upscale   : make a small image x2 / x3 / x4 bigger (bicubic, then SRCNN)
  Evaluate  : treat the input as the sharp original, shrink it, restore it with every method and
              report PSNR / SSIM against the original (nearest, bilinear, bicubic, Lanczos, SRCNN)

Low-resolution inputs are made like the SR benchmarks (Set5 / Set14 / BSD100): bicubic shrinking WITH
anti-aliasing (MATLAB imresize). OpenCV's INTER_CUBIC does not anti-alias when shrinking; a network
trained on one kind of input and tested on the other loses up to 1 dB.

One network serves x2, x3 and x4. At x2 it would add too much detail, so the predicted detail is
scaled per factor (output = bicubic + gain_s * detail); the gains are picked on held-out validation
photos (python -m topics.t10_super_resolution calibrate), never on the test benchmarks.

Train (one model for scales 2-4) on the 399 BSDS500 photos (python evaluation/fetch_datasets.py bsds_train),
or on any folder of photos:
    python -m topics.t10_super_resolution train [folder_with_photos]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from core import utils
from core.base import DIPModule, ModuleResult, Param
from core.registry import register

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEIGHTS = PROJECT_ROOT / "models" / "srcnn_x234.pth"
GAINS = PROJECT_ROOT / "models" / "srcnn_gain.json"     # how much of the predicted detail to add, per scale
TRAIN_PHOTOS = PROJECT_ROOT / "datasets" / "real" / "bsds_train"


def shrink(img: np.ndarray, scale: int) -> np.ndarray:
    """Bicubic downscale with anti-aliasing (Pillow, like MATLAB imresize) - how SR benchmarks are made."""
    from PIL import Image
    h, w = img.shape[:2]
    size = (w // scale, h // scale)
    if img.ndim == 2:
        return np.asarray(Image.fromarray(img).resize(size, Image.BICUBIC))
    small = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).resize(size, Image.BICUBIC)
    return cv2.cvtColor(np.asarray(small), cv2.COLOR_RGB2BGR)
INTERPOLATIONS = {"Nearest": cv2.INTER_NEAREST, "Bilinear": cv2.INTER_LINEAR,
                  "Bicubic": cv2.INTER_CUBIC, "Lanczos": cv2.INTER_LANCZOS4}


def build_srcnn():
    import torch.nn as nn

    class SRCNN(nn.Module):
        """SRCNN layers with a residual connection: output = bicubic input + predicted detail."""

        def __init__(self) -> None:
            super().__init__()
            self.body = nn.Sequential(
                nn.Conv2d(1, 64, 9, padding=4), nn.ReLU(inplace=True),
                nn.Conv2d(64, 32, 1), nn.ReLU(inplace=True),
                nn.Conv2d(32, 1, 5, padding=2),
            )

        def forward(self, x):
            return x + self.body(x)

    return SRCNN()


@register
class SuperResolution(DIPModule):
    topic_no = 10
    name = "Image Super Resolution (SRCNN)"
    description = ("Upscales images x2-x4 with a 3-layer CNN (SRCNN) trained in this project, and compares "
                   "it with nearest / bilinear / bicubic / Lanczos interpolation using PSNR and SSIM.")
    samples = ("face_astronaut.png", "flower_query_sunflower.jpg", "note_reference_genuine.png")

    _model = None

    def parameters(self) -> list[Param]:
        return [
            Param("mode", "Mode", "select", "Evaluate (shrink then restore)",
                  options=("Evaluate (shrink then restore)", "Upscale")),
            Param("scale", "Scale factor", "select", 2, options=(2, 3, 4)),
        ]

    # ----- pipeline -----------------------------------------------------------
    def process(self, image, *, mode, scale) -> ModuleResult:
        img = utils.ensure_bgr(image)
        if mode == "Upscale":
            img, _ = utils.resize_max(img, 700)                     # output stays manageable
            bicubic = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            sr = self.super_resolve(img, scale)
            self._add_step(f"Input ({img.shape[1]}x{img.shape[0]})", img)
            self._add_step(f"Bicubic x{scale}", bicubic)
            self._add_step(f"SRCNN x{scale}", sr)
            self._add_step("Zoomed comparison: bicubic (left) vs SRCNN (right)", self._zoom_pair(bicubic, sr))
            return ModuleResult(sr, info={"Input": f"{img.shape[1]} x {img.shape[0]}",
                                          "Output": f"{sr.shape[1]} x {sr.shape[0]}", "Scale": f"x{scale}"})

        # Evaluate: input = ground truth
        hr, _ = utils.resize_max(img, 900)
        h, w = (hr.shape[0] // scale) * scale, (hr.shape[1] // scale) * scale
        hr = hr[:h, :w]
        lr = shrink(hr, scale)
        self._add_step(f"Low-resolution input (1/{scale} size)", lr)
        rows, results = [], {}
        for name, flag in INTERPOLATIONS.items():
            results[name] = cv2.resize(lr, (w, h), interpolation=flag)
        results["SRCNN"] = self.super_resolve(lr, scale)
        for name, out in results.items():
            rows.append({"Method": name, "PSNR (dB)": round(self._psnr_y(out, hr), 2),
                         "SSIM": round(utils.ssim(out, hr), 4)})
        best = max(rows, key=lambda r: r["PSNR (dB)"])
        self._add_step("Zoomed: bicubic | SRCNN | original", self._zoom_triple(results["Bicubic"], results["SRCNN"], hr))
        gain = rows[-1]["PSNR (dB)"] - rows[2]["PSNR (dB)"]
        return ModuleResult(results["SRCNN"], info={
            "Best method": best["Method"],
            "SRCNN vs bicubic": f"{gain:+.2f} dB",
            "Scale": f"x{scale}",
            "Comparison": rows,
        })

    # ----- model ------------------------------------------------------------------
    @classmethod
    def _load(cls):
        if cls._model is None:
            import torch

            if not WEIGHTS.exists():
                raise FileNotFoundError("SRCNN is not trained yet: python -m topics.t10_super_resolution train")
            model = build_srcnn()
            model.load_state_dict(torch.load(WEIGHTS, map_location="cpu", weights_only=True))
            cls._model = model.eval()
        return cls._model

    def super_resolve(self, img: np.ndarray, scale: int, gain: float | None = None) -> np.ndarray:
        import torch

        if gain is None:
            gain = json.loads(GAINS.read_text()).get(str(scale), 1.0) if GAINS.exists() else 1.0
        up = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        ycc = cv2.cvtColor(up, cv2.COLOR_BGR2YCrCb)
        y = torch.from_numpy(ycc[:, :, 0].astype(np.float32) / 255)[None, None]
        with torch.no_grad():
            detail = self._load().body(y)                    # residual: the missing high frequencies
            y_sr = (y + gain * detail)[0, 0].clamp(0, 1).numpy()
        ycc[:, :, 0] = np.round(y_sr * 255).astype(np.uint8)
        return cv2.cvtColor(ycc, cv2.COLOR_YCrCb2BGR)

    @staticmethod
    def _psnr_y(a: np.ndarray, b: np.ndarray, border: int = 6) -> float:
        """PSNR on the Y channel without the border (the usual way SR papers report it)."""
        ya = cv2.cvtColor(a, cv2.COLOR_BGR2YCrCb)[border:-border, border:-border, 0]
        yb = cv2.cvtColor(b, cv2.COLOR_BGR2YCrCb)[border:-border, border:-border, 0]
        return utils.psnr(ya, yb)

    # ----- drawing ------------------------------------------------------------------
    @staticmethod
    def _crop_centre(img, size=120):
        h, w = img.shape[:2]
        s = min(size, h, w)
        return img[(h - s) // 2:(h + s) // 2, (w - s) // 2:(w + s) // 2]

    def _zoom_pair(self, a, b):
        return np.hstack([utils.enlarge(self._crop_centre(a), 3), utils.enlarge(self._crop_centre(b), 3)])

    def _zoom_triple(self, a, b, c):
        return np.hstack([utils.enlarge(self._crop_centre(x), 3) for x in (a, b, c)])


# ===== training ======================================================================
def _training_images(folder: Path | None) -> list[np.ndarray]:
    images = []
    if folder is None and TRAIN_PHOTOS.is_dir():
        folder = TRAIN_PHOTOS
    if folder is not None:
        images += [utils.read_image(p) for p in sorted(folder.rglob("*"))
                   if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
    else:
        from skimage import data
        for name in ("astronaut", "coffee", "chelsea", "rocket", "camera", "coins", "page", "text",
                     "immunohistochemistry", "hubble_deep_field", "retina", "colorwheel"):
            try:
                im = getattr(data, name)()
                images.append(cv2.cvtColor(im, cv2.COLOR_RGB2BGR) if im.ndim == 3 else cv2.cvtColor(im, cv2.COLOR_GRAY2BGR))
            except Exception:                                       # some need an internet download
                pass
        wallpapers = Path("C:/Windows/Web/Wallpaper")
        if wallpapers.exists():
            images += [utils.resize_max(utils.read_image(p), 1000)[0] for p in sorted(wallpapers.rglob("*.jpg"))]
    return images


def train(folder: Path | None = None, epochs: int = 30, patch: int = 33, per_image: int = 150) -> None:
    import torch

    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    images = _training_images(folder)
    if len(images) < 3:
        raise SystemExit("Need at least 3 training photos.")
    n_val = 10 if len(images) >= 50 else 2
    val_imgs, train_imgs = images[:n_val], images[n_val:]     # kept aside to pick the best epoch
    print(f"Training on {len(train_imgs)} images, validating on {len(val_imgs)}")

    def make_patches():
        xs, ys = [], []
        for img in train_imgs:
            y = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)[:, :, 0]
            for s in (2, 3, 4):                               # one model for all scales
                h, w = (y.shape[0] // s) * s, (y.shape[1] // s) * s
                hr = y[:h, :w]
                blurry = cv2.resize(shrink(hr, s), (w, h), interpolation=cv2.INTER_CUBIC)
                for _ in range(per_image // 3):
                    r, c = rng.integers(0, h - patch), rng.integers(0, w - patch)
                    xs.append(blurry[r:r + patch, c:c + patch])
                    ys.append(hr[r:r + patch, c:c + patch])
        return (torch.from_numpy(np.array(xs, np.float32) / 255)[:, None],
                torch.from_numpy(np.array(ys, np.float32) / 255)[:, None])

    model = build_srcnn()
    optim = torch.optim.Adam(model.parameters(), lr=5e-4)
    sched = torch.optim.lr_scheduler.StepLR(optim, step_size=10, gamma=0.5)
    loss_fn = torch.nn.MSELoss()
    module = SuperResolution()

    def validation_gain() -> float:
        """Mean PSNR gain of SRCNN over bicubic (Y channel, x2 / x3 / x4) on the held-out images."""
        gains = []
        for img in val_imgs:
            for s in (2, 3, 4):
                h, w = (img.shape[0] // s) * s, (img.shape[1] // s) * s
                hr = img[:h, :w]
                lr = shrink(hr, s)
                gains.append(module._psnr_y(module.super_resolve(lr, s, 1.0), hr)
                             - module._psnr_y(cv2.resize(lr, (w, h), interpolation=cv2.INTER_CUBIC), hr))
        return float(np.mean(gains))

    best_gain, best_state = -np.inf, None
    for epoch in range(1, epochs + 1):
        X, Y = make_patches()                                  # fresh random patches every epoch
        perm = torch.randperm(len(X))
        start, total = time.time(), 0.0
        model.train()
        for i in range(0, len(X), 128):
            idx = perm[i:i + 128]
            optim.zero_grad()
            loss = loss_fn(model(X[idx]), Y[idx])
            loss.backward()
            optim.step()
            total += loss.item() * len(idx)
        sched.step()
        model.eval()
        SuperResolution._model = model
        gain = validation_gain()
        if gain > best_gain:                                   # early stopping: keep the best epoch
            best_gain = gain
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            WEIGHTS.parent.mkdir(exist_ok=True)
            torch.save(best_state, WEIGHTS)                    # saved now, so an interrupted run keeps it
        print(f"epoch {epoch:2d}  loss {total / len(X):.5f}  val gain over bicubic (x2-4) {gain:+.3f} dB"
              f"{'  (best)' if gain == best_gain else ''}  ({time.time() - start:.0f}s)")
    print(f"Best epoch kept ({best_gain:+.3f} dB over bicubic) in {WEIGHTS}")
    SuperResolution._model = None
    calibrate(val_imgs)


def calibrate(val_imgs: list[np.ndarray] | None = None) -> dict[str, float]:
    """Pick the detail gain of every scale on the validation photos (the first 10 training photos)."""
    if val_imgs is None:
        images = _training_images(None)
        val_imgs = images[:10 if len(images) >= 50 else 2]
    module = SuperResolution()
    gains = {}
    for s in (2, 3, 4):
        def mean_gain(g: float) -> float:
            out = []
            for img in val_imgs:
                h, w = (img.shape[0] // s) * s, (img.shape[1] // s) * s
                hr = img[:h, :w]
                lr = shrink(hr, s)
                out.append(module._psnr_y(module.super_resolve(lr, s, g), hr)
                           - module._psnr_y(cv2.resize(lr, (w, h), interpolation=cv2.INTER_CUBIC), hr))
            return float(np.mean(out))
        scores = {g: mean_gain(g) for g in np.round(np.arange(0.2, 2.01, 0.1), 1)}
        best = max(scores, key=scores.get)
        gains[str(s)] = float(best)
        print(f"x{s}: best detail gain {best:.1f} ({scores[best]:+.3f} dB on validation, gain 1.0: {scores[1.0]:+.3f} dB)")
    GAINS.write_text(json.dumps(gains))
    return gains


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "train":
        train(Path(sys.argv[2]) if len(sys.argv) > 2 else None)
    elif len(sys.argv) >= 2 and sys.argv[1] == "calibrate":
        calibrate()
    else:
        from core.cli import run_cli
        run_cli(SuperResolution)
