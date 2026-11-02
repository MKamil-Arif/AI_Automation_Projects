"""
Robustness test: every topic gets awkward inputs. A clean error message (ValueError /
FileNotFoundError / RuntimeError, shown in the app) is fine; any OTHER exception is a bug.

    python evaluation/robustness_test.py           # all topics
    python evaluation/robustness_test.py 6 17      # only some
"""
from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core import utils  # noqa: E402
from core.registry import available_modules  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CLEAN_ERRORS = (ValueError, FileNotFoundError, RuntimeError)
CLOUD = ("Gemini", "Groq")                                 # skip cloud options (network, quota)


def inputs() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    photo = utils.read_image(ROOT / "samples" / "face_astronaut.png")
    rgba = cv2.cvtColor(photo, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = 128
    return {
        "gray 2-D": cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY),
        "BGRA": rgba,
        "tiny 24x24": cv2.resize(photo, (24, 24)),
        "thin 600x40": cv2.resize(photo, (600, 40)),
        "black": np.zeros((300, 400, 3), np.uint8),
        "white": np.full((300, 400, 3), 255, np.uint8),
        "noise": rng.integers(0, 256, (300, 400, 3), dtype=np.uint8),
        "huge 4000x3000": cv2.resize(photo, (4000, 3000)),
    }


def option_sets(module) -> list[dict]:
    """Default params + every option of each select (one at a time), skipping cloud ones."""
    sets = [{}]
    for p in module.parameters():
        if p.kind == "select":
            for opt in p.options:
                if opt != p.default and not any(c in str(opt) for c in CLOUD):
                    sets.append({p.name: opt})
    return sets


def main() -> None:
    bugs, clean, ok = [], 0, 0
    tests = inputs()
    only = {int(a) for a in sys.argv[1:]}
    for no, cls in available_modules().items():
        if only and no not in only:
            continue
        module = cls()
        for params in option_sets(module):
            if any(str(v).startswith(("Train", "Evaluate", "Enroll", "Mark")) for v in params.values()):
                continue                                   # these write files / take minutes
            for name, img in tests.items():
                start = time.time()
                try:
                    result = module.run(img.copy(), **params)
                    assert result.output is not None and result.output.size > 0, "empty output"
                    assert result.output.dtype == np.uint8, f"output dtype {result.output.dtype}"
                    ok += 1
                except CLEAN_ERRORS:
                    clean += 1
                except Exception as err:                    # noqa: BLE001 - this is what we look for
                    bugs.append((no, cls.__name__, params, name, f"{type(err).__name__}: {err}",
                                 traceback.format_exc().strip().splitlines()[-3]))
                if time.time() - start > 60:
                    bugs.append((no, cls.__name__, params, name, "SLOW (> 60 s)", ""))
        print(f"topic {no:2d} done", flush=True)
    print(f"\n{ok} runs OK, {clean} clean error messages, {len(bugs)} BUGS")
    for b in bugs:
        print("BUG", b)


if __name__ == "__main__":
    main()
