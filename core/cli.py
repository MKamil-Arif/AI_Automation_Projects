"""
Run any topic from the terminal without the UI:

    python -m topics.t23_blood_cell_counter samples/blood_smear_synthetic.png method=Watershed

Output image + every pipeline step are saved in outputs/topic_<no>/.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

from core import utils
from core.base import DIPModule

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"


def _convert(raw: str, default):
    if isinstance(default, bool):
        return raw.lower() in ("1", "true", "yes", "on")
    if isinstance(default, int):
        return int(raw)
    if isinstance(default, float):
        return float(raw)
    return raw


def parse_params(module: DIPModule, args: list[str]) -> dict:
    params = {}
    specs = {p.name: p for p in module.parameters()}
    for arg in args:
        key, _, raw = arg.partition("=")
        if key not in specs:
            raise SystemExit(f"Unknown parameter '{key}'. Valid: {', '.join(specs)}")
        spec = specs[key]
        params[key] = utils.read_image(raw) if spec.kind == "image" else _convert(raw, spec.default)
    return params


def run_cli(module_cls: type[DIPModule], argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if hasattr(sys.stdout, "reconfigure"):          # Windows consoles (cp1252) cannot print e.g. "∞"
        sys.stdout.reconfigure(errors="replace")
    module = module_cls()
    if not argv:
        print(f"{module}\n{module.description}\n")
        print("Usage: python -m topics.<file> <image_path> [name=value ...]\nParameters:")
        for p in module.parameters():
            extra = f" options={list(p.options)}" if p.options else ""
            print(f"  {p.name:<16} default={p.default!r}{extra}  {p.help}")
        return

    image = utils.read_image(argv[0])
    result = module.run(image, **parse_params(module, argv[1:]))

    out_dir = OUTPUT_DIR / f"topic_{module.topic_no:02d}"
    utils.write_image(out_dir / "output.png", result.output)
    for i, step in enumerate(result.steps, 1):
        safe = re.sub(r"[^A-Za-z0-9]+", "_", step.title).strip("_")[:50]
        utils.write_image(out_dir / f"step_{i:02d}_{safe}.png", step.image)

    print(module)
    for key, value in result.info.items():
        print(f"  {key}: {value}")
    print(f"Saved output + {len(result.steps)} steps to {out_dir}")
