"""
DIP Studio - one app for all Digital Image Processing topics.

    streamlit run app.py

The app knows nothing about any specific topic: it asks each module for its parameters,
builds the widgets, and calls module.run() (polymorphism).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from core import utils
from core.base import DIPModule, Param
from core.registry import ALL_TOPICS, available_modules

SAMPLES_DIR = Path(__file__).resolve().parent / "samples"
IMAGE_TYPES = ["png", "jpg", "jpeg", "bmp", "webp"]
MAX_SIDE = 2000          # very large uploads are shrunk so the app stays responsive

st.set_page_config(page_title="DIP Studio", page_icon="🖼️", layout="wide")


# ----- widgets ---------------------------------------------------------------------
def render_param(p: Param, key: str):
    """Turn one Param description into a Streamlit widget and return its value."""
    help_text = p.help or None
    if p.kind == "slider":
        return st.slider(p.label, min_value=p.min, max_value=p.max, value=p.default,
                         step=p.step, help=help_text, key=key)
    if p.kind == "number":
        return st.number_input(p.label, min_value=p.min, max_value=p.max, value=p.default,
                               step=p.step, help=help_text, key=key)
    if p.kind == "select":
        return st.selectbox(p.label, p.options, index=p.options.index(p.default), help=help_text, key=key)
    if p.kind == "checkbox":
        return st.checkbox(p.label, value=p.default, help=help_text, key=key)
    if p.kind == "text":
        return st.text_input(p.label, value=p.default, help=help_text, key=key)
    if p.kind == "color":
        return st.color_picker(p.label, value=p.default, help=help_text, key=key)
    if p.kind == "image":
        file = st.file_uploader(p.label, type=IMAGE_TYPES, help=help_text, key=key)
        return utils.decode_image(file.getvalue()) if file else None
    raise ValueError(f"Unknown parameter kind: {p.kind}")


def pick_image(module: DIPModule) -> tuple[np.ndarray | None, str]:
    key = f"t{module.topic_no}"
    source = st.radio("Image source", ["Upload", "Sample", "Camera"], horizontal=True, key=f"{key}_src")
    if source == "Upload":
        file = st.file_uploader("Upload an image", type=IMAGE_TYPES, key=f"{key}_upload")
        return (utils.decode_image(file.getvalue()), Path(file.name).stem) if file else (None, "")
    if source == "Sample":
        files = [SAMPLES_DIR / name for name in module.samples if (SAMPLES_DIR / name).exists()]
        files = files or sorted(SAMPLES_DIR.glob("*.png"))
        if not files:
            st.info("No sample images yet. Run:  python samples/generate_samples.py")
            return None, ""
        choice = st.selectbox("Sample image", files, format_func=lambda p: p.name, key=f"{key}_sample")
        return utils.read_image(choice), choice.stem
    shot = st.camera_input("Take a photo", key=f"{key}_camera")
    return (utils.decode_image(shot.getvalue()), "camera") if shot else (None, "")


def show_info(info: dict) -> None:
    """Short values as metric cards, long text as lines, lists of dicts as tables."""
    tables = {k: v for k, v in info.items() if isinstance(v, list)}
    scalars = {k: v for k, v in info.items() if k not in tables}
    short = [(k, v) for k, v in scalars.items() if len(str(v)) <= 12]
    for i in range(0, len(short), 3):
        for col, (k, v) in zip(st.columns(3), short[i:i + 3]):
            col.metric(k, v)
    for k, v in scalars.items():
        if len(str(v)) > 12:
            st.markdown(f"**{k}:** {v}")
    for k, rows in tables.items():
        st.markdown(f"**{k}**")
        st.dataframe(pd.DataFrame(rows), hide_index=True)


# ----- page ------------------------------------------------------------------------
modules = available_modules()

with st.sidebar:
    st.title("🖼️ DIP Studio")
    st.caption(f"{len(modules)} of {len(ALL_TOPICS)} topics ready")
    topic_no = st.radio("Topic", list(modules), format_func=lambda n: f"{n:02d} · {modules[n].name}")
    with st.expander("Coming soon"):
        st.markdown("\n".join(f"- {n:02d} · {title}" for n, title in ALL_TOPICS.items() if n not in modules))

module = modules[topic_no]()
st.title(f"Topic {topic_no}: {module.name}")
st.write(module.description)

left, right = st.columns([1, 2], gap="large")
with left:
    st.subheader("Input")
    image, image_name = pick_image(module)
    st.subheader("Settings")
    values: dict = {}
    for param in module.parameters():
        # hidden params keep their default; controlling params (e.g. "mode") come first
        visible = param.is_visible(values)
        values[param.name] = render_param(param, f"t{topic_no}_{param.name}") if visible else param.default

with right:
    if image is None:
        st.info("Upload, pick a sample or take a photo to start.")
        st.stop()
    image, _ = utils.resize_max(image, MAX_SIDE)
    if values.get("mode") in module.confirm_modes and not st.button(f"▶ Start: {values['mode']}", type="primary"):
        st.info("This can take a while, so it only runs when you press the button.")
        st.stop()
    try:
        with st.spinner("Processing..."):
            result = module.run(image, **values)
    except (ValueError, FileNotFoundError, RuntimeError) as err:
        st.error(str(err))
        st.stop()

    if module.wide_output:
        st.image(utils.to_display(result.output), caption="Output", width="stretch")
    else:
        col_in, col_out = st.columns(2)
        col_in.image(utils.to_display(image), caption="Input", width="stretch")
        col_out.image(utils.to_display(result.output), caption="Output", width="stretch")
    st.download_button("⬇️ Download output (PNG)", utils.encode_png(result.output),
                       file_name=f"{image_name or 'output'}_topic{topic_no}.png", mime="image/png")
    show_info(result.info)

st.divider()
st.subheader("Behind the scenes: pipeline steps")
cols = st.columns(3)
for i, step in enumerate(result.steps):
    cols[i % 3].image(utils.to_display(step.image), caption=step.title, width="stretch")
