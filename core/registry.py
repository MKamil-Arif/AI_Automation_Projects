"""
Keeps track of which topics are implemented.

A topic file only has to put @register above its class - the app finds it automatically.
"""
from __future__ import annotations

import importlib
import pkgutil

from core.base import DIPModule

ALL_TOPICS: dict[int, str] = {
    1: "Tree Detection & Counting",
    2: "Fish Species Recognition",
    3: "Flower Species Recognition",
    4: "Face Recognition Attendance System",
    5: "Number Plate Recognition (ANPR)",
    6: "Fingerprint Minutiae Matching",
    7: "Fake Currency Detection",
    8: "Face Mask & Helmet Detection",
    9: "Image Caption Generator",
    10: "Image Super Resolution (SRCNN)",
    11: "Real-time Emotion Recognition",
    12: "Image Search Engine (CBIR)",
    13: "Road Pothole & Crack Detection",
    14: "Waste Segregation",
    15: "Smart Auto Photo Editor",
    16: "Old Photo Restoration Tool",
    17: "Background Remover & Changer",
    18: "Image Forgery Detection",
    19: "Invisible Watermarking System",
    20: "Plant Leaf Disease Detection",
    21: "Tree Species Recognition from Leaf",
    22: "Fruit Ripeness Detector",
    23: "Blood Cell Counter",
    24: "Crop Weed Detection from Drone Image",
}

_registry: dict[int, type[DIPModule]] = {}


def register(cls: type[DIPModule]) -> type[DIPModule]:
    """Class decorator: adds a topic class to the registry."""
    if not issubclass(cls, DIPModule):
        raise TypeError(f"{cls.__name__} must inherit from DIPModule")
    _registry[cls.topic_no] = cls     # overwrite is fine (Streamlit hot-reload re-imports files)
    return cls


def available_modules() -> dict[int, type[DIPModule]]:
    """All implemented topics, sorted by topic number."""
    import topics

    # importing each topic file runs its @register decorator
    for info in pkgutil.iter_modules(topics.__path__):
        if info.name.startswith("t") and info.name[1:3].isdigit():
            importlib.import_module(f"topics.{info.name}")
    return dict(sorted(_registry.items()))
