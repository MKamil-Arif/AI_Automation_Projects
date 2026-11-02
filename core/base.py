"""
Core building blocks shared by every DIP topic.

OOP concepts used in this project:
- Abstraction   : DIPModule is an abstract class; every topic MUST implement process().
- Inheritance   : every topic class (e.g. BloodCellCounter) extends DIPModule.
- Encapsulation : pipeline steps live in a private list (__steps) and are only
                  exposed read-only through the `steps` property / _add_step().
- Polymorphism  : the app calls run() on any module without knowing which topic it is.
- Template Method pattern: run() fixes the order (validate -> reset -> process),
                  subclasses only fill in process().
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Param:
    """One user-adjustable setting. Any UI (Streamlit, CLI) can build a widget from it.

    kind: "slider" | "number" | "select" | "checkbox" | "text" | "color" | "image"
    visible_when: {"other_param": value or (value1, value2)} - show only when all match.
    """
    name: str
    label: str
    kind: str
    default: Any = None
    min: float | None = None
    max: float | None = None
    step: float | None = None
    options: tuple = ()
    help: str = ""
    visible_when: dict[str, Any] | None = None

    def is_visible(self, values: dict[str, Any]) -> bool:
        if not self.visible_when:
            return True
        for key, allowed in self.visible_when.items():
            allowed = allowed if isinstance(allowed, tuple) else (allowed,)
            if values.get(key) not in allowed:
                return False
        return True


@dataclass
class Step:
    """One intermediate image of the pipeline (shown in 'Behind the scenes')."""
    title: str
    image: np.ndarray


@dataclass
class ModuleResult:
    """What every module returns: final image + numbers/labels + pipeline steps."""
    output: np.ndarray
    info: dict[str, Any] = field(default_factory=dict)
    steps: list[Step] = field(default_factory=list)


class DIPModule(ABC):
    """Abstract parent class of all 24 topics."""

    topic_no: int = 0
    name: str = "Untitled"
    description: str = ""
    samples: tuple[str, ...] = ()          # file names inside samples/ that suit this topic
    wide_output: bool = False              # UI hint: show the output full-width (grids, charts)
    confirm_modes: tuple[str, ...] = ()    # UI hint: "mode" values that only run on a button press

    def __init__(self) -> None:
        self.__steps: list[Step] = []      # private (name-mangled) - encapsulation

    # ----- public API -------------------------------------------------------
    def parameters(self) -> list[Param]:
        """Settings this module accepts. Subclasses override to add their own."""
        return []

    def default_params(self) -> dict[str, Any]:
        return {p.name: p.default for p in self.parameters()}

    def run(self, image: np.ndarray, **params) -> ModuleResult:
        """Template method: validate -> reset steps -> process -> attach steps."""
        self._validate(image)
        self.__steps = []
        merged = {**self.default_params(), **params}
        result = self.process(image, **merged)
        result.steps = list(self.__steps)
        return result

    @abstractmethod
    def process(self, image: np.ndarray, **params) -> ModuleResult:
        """The actual image-processing pipeline of the topic."""

    @property
    def steps(self) -> tuple[Step, ...]:
        return tuple(self.__steps)

    # ----- helpers for subclasses ------------------------------------------
    def _add_step(self, title: str, image: np.ndarray) -> None:
        self.__steps.append(Step(title, image.copy()))

    @staticmethod
    def _validate(image: Any) -> None:
        if not isinstance(image, np.ndarray) or image.size == 0:
            raise ValueError("Input image is empty or invalid.")
        if image.ndim not in (2, 3):
            raise ValueError(f"Expected a 2-D or 3-D image, got shape {image.shape}.")

    def __str__(self) -> str:
        return f"Topic {self.topic_no}: {self.name}"
