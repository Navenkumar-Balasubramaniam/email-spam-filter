from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


Label = Literal["spam", "ham"]


@dataclass(frozen=True)
class ModelMetadata:
    package_name: str
    version: str
    trained_at: str
    spam_label: str
    classes: list[str]
    metrics: dict


@dataclass(frozen=True)
class PredictionResult:
    label: Label
    spam_probability: float