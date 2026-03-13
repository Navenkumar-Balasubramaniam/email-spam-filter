from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix  # type: ignore

from ..logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class EvalResult:
    metrics: Dict[str, float]
    confusion: np.ndarray
    classes: List[str]


class Evaluator:
    def evaluate(self, y_true, y_pred, classes: List[str]) -> EvalResult:
        cm = confusion_matrix(y_true, y_pred, labels=classes)
        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
            "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
            "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        }
        logger.info("Evaluation metrics: %s", metrics)
        return EvalResult(metrics=metrics, confusion=cm, classes=classes)