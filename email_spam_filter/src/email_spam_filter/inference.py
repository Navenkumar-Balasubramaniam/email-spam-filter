from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Union

import joblib
import numpy as np
from sklearn.pipeline import Pipeline  # type: ignore

from .logging_utils import get_logger
from .schemas import PredictionResult

logger = get_logger(__name__)

TextLike = Union[str, Sequence[str]]


@dataclass
class ModelPaths:
    model_resource_relpath: str = "resources/spam_pipeline.joblib"
    metadata_resource_relpath: str = "resources/metadata.json"


class ModelLoader:
    """
    Loads a pretrained sklearn Pipeline from either:
      1) environment override EMAIL_SPAM_FILTER_MODEL_PATH (file path), or
      2) packaged resources (default).
    """
    def __init__(self, paths: Optional[ModelPaths] = None) -> None:
        self.paths = paths or ModelPaths()

    def load_pipeline(self) -> Pipeline:
        env_path = os.getenv("EMAIL_SPAM_FILTER_MODEL_PATH")
        if env_path:
            logger.info("Loading pipeline from EMAIL_SPAM_FILTER_MODEL_PATH=%s", env_path)
            return self._load_from_path(Path(env_path))

        logger.info("Loading pipeline from packaged resource: %s", self.paths.model_resource_relpath)
        return self._load_from_package_resource(self.paths.model_resource_relpath)

    def load_metadata(self) -> Optional[dict]:
        env_meta = os.getenv("EMAIL_SPAM_FILTER_METADATA_PATH")
        if env_meta:
            logger.info("Loading metadata from EMAIL_SPAM_FILTER_METADATA_PATH=%s", env_meta)
            return json.loads(Path(env_meta).read_text(encoding="utf-8"))

        try:
            text = self._read_text_from_package_resource(self.paths.metadata_resource_relpath)
            return json.loads(text)
        except Exception as e:
            logger.warning("Metadata not found or failed to load (%s). Continuing without it.", e)
            return None

    def _load_from_path(self, path: Path) -> Pipeline:
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        obj = joblib.load(path)
        if not hasattr(obj, "predict"):
            raise TypeError("Loaded object does not look like an sklearn Pipeline/model.")
        return obj

    def _load_from_package_resource(self, relpath: str) -> Pipeline:
        # Works when installed as a wheel (zip-safe).
        ref = resources.files("email_spam_filter").joinpath(relpath)
        with resources.as_file(ref) as model_path:
            return self._load_from_path(Path(model_path))

    def _read_text_from_package_resource(self, relpath: str) -> str:
        ref = resources.files("email_spam_filter").joinpath(relpath)
        with resources.as_file(ref) as p:
            return Path(p).read_text(encoding="utf-8")


class SpamClassifier:
    """
    High-level class for inference.
    Assumes the underlying model is a Pipeline with vectorizer+classifier (recommended).
    """
    def __init__(self, loader: Optional[ModelLoader] = None) -> None:
        self.loader = loader or ModelLoader()
        self._metadata = None

    @property
    def metadata(self) -> Optional[dict]:
        if self._metadata is None:
            self._metadata = self.loader.load_metadata()
        return self._metadata

    @lru_cache(maxsize=1)
    def pipeline(self) -> Pipeline:
        return self.loader.load_pipeline()

    def classes(self) -> List[str]:
        pipe = self.pipeline()
        if hasattr(pipe, "classes_"):
            return list(pipe.classes_)  # type: ignore[attr-defined]
        # If pipeline wraps estimator, try final step
        if hasattr(pipe, "named_steps") and "clf" in pipe.named_steps and hasattr(pipe.named_steps["clf"], "classes_"):
            return list(pipe.named_steps["clf"].classes_)  # type: ignore[attr-defined]
        raise AttributeError("Could not determine class order (classes_ missing).")

    def spam_index(self, spam_label: str = "spam") -> int:
        classes = self.classes()
        if spam_label not in classes:
            raise ValueError(f"spam_label='{spam_label}' not in classes={classes}")
        return classes.index(spam_label)

    def predict(self, texts: TextLike, spam_label: str = "spam") -> Union[str, List[str]]:
        pipe = self.pipeline()
        if isinstance(texts, str):
            return str(pipe.predict([texts])[0])
        arr = pipe.predict(list(texts))
        return [str(x) for x in arr]

    def predict_proba(self, texts: TextLike) -> Union[np.ndarray, np.ndarray]:
        pipe = self.pipeline()
        if not hasattr(pipe, "predict_proba"):
            raise AttributeError("Model does not support predict_proba()")
        if isinstance(texts, str):
            return pipe.predict_proba([texts])[0]
        return pipe.predict_proba(list(texts))

    def spam_probability(self, texts: TextLike, spam_label: str = "spam") -> Union[float, np.ndarray]:
        idx = self.spam_index(spam_label=spam_label)
        probs = self.predict_proba(texts)
        if isinstance(texts, str):
            return float(probs[idx])
        return probs[:, idx]

    def classify(self, text: str, spam_label: str = "spam", threshold: float = 0.5) -> PredictionResult:
        p_spam = self.spam_probability(text, spam_label=spam_label)
        label = "spam" if p_spam >= threshold else "ham"
        return PredictionResult(label=label, spam_probability=float(p_spam))


# Convenience functional API (nice for users)
_default_classifier = SpamClassifier()


def predict(text: str) -> str:
    return str(_default_classifier.predict(text))


def predict_proba(text: str) -> np.ndarray:
    return _default_classifier.predict_proba(text)  # type: ignore[return-value]


def spam_probability(text: str, spam_label: str = "spam") -> float:
    return float(_default_classifier.spam_probability(text, spam_label=spam_label))


def classify(text: str, threshold: float = 0.5, spam_label: str = "spam") -> PredictionResult:
    return _default_classifier.classify(text, threshold=threshold, spam_label=spam_label)