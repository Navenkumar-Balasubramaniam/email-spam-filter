from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List

from sklearn.base import BaseEstimator, TransformerMixin  # type: ignore
from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore

from ..logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class PreprocessConfig:
    lowercase: bool = True
    strip_urls: bool = True
    strip_emails: bool = True
    strip_numbers: bool = False


class TextPreprocessor(BaseEstimator, TransformerMixin):
    """
    sklearn-compatible transformer so preprocessing is baked into the Pipeline.
    """
    def __init__(self, config: PreprocessConfig | None = None) -> None:
        self.config = config or PreprocessConfig()

    def fit(self, X, y=None):  # noqa: N803
        return self

    def transform(self, X):  # noqa: N803
        texts = [str(x) for x in X]
        return [self._clean(t) for t in texts]

    def _clean(self, text: str) -> str:
        c = self.config
        t = text
        if c.lowercase:
            t = t.lower()
        if c.strip_urls:
            t = re.sub(r"https?://\S+|www\.\S+", " ", t)
        if c.strip_emails:
            t = re.sub(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", " ", t)
        if c.strip_numbers:
            t = re.sub(r"\d+", " ", t)
        t = re.sub(r"\s+", " ", t).strip()
        return t


@dataclass
class VectorizerConfig:
    max_features: int = 5000
    ngram_range: tuple[int, int] = (1, 2)
    min_df: int = 2
    stop_words: str | None = "english"


class VectorizerFactory:
    def __init__(self, config: VectorizerConfig | None = None) -> None:
        self.config = config or VectorizerConfig()

    def build(self) -> TfidfVectorizer:
        c = self.config
        logger.info("Building TF-IDF vectorizer: max_features=%s ngram_range=%s", c.max_features, c.ngram_range)
        return TfidfVectorizer(
            max_features=c.max_features,
            ngram_range=c.ngram_range,
            min_df=c.min_df,
            stop_words=c.stop_words,
        )