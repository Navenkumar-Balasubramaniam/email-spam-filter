from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # type: ignore
from sklearn.pipeline import Pipeline  # type: ignore

from .logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class VizConfig:
    top_n_tokens: int = 20


class Visualizer:
    def __init__(self, config: Optional[VizConfig] = None) -> None:
        self.config = config or VizConfig()

    def plot_probability_histogram(self, spam_probs: Sequence[float], title: str = "Spam Probability Distribution"):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.hist(list(spam_probs), bins=20)
        ax.set_title(title)
        ax.set_xlabel("P(spam)")
        ax.set_ylabel("Count")
        fig.tight_layout()
        return fig

    def plot_class_counts(self, labels: Sequence[str], title: str = "Predicted Class Counts"):
        labels_list = list(labels)
        spam_count = sum(1 for x in labels_list if x == "spam")
        ham_count = sum(1 for x in labels_list if x == "ham")

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.bar(["ham", "spam"], [ham_count, spam_count])
        ax.set_title(title)
        ax.set_ylabel("Count")
        fig.tight_layout()
        return fig

    def plot_confusion_matrix(self, cm: np.ndarray, class_names: List[str]):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        im = ax.imshow(cm)
        ax.set_title("Confusion Matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names)
        ax.set_yticklabels(class_names)

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center")

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        return fig

    def plot_top_tokens(self, pipeline: Pipeline, class_label: str = "spam", n: Optional[int] = None):
        """
        Works best for MultinomialNB inside a Pipeline that includes a vectorizer step named 'tfidf'
        and a classifier step named 'clf'.
        """
        n = n or self.config.top_n_tokens
        if not hasattr(pipeline, "named_steps"):
            raise TypeError("Expected an sklearn Pipeline with named_steps.")

        if "tfidf" not in pipeline.named_steps or "clf" not in pipeline.named_steps:
            raise ValueError("Pipeline must contain steps named 'tfidf' and 'clf'.")

        vect = pipeline.named_steps["tfidf"]
        clf = pipeline.named_steps["clf"]

        if not hasattr(vect, "get_feature_names_out"):
            raise TypeError("Vectorizer must support get_feature_names_out().")
        if not hasattr(clf, "feature_log_prob_") or not hasattr(clf, "classes_"):
            raise TypeError("Classifier must be NB-like with feature_log_prob_ and classes_.")

        classes = list(clf.classes_)
        if class_label not in classes:
            raise ValueError(f"class_label='{class_label}' not in classes={classes}")

        idx = classes.index(class_label)
        feature_names = vect.get_feature_names_out()
        weights = clf.feature_log_prob_[idx]  # log prob for each feature
        top_idx = np.argsort(weights)[-n:][::-1]

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.bar(range(n), weights[top_idx])
        ax.set_xticks(range(n))
        ax.set_xticklabels([feature_names[i] for i in top_idx], rotation=45, ha="right")
        ax.set_title(f"Top {n} tokens for class: {class_label}")
        fig.tight_layout()
        return fig