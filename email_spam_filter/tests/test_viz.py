import joblib
import numpy as np
import pytest
from pathlib import Path

from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

from email_spam_filter.viz import Visualizer


@pytest.fixture()
def nb_pipeline() -> Pipeline:
    X = ["win prize", "free money", "project meeting", "schedule update"]
    y = ["spam", "spam", "ham", "ham"]
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english")),
        ("clf", MultinomialNB()),
    ])
    pipe.fit(X, y)
    return pipe


def test_plot_top_tokens(nb_pipeline: Pipeline):
    viz = Visualizer()
    fig = viz.plot_top_tokens(nb_pipeline, class_label="spam", n=5)
    assert fig is not None


def test_plot_confusion_matrix():
    viz = Visualizer()
    cm = np.array([[5, 1], [2, 7]])
    fig = viz.plot_confusion_matrix(cm, ["ham", "spam"])
    assert fig is not None