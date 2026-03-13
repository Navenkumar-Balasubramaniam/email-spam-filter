import os
from pathlib import Path

import joblib
import numpy as np
import pytest
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

from email_spam_filter.inference import SpamClassifier


@pytest.fixture()
def tiny_model_path(tmp_path: Path) -> Path:
    X = [
        "win money now", "free prize click", "limited offer winner",
        "meeting tomorrow", "project update attached", "let's have lunch",
    ]
    y = ["spam", "spam", "spam", "ham", "ham", "ham"]

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english")),
        ("clf", MultinomialNB()),
    ])
    pipe.fit(X, y)

    model_path = tmp_path / "spam_pipeline.joblib"
    joblib.dump(pipe, model_path)
    return model_path


def test_predict_and_probability_env_override(tiny_model_path: Path, monkeypatch):
    monkeypatch.setenv("EMAIL_SPAM_FILTER_MODEL_PATH", str(tiny_model_path))

    clf = SpamClassifier()
    pred = clf.predict("free prize winner")
    assert pred in ["spam", "ham"]

    p = clf.spam_probability("free prize winner", spam_label="spam")
    assert 0.0 <= p <= 1.0


def test_classify_threshold(tiny_model_path: Path, monkeypatch):
    monkeypatch.setenv("EMAIL_SPAM_FILTER_MODEL_PATH", str(tiny_model_path))

    clf = SpamClassifier()
    res = clf.classify("free prize winner", threshold=0.01)
    assert res.label in ["spam", "ham"]
    assert 0.0 <= res.spam_probability <= 1.0