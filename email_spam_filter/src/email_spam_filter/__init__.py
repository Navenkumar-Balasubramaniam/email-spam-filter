from .inference import predict, predict_proba, spam_probability, classify, SpamClassifier
from .viz import Visualizer

__all__ = [
    "predict",
    "predict_proba",
    "spam_probability",
    "classify",
    "SpamClassifier",
    "Visualizer",
]

__version__ = "0.1.0"
