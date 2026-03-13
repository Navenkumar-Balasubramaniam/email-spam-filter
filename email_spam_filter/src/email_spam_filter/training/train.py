from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import joblib
from sklearn.model_selection import train_test_split  # type: ignore
from sklearn.naive_bayes import MultinomialNB  # type: ignore
from sklearn.pipeline import Pipeline  # type: ignore

from ..logging_utils import LogConfig, get_logger
from .data import DatasetConfig, DatasetLoader
from .evaluate import Evaluator
from .features import PreprocessConfig, TextPreprocessor, VectorizerConfig, VectorizerFactory

logger = get_logger(__name__)


@dataclass
class TrainConfig:
    test_size: float = 0.2
    random_state: int = 42


class Trainer:
    def __init__(
        self,
        dataset_cfg: DatasetConfig,
        preprocess_cfg: PreprocessConfig,
        vectorizer_cfg: VectorizerConfig,
        train_cfg: TrainConfig | None = None,
    ) -> None:
        self.dataset_cfg = dataset_cfg
        self.preprocess_cfg = preprocess_cfg
        self.vectorizer_cfg = vectorizer_cfg
        self.train_cfg = train_cfg or TrainConfig()

        self.loader = DatasetLoader(dataset_cfg)
        self.vectorizer_factory = VectorizerFactory(vectorizer_cfg)
        self.evaluator = Evaluator()

    def build_pipeline(self) -> Pipeline:
        vect = self.vectorizer_factory.build()
        pipe = Pipeline(
            steps=[
                ("prep", TextPreprocessor(self.preprocess_cfg)),
                ("tfidf", vect),
                ("clf", MultinomialNB()),
            ]
        )
        return pipe

    def train_from_csv(self, csv_path: Path):
        df = self.loader.validate_and_clean(self.loader.load_csv(csv_path))
        c = self.dataset_cfg

        X = df[c.text_col].tolist()
        y = df[c.label_col].tolist()

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.train_cfg.test_size,
            random_state=self.train_cfg.random_state,
            stratify=y,
        )

        pipe = self.build_pipeline()
        logger.info("Training pipeline...")
        pipe.fit(X_train, y_train)

        classes = list(pipe.classes_) if hasattr(pipe, "classes_") else sorted(list(set(y)))
        y_pred = pipe.predict(X_test)

        eval_res = self.evaluator.evaluate(y_test, y_pred, classes=classes)
        return pipe, eval_res


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="email_spam_filter.training.train",
        description="Train spam filter pipeline and export joblib + metadata.",
    )
    p.add_argument("csv_path", help="Path to training CSV, e.g. data.csv")
    p.add_argument("--text-col", default="text", help="Text column name (default: text)")
    p.add_argument("--label-col", default="label", help="Label column name (default: label)")
    p.add_argument("--spam-label", default="spam", help="Spam label value (default: spam)")
    p.add_argument("--ham-label", default="ham", help="Ham label value (default: ham)")

    p.add_argument("--out-model", default="src/email_spam_filter/resources/spam_pipeline.joblib",
                   help="Output path for joblib model")
    p.add_argument("--out-meta", default="src/email_spam_filter/resources/metadata.json",
                   help="Output path for metadata json")

    p.add_argument("--test-size", type=float, default=0.2, help="Test split ratio (default: 0.2)")
    p.add_argument("--random-state", type=int, default=42, help="Random seed (default: 42)")
    p.add_argument("--log-level", default=None, help="Logging level (INFO, DEBUG, ...)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    LogConfig(level=args.log_level).configure_root()

    csv_path = Path(args.csv_path)
    out_model = Path(args.out_model)
    out_meta = Path(args.out_meta)

    out_model.parent.mkdir(parents=True, exist_ok=True)
    out_meta.parent.mkdir(parents=True, exist_ok=True)

    dataset_cfg = DatasetConfig(
        text_col=args.text_col,
        label_col=args.label_col,
        spam_label=args.spam_label,
        ham_label=args.ham_label,
    )
    preprocess_cfg = PreprocessConfig()
    vectorizer_cfg = VectorizerConfig()
    train_cfg = TrainConfig(test_size=args.test_size, random_state=args.random_state)

    trainer = Trainer(dataset_cfg, preprocess_cfg, vectorizer_cfg, train_cfg)
    pipeline, eval_res = trainer.train_from_csv(csv_path)

    logger.info("Saving model to: %s", out_model)
    joblib.dump(pipeline, out_model)

    metadata = {
        "package_name": "email_spam_filter",
        "version": "0.1.0",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "spam_label": dataset_cfg.spam_label,
        "classes": eval_res.classes,
        "metrics": eval_res.metrics,
    }
    logger.info("Saving metadata to: %s", out_meta)
    out_meta.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    logger.info("Done.")


if __name__ == "__main__":
    main()