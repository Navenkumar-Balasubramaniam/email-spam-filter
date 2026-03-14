from pathlib import Path
import json

from email_spam_filter.training.train import Trainer, TrainConfig
from email_spam_filter.training.data import DatasetConfig
from email_spam_filter.training.features import PreprocessConfig, VectorizerConfig


def test_training_end_to_end(tmp_path: Path):
    csv_path = tmp_path / "data.csv"
    csv_path.write_text(
        "text,label\n"
        "win money now,spam\n"
        "free prize click,spam\n"
        "meeting tomorrow,ham\n"
        "project update,ham\n",
        encoding="utf-8"
    )

    trainer = Trainer(
        dataset_cfg=DatasetConfig(text_col="text", label_col="label", spam_label="spam", ham_label="ham"),
        preprocess_cfg=PreprocessConfig(),
        vectorizer_cfg=VectorizerConfig(min_df=1),  # tiny dataset
        train_cfg=TrainConfig(test_size=0.5, random_state=1),
    )

    pipe, eval_res = trainer.train_from_csv(csv_path)
    assert hasattr(pipe, "predict")
    assert "f1_macro" in eval_res.metrics
    assert eval_res.confusion.shape == (2, 2)