from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import pandas as pd

from ..logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class DatasetConfig:
    text_col: str = "text"
    label_col: str = "label"
    spam_label: str = "spam"
    ham_label: str = "ham"


class DatasetLoader:
    def __init__(self, config: DatasetConfig | None = None) -> None:
        self.config = config or DatasetConfig()

    def load_csv(self, csv_path: Path) -> pd.DataFrame:
        logger.info("Loading dataset: %s", csv_path)
        df = pd.read_csv(csv_path)
        return df

    def validate_and_clean(self, df: pd.DataFrame) -> pd.DataFrame:
        c = self.config
        if c.text_col not in df.columns or c.label_col not in df.columns:
            raise ValueError(f"Dataset must contain columns: '{c.text_col}' and '{c.label_col}'")

        out = df[[c.text_col, c.label_col]].copy()
        out[c.text_col] = out[c.text_col].astype(str).fillna("")
        out[c.label_col] = out[c.label_col].astype(str).str.lower().str.strip()
        out = out.dropna()

        # Keep only spam/ham
        out = out[out[c.label_col].isin([c.spam_label, c.ham_label])].reset_index(drop=True)
        if out.empty:
            raise ValueError("No valid rows after cleaning. Check label values.")

        logger.info("Dataset rows after cleaning: %d", len(out))
        return out