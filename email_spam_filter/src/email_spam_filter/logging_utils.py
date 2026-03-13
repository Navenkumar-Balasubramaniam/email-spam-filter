from __future__ import annotations

import logging
import os
from typing import Optional


_DEFAULT_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


class LogConfig:
    """Central logging config used across the package."""
    def __init__(self, level: Optional[str] = None, fmt: str = _DEFAULT_FORMAT) -> None:
        self.level = (level or os.getenv("EMAIL_SPAM_FILTER_LOG_LEVEL", "INFO")).upper()
        self.fmt = fmt

    def configure_root(self) -> None:
        logging.basicConfig(level=self.level, format=self.fmt)


def get_logger(name: str) -> logging.Logger:
    # Root config is set by LogConfig.configure_root() in entrypoints (train, UI, etc.).
    # This getter is still safe even if configure_root wasn't called.
    return logging.getLogger(name)