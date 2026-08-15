"""Experiment logging.

Real logging rather than bare `print`, so experiment output can be captured to a
file alongside the checkpoint and metrics without rewriting call sites.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from .io import ensure_dir

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str = "materials_adv") -> logging.Logger:
    return logging.getLogger(name)


def setup_logging(
    level: int | str = logging.INFO,
    *,
    log_file: str | Path | None = None,
    name: str = "materials_adv",
) -> logging.Logger:
    """Configure the package logger. Idempotent -- safe to call more than once."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(_FORMAT, datefmt=_DATEFMT)

    stream = logging.StreamHandler(sys.stderr)
    stream.setFormatter(formatter)
    logger.addHandler(stream)

    if log_file is not None:
        path = Path(log_file)
        ensure_dir(path.parent)
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
