"""Project-wide logging helpers.

Console logs are intentionally step-level only, while the log file records both
step-level progress and detailed debug information.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional


class StepOnlyFilter(logging.Filter):
    """Allow console output only for records explicitly marked as steps."""

    def filter(self, record: logging.LogRecord) -> bool:
        return bool(getattr(record, "is_step", False))


def setup_logging(
    name: str,
    log_file: Optional[Path | str] = None,
    logs_dir: Path | str = "logs",
    file_level: int = logging.DEBUG,
    console_level: int = logging.INFO,
) -> logging.Logger:
    """Create a logger with step-only console output and verbose file output."""

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    log_path = Path(log_file) if log_file else Path(logs_dir) / f"{name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.addFilter(StepOnlyFilter())
    console_handler.setFormatter(logging.Formatter("%(message)s"))

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(file_level)
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.debug("Logger initialized. log_file=%s", log_path)
    return logger


def log_step(logger: logging.Logger, message: str) -> None:
    """Log a high-level progress message to both console and file."""

    logger.info(message, extra={"is_step": True})

