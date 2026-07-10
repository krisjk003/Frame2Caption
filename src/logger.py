"""Centralized logging configuration for the pipeline."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

_CONFIGURED_LOGGERS: dict[str, logging.Logger] = {}

_LEVEL_COLORS = {
    logging.DEBUG: "\033[36m",  # cyan
    logging.INFO: "\033[32m",  # green
    logging.WARNING: "\033[33m",  # yellow
    logging.ERROR: "\033[31m",  # red
    logging.CRITICAL: "\033[41m",  # red background
}
_RESET = "\033[0m"


class _ColorFormatter(logging.Formatter):
    """Formatter that adds ANSI color to console output when supported."""

    def __init__(self, use_color: bool) -> None:
        super().__init__(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        self._use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        message = super().format(record)
        if not self._use_color:
            return message
        color = _LEVEL_COLORS.get(record.levelno, "")
        return f"{color}{message}{_RESET}" if color else message


def setup_logger(
    name: str = "video_captioning",
    level: str = "INFO",
    log_dir: Path | None = None,
) -> logging.Logger:
    """Configure and return the application's logger.

    Safe to call multiple times: re-configuring is skipped if a logger with
    the given name has already been set up in this process.

    Args:
        name: Logger name.
        level: One of DEBUG, INFO, WARNING, ERROR, CRITICAL.
        log_dir: If provided, log records are also written to
            ``log_dir/app.log`` (directory is created if missing).
    """

    if name in _CONFIGURED_LOGGERS:
        return _CONFIGURED_LOGGERS[name]

    logger = logging.getLogger(name)
    logger.setLevel(level.upper())
    logger.propagate = False

    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setFormatter(_ColorFormatter(use_color=sys.stdout.isatty()))
    logger.addHandler(console_handler)

    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_dir / "app.log", encoding="utf-8")
        file_handler.setFormatter(_ColorFormatter(use_color=False))
        logger.addHandler(file_handler)

    _CONFIGURED_LOGGERS[name] = logger
    return logger


def get_logger(name: str = "video_captioning") -> logging.Logger:
    """Return the previously configured logger, or lazily configure a
    default one (console-only, INFO level) if :func:`setup_logger` has not
    been called yet.
    """

    if name in _CONFIGURED_LOGGERS:
        return _CONFIGURED_LOGGERS[name]
    return setup_logger(name)
