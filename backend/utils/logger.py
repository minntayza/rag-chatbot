"""
Centralised logging with Loguru.

Why Loguru over stdlib logging?
  - coloured output out of the box
  - simple file rotation
  - exception formatting with full tracebacks

Usage::

    from utils.logger import logger
    logger.info("something happened")
"""

from __future__ import annotations

import sys

from loguru import logger as _logger

from config import get_settings

settings = get_settings()

# Remove the default stderr handler — we add our own with a custom format.
_logger.remove()

# ── Console handler ──────────────────────────────────────────────────
LOG_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
    "<level>{message}</level>"
)

_logger.add(
    sys.stderr,
    format=LOG_FORMAT,
    level="DEBUG" if settings.app_debug else "INFO",
    colorize=True,
    backtrace=True,
    diagnose=True,
)

# ── File handler (rotated daily, kept 7 days) ───────────────────────
_logger.add(
    "logs/app_{time:YYYY-MM-DD}.log",
    format=LOG_FORMAT,
    level="DEBUG",
    rotation="00:00",       # new file at midnight
    retention="7 days",     # keep one week
    compression="zip",
    enqueue=True,           # thread-safe writes
)

# Export the configured logger
logger = _logger
