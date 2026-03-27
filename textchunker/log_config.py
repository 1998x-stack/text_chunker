from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger

from .settings import Settings

_CONSOLE_FORMAT = (
    "<green>{time:HH:mm:ss}</green> | "
    "<level>{level:<8}</level> | "
    "<cyan>{module}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
    "<level>{message}</level>"
)


def setup_logging(settings: Settings) -> None:
    """Configure loguru with console + optional JSON file sinks.

    Removes all existing handlers first to allow re-configuration.
    """
    logger.remove()

    # Console sink — human-readable, colored
    logger.add(
        sys.stderr,
        format=_CONSOLE_FORMAT,
        level=settings.log_level.upper(),
        colorize=True,
    )

    # File sink — JSON structured logs with rotation
    log_dir = Path(settings.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "textchunker.log"

    logger.add(
        str(log_file),
        format="{message}",
        level=settings.log_level.upper(),
        rotation=settings.log_rotation,
        retention=settings.log_retention,
        serialize=settings.log_json,
        encoding="utf-8",
    )
