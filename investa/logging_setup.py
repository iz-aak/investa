"""Structured logging setup using structlog.

All modules should acquire loggers via ``structlog.get_logger(__name__)`` and
bind contextual keys — most importantly ``ticker`` and ``stage`` — so that
per-ticker failures during a scan are trivially greppable.
"""

from __future__ import annotations

import logging
import sys

import structlog

from .config import Settings


def configure_logging(settings: Settings) -> None:
    """Configure structlog + stdlib logging.

    Uses a human-readable console renderer by default; switches to JSON output
    when ``LOG_JSON=true`` (recommended for production / containers).
    """
    level = getattr(logging, settings.log_level.upper(), logging.INFO)

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=level,
    )

    shared_processors: list[structlog.typing.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.TimeStamper(fmt="iso", utc=True),
    ]

    renderer: structlog.typing.Processor = (
        structlog.processors.JSONRenderer()
        if settings.log_json
        else structlog.dev.ConsoleRenderer(colors=sys.stdout.isatty())
    )

    structlog.configure(
        processors=[*shared_processors, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Return a structlog logger. Thin wrapper for import ergonomics."""
    return structlog.get_logger(name)
