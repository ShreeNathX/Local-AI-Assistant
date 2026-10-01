"""
slm/core/logging.py
===================
Structured logging for the SLM project.

Uses loguru for structured, leveled, rotating logs.

Features:
  - JSON-structured or human-readable output (configurable)
  - Automatic log rotation and retention
  - Per-component context binding
  - Console + file sinks

Usage:
    from slm.core.logging import get_logger

    logger = get_logger("DataLoader")
    logger.info("Loading dataset", records=100000, split="train")
    logger.warning("Validation threshold exceeded", valid=92000, threshold=95000)
    logger.error("File not found", path="/data/raw/wiki.txt")
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from loguru import logger as _loguru_logger


# ---------------------------------------------------------------------------
# Global setup — called once at application startup
# ---------------------------------------------------------------------------

_is_configured = False


def configure_logging(
    level: str = "INFO",
    log_dir: str | Path | None = None,
    use_json: bool = False,
    rotation: str = "10 MB",
    retention: str = "30 days",
) -> None:
    """
    Configure global logging for the application.

    Call this ONCE at application startup (e.g., in __main__).
    Subsequent calls are idempotent.

    Parameters
    ----------
    level : str
        Minimum log level: DEBUG | INFO | WARNING | ERROR | CRITICAL
    log_dir : str or Path, optional
        Directory where log files will be written. If None, only
        console logging is enabled.
    use_json : bool
        If True, emit JSON-structured logs (for machine parsing).
        If False, emit human-readable colored console output.
    rotation : str
        loguru rotation trigger (e.g. "10 MB", "1 day").
    retention : str
        loguru retention period (e.g. "30 days").
    """
    global _is_configured
    if _is_configured:
        return

    # Remove default loguru handler
    _loguru_logger.remove()

    # --- Console sink ---
    if use_json:
        fmt = _json_formatter
    else:
        fmt = (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{extra[component]}</cyan> | "
            "{message}"
        )

    _loguru_logger.add(
        sys.stdout,
        format=fmt,
        level=level,
        colorize=not use_json,
        backtrace=True,
        diagnose=True,
    )

    # --- File sink (if log_dir provided) ---
    if log_dir is not None:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        _loguru_logger.add(
            str(log_path / "slm_{time:YYYY-MM-DD}.log"),
            format=_json_formatter,
            level=level,
            rotation=rotation,
            retention=retention,
            compression="gz",
            backtrace=True,
            diagnose=True,
            enqueue=True,   # thread-safe async writes
        )

    _is_configured = True


def _json_formatter(record: dict) -> str:  # type: ignore[type-arg]
    """Format a loguru record as a single-line JSON object."""
    log_entry = {
        "timestamp": record["time"].strftime("%Y-%m-%dT%H:%M:%S.%f"),
        "level":     record["level"].name,
        "component": record["extra"].get("component", "SLM"),
        "message":   record["message"],
        "file":      record["file"].name,
        "line":      record["line"],
    }
    # Merge any extra kwargs passed by the caller (e.g., records=1000)
    for k, v in record["extra"].items():
        if k != "component":
            log_entry[k] = v
    if record["exception"]:
        exc = record["exception"]
        log_entry["exception"] = {
            "type":    exc.type.__name__ if exc.type else None,
            "value":   str(exc.value),
            "traceback": record["exception"].traceback is not None,
        }
    return json.dumps(log_entry, default=str) + "\n"


# ---------------------------------------------------------------------------
# Per-component logger factory
# ---------------------------------------------------------------------------

class ComponentLogger:
    """
    A thin wrapper around a loguru logger bound to a specific component.

    Usage:
        logger = get_logger("DataLoader")
        logger.info("records loaded", count=50000)
        # Outputs: [DataLoader] records loaded | count=50000
    """

    def __init__(self, component: str) -> None:
        self._logger = _loguru_logger.bind(component=component)
        self.component = component

    # --- Level methods — accept arbitrary kwargs as structured fields ---

    def debug(self, message: str, **kwargs: Any) -> None:
        self._logger.opt(depth=1).debug(self._format(message, kwargs))

    def info(self, message: str, **kwargs: Any) -> None:
        self._logger.opt(depth=1).info(self._format(message, kwargs))

    def warning(self, message: str, **kwargs: Any) -> None:
        self._logger.opt(depth=1).warning(self._format(message, kwargs))

    def error(self, message: str, **kwargs: Any) -> None:
        self._logger.opt(depth=1).error(self._format(message, kwargs))

    def critical(self, message: str, **kwargs: Any) -> None:
        self._logger.opt(depth=1).critical(self._format(message, kwargs))

    def exception(self, message: str, **kwargs: Any) -> None:
        """Log an exception with full traceback."""
        self._logger.opt(depth=1, exception=True).error(self._format(message, kwargs))

    @staticmethod
    def _format(message: str, kwargs: dict[str, Any]) -> str:
        if kwargs:
            kv = " | ".join(f"{k}={v}" for k, v in kwargs.items())
            return f"{message} | {kv}"
        return message


def get_logger(component: str) -> ComponentLogger:
    """
    Get a structured logger bound to the given component name.

    If logging has not been configured yet, a minimal default
    configuration is applied automatically (console, INFO level).

    Parameters
    ----------
    component : str
        Component name, e.g. "DataLoader", "Tokenizer", "Trainer".
    """
    if not _is_configured:
        # Auto-configure with sensible defaults so modules can import
        # without requiring explicit setup (useful for tests and scripts).
        log_level = os.environ.get("SLM_LOG_LEVEL", "INFO").upper()
        configure_logging(level=log_level, use_json=False)

    return ComponentLogger(component)
