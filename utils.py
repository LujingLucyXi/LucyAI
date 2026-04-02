"""
Logging and helper utilities for the ML system design framework.
Provides standardized logging, timing, and common utility functions.
"""

import logging
import os
import time
import functools
from typing import Any, Callable, Optional

from config import LoggingConfig


def setup_logger(
    name: str,
    config: Optional[LoggingConfig] = None,
) -> logging.Logger:
    """
    Configure and return a named logger.

    Parameters
    ----------
    name:
        Logger name (typically the calling module's ``__name__``).
    config:
        Logging configuration.  Defaults to ``LoggingConfig()`` when *None*.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    if config is None:
        config = LoggingConfig()

    logger = logging.getLogger(name)
    if logger.handlers:
        # Avoid duplicate handlers when the logger is requested multiple times.
        return logger

    logger.setLevel(getattr(logging, config.log_level.upper(), logging.INFO))

    formatter = logging.Formatter(config.log_format)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if config.log_file:
        from logging.handlers import RotatingFileHandler

        log_dir = os.path.dirname(config.log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        file_handler = RotatingFileHandler(
            config.log_file,
            maxBytes=config.max_log_size_mb * 1024 * 1024,
            backupCount=config.backup_count,
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def timer(func: Callable) -> Callable:
    """Decorator that logs the execution time of a function."""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        _logger = logging.getLogger(func.__module__ or __name__)
        start = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - start
            _logger.debug("%s completed in %.3fs", func.__qualname__, elapsed)
            return result
        except Exception:
            elapsed = time.perf_counter() - start
            _logger.error(
                "%s failed after %.3fs", func.__qualname__, elapsed, exc_info=True
            )
            raise

    return wrapper


def ensure_directory(path: str) -> None:
    """Create *path* (and any missing parents) if it does not already exist."""
    os.makedirs(path, exist_ok=True)


def clamp(value: float, low: float, high: float) -> float:
    """Return *value* clamped to the range [*low*, *high*]."""
    return max(low, min(value, high))


def flatten_dict(nested: dict, sep: str = ".") -> dict:
    """
    Flatten a nested dictionary using *sep* as the key separator.

    Example
    -------
    >>> flatten_dict({"a": {"b": 1}, "c": 2})
    {'a.b': 1, 'c': 2}
    """
    result: dict = {}

    def _flatten(obj: Any, prefix: str = "") -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                new_key = f"{prefix}{sep}{k}" if prefix else k
                _flatten(v, new_key)
        else:
            result[prefix] = obj

    _flatten(nested)
    return result
