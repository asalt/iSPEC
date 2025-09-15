# ispec/logging/logging.py
import os
import logging
import sys
from pathlib import Path
# from functools import lru_cache

_DEFAULT_LOG_DIR = Path(os.environ.get("ISPEC_LOG_DIR", Path.home() / ".ispec" / "logs"))
_DEFAULT_LOG_FILE = _DEFAULT_LOG_DIR / "ispec.log"
# Singleton record to track which loggers are already configured
_LOGGER_INITIALIZED = {}

def _resolve_log_dir(log_dir=None):
    if log_dir is not None:
        return Path(log_dir)
    return Path(os.environ.get("ISPEC_LOG_DIR", Path.home() / ".ispec" / "logs"))

def _resolve_log_file(log_file=None, log_dir=None):
    if log_file is not None:
        return Path(log_file)
    return _resolve_log_dir(log_dir) / "ispec.log"

def ensure_log_dir(log_dir=None):
    dir_ = _resolve_log_dir(log_dir)
    dir_.mkdir(parents=True, exist_ok=True)

def get_logger(
    name="ispec",
    level=logging.INFO,
    log_file=None,
    log_dir=None,
    console=True,
    filemode="a",
    fmt="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    encoding="utf-8",
    propagate=False
):
    """
    Get or create a logger with optional configuration.
    - name: Logger name (default 'ispec')
    - level: Logging level (default logging.INFO)
    - log_file: File path for logs (default: <log_dir>/ispec.log)
    - log_dir: Directory for logs (default: ~/.ispec/logs)
    - console: If True, logs also go to stderr
    - filemode: File mode for log file ('a' append, 'w' overwrite)
    - fmt, datefmt: Formatting for log messages
    - encoding: Encoding for file log
    - propagate: Whether to propagate to root logger (default False)
    """
    logger = logging.getLogger(name)
    if not _LOGGER_INITIALIZED.get(name, False):
        ensure_log_dir(log_dir)
        logger.setLevel(level)
        logger.propagate = propagate  # Typically False for application loggers
        formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)

        if log_file or log_dir:
            file_path = _resolve_log_file(log_file, log_dir)
        else:
            file_path = _resolve_log_file()

        # File handler
        fh = logging.FileHandler(file_path, mode=filemode, encoding=encoding)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        # Console handler
        if console:
            ch = logging.StreamHandler(sys.stderr)
            ch.setFormatter(formatter)
            logger.addHandler(ch)

        _LOGGER_INITIALIZED[name] = True

    return logger


def reset_logger(name=None):
    """Reset configured loggers so they can be reconfigured.

    Parameters
    ----------
    name : str, optional
        Name of the logger to reset. If omitted, all loggers tracked by
        :func:`get_logger` are reset.

    Examples
    --------
    >>> logger = get_logger("demo", level=logging.DEBUG)
    >>> reset_logger("demo")
    >>> logger = get_logger("demo", level=logging.INFO)
    """
    if name is None:
        names = list(_LOGGER_INITIALIZED.keys())
    else:
        names = [name]

    for n in names:
        logger = logging.getLogger(n)
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            handler.close()

    if name is None:
        _LOGGER_INITIALIZED.clear()
    else:
        _LOGGER_INITIALIZED.pop(name, None)


def get_configured_level(name="ispec"):
    """Return the configured logging level name for ``name``."""

    level = logging.getLogger(name).getEffectiveLevel()
    return logging.getLevelName(level)

