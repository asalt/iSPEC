"""Helpers for persisting logging configuration."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional


def _default_config_dir() -> Path:
    """Return the default config directory, honoring env overrides."""

    raw = os.environ.get("ISPEC_CONFIG_DIR")
    if raw and raw.strip():
        return Path(raw).expanduser()
    return Path.home() / ".ispec"


def _default_config_path() -> Path:
    """Return the default logging config path, honoring env overrides."""

    raw = os.environ.get("ISPEC_LOG_CONFIG")
    if raw and raw.strip():
        return Path(raw).expanduser()
    return _default_config_dir() / "logging.json"


def _resolve_config_path(config_file: Optional[os.PathLike[str] | str] = None) -> Path:
    """Return the path to the logging configuration file."""

    if config_file is not None:
        return Path(config_file)
    return _default_config_path()


def load_config(config_file: Optional[os.PathLike[str] | str] = None) -> Dict[str, Any]:
    """Load the logging configuration JSON file.

    Parameters
    ----------
    config_file:
        Optional override for the configuration file path. When omitted, the
        path is resolved from ``ISPEC_LOG_CONFIG`` or defaults to
        ``~/.ispec/logging.json``.

    Returns
    -------
    dict
        Parsed configuration content. Missing files or decoding errors are
        treated as an empty configuration.
    """

    path = _resolve_config_path(config_file)
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except FileNotFoundError:
        return {}
    except (OSError, json.JSONDecodeError):
        return {}

    if isinstance(data, dict):
        return data
    return {}


def save_config(
    config: Dict[str, Any],
    config_file: Optional[os.PathLike[str] | str] = None,
) -> Path:
    """Persist the provided logging configuration to disk."""

    path = _resolve_config_path(config_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return path


def _normalize_level(
    level: str | int,
) -> tuple[str, int]:
    """Coerce a logging level into a (name, numeric value) pair.

    Raises
    ------
    ValueError
        If the provided level cannot be resolved.
    """

    if isinstance(level, int):
        numeric = level
        name = logging.getLevelName(level)
        if not isinstance(name, str) or name.startswith("Level "):
            name = str(level)
        return name, numeric

    candidate = logging.getLevelName(str(level).upper())
    if isinstance(candidate, int):
        return str(level).upper(), candidate

    raise ValueError(f"Unknown logging level: {level!r}")


def load_log_level(
    config_file: Optional[os.PathLike[str] | str] = None,
) -> Optional[int]:
    """Fetch the persisted log level, if any."""

    value = load_config(config_file).get("log_level")
    if value is None:
        return None

    if isinstance(value, int):
        return value

    candidate = logging.getLevelName(str(value).upper())
    if isinstance(candidate, int):
        return candidate

    return None


def save_log_level(
    level: str | int,
    config_file: Optional[os.PathLike[str] | str] = None,
) -> Path:
    """Persist the supplied log level and return the config path."""

    name, _ = _normalize_level(level)
    config = load_config(config_file)
    config["log_level"] = name
    return save_config(config, config_file)


__all__ = [
    "load_config",
    "save_config",
    "load_log_level",
    "save_log_level",
]
