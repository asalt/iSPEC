"""Helpers for loading Make-style ``KEY=value`` env files.

The top-level repo uses `.env.local` files primarily for the Makefile, but
Python subprocesses (like the `ispec` CLI) do not automatically inherit those
values unless they are exported. This module provides a small, dependency-free
loader so CLI commands can accept `--env-file` and behave predictably.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
import os


def extract_env_files(argv: Sequence[str]) -> tuple[list[str], list[str]]:
    """Extract ``--env-file`` arguments from argv.

    Supports both ``--env-file path`` and ``--env-file=path``. The extracted
    arguments are removed from the returned argv so downstream argparse parsing
    won't fail when the flag is placed after subcommands.
    """

    env_files: list[str] = []
    remaining: list[str] = []

    idx = 0
    while idx < len(argv):
        token = argv[idx]
        if token == "--env-file":
            if idx + 1 >= len(argv):
                raise SystemExit("--env-file requires a file path")
            env_files.append(argv[idx + 1])
            idx += 2
            continue

        if token.startswith("--env-file="):
            env_files.append(token.split("=", 1)[1])
            idx += 1
            continue

        remaining.append(token)
        idx += 1

    return env_files, remaining


def _strip_inline_comment(value: str) -> str:
    """Strip trailing ``# ...`` comments when the ``#`` is not quoted."""

    in_single = False
    in_double = False
    prev: str | None = None

    for idx, ch in enumerate(value):
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        elif ch == "#" and not in_single and not in_double and (prev is None or prev.isspace()):
            return value[:idx].rstrip()
        prev = ch
    return value.rstrip()


def parse_env_file_text(text: str) -> dict[str, str]:
    """Parse env-style lines into a dict."""

    parsed: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        if line.startswith("export "):
            line = line[len("export ") :].lstrip()

        if "=" not in line:
            continue

        key, raw_value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue

        value = raw_value.strip()
        if len(value) >= 2 and value[0] in {"'", '"'} and value[-1] == value[0]:
            value = value[1:-1]
        else:
            value = _strip_inline_comment(value)

        parsed[key] = value

    return parsed


def load_env_file(path: str | Path, *, override: bool = True) -> dict[str, str]:
    """Load env vars from a file into ``os.environ``."""

    resolved = Path(path).expanduser()
    if not resolved.exists():
        raise SystemExit(f"--env-file does not exist: {resolved}")

    parsed = parse_env_file_text(resolved.read_text(encoding="utf-8"))
    for key, value in parsed.items():
        if override or key not in os.environ:
            os.environ[key] = value
    return parsed


def load_env_files(paths: Iterable[str | Path], *, override: bool = True) -> dict[str, str]:
    """Load multiple env files in order; later files override earlier ones."""

    merged: dict[str, str] = {}
    for path in paths:
        merged.update(load_env_file(path, override=override))
    return merged

