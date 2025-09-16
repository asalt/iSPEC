"""Tests for the console script entry point definition."""
import sys
import types
from importlib import import_module
from pathlib import Path
from typing import Any

try:  # pragma: no cover - fallback for Python < 3.11
    import tomllib  # type: ignore[attr-defined]
except ModuleNotFoundError:  # pragma: no cover - executed when tomllib missing
    import tomli as tomllib  # type: ignore[assignment]


def _noop(*args: Any, **kwargs: Any) -> None:
    """A placeholder callable used to satisfy import-time expectations."""


def _load_entry_point(entry: str) -> Any:
    module_path, _, attr_path = entry.partition(":")
    if not module_path or not attr_path:
        raise AssertionError("Invalid entry point definition")

    # Stub modules that are expensive or unavailable in the test environment.
    fake_db = sys.modules.setdefault("ispec.db", types.ModuleType("ispec.db"))
    if not hasattr(fake_db, "get_session"):
        fake_db.get_session = _noop

    fake_operations = sys.modules.setdefault(
        "ispec.db.operations", types.ModuleType("ispec.db.operations")
    )
    for attr in ("check_status", "show_tables", "import_file", "export_table", "initialize"):
        fake_operations.__dict__.setdefault(attr, _noop)

    if not hasattr(fake_db, "operations"):
        fake_db.operations = fake_operations  # type: ignore[attr-defined]

    module = import_module(module_path)
    target: Any = module
    for part in attr_path.split("."):
        target = getattr(target, part)
    return target


def test_console_script_entry_point_is_callable() -> None:
    project_root = Path(__file__).resolve().parents[3]
    pyproject = tomllib.loads((project_root / "pyproject.toml").read_text())
    entry_point = pyproject["project"]["scripts"]["ispec"]

    loaded = _load_entry_point(entry_point)
    assert callable(loaded)
