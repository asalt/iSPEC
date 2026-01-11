import os
import sys
from pathlib import Path
import types

import pytest

# Ensure the src directory is on the path for imports
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / 'src'))

log_dir = root / "logs"
log_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("ISPEC_LOG_DIR", str(log_dir))

# Provide a lightweight stub for requests if it's not installed
try:
    import requests  # type: ignore  # noqa: F401
except Exception:  # pragma: no cover - executed only when requests missing
    requests_stub = types.ModuleType('requests')
    requests_stub.put = lambda *args, **kwargs: None
    sys.modules['requests'] = requests_stub


def _is_truthy(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-testclient",
        action="store_true",
        default=_is_truthy(os.getenv("ISPEC_RUN_TESTCLIENT")),
        help=(
            "Run tests that require FastAPI/Starlette TestClient. "
            "These tests can hang in some sandboxed environments."
        ),
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "testclient: requires FastAPI/Starlette TestClient (opt-in via --run-testclient)",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--run-testclient"):
        return

    skip = pytest.mark.skip(
        reason="Requires FastAPI/Starlette TestClient (opt-in with --run-testclient or ISPEC_RUN_TESTCLIENT=1)."
    )
    for item in items:
        if "testclient" in item.keywords:
            item.add_marker(skip)
