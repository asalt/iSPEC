import os
import sys
from pathlib import Path
import types

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
