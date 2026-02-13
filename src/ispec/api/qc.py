from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

_QC_MAP_ENV_VAR = "ISPEC_QC_EXPERIMENTS_JSON"
_DEFAULT_QC_MAP_RELATIVE_PATH = Path("data") / "qc-experiments.json"
_DEFAULT_QC_EXPERIMENTS: dict[int, str] = {
    99990: "Orbitrap Exploris",
    99991: "Orbitrap Eclipse",
    99992: "timsTOF 2",
    99995: "Orbitrap Lumos ETD",
    99999: "Orbitrap Fusion",
}


def qc_mapping_path() -> Path:
    raw = (os.getenv(_QC_MAP_ENV_VAR) or "").strip()
    if raw:
        return Path(raw).expanduser()
    # src/ispec/api/qc.py -> parents[3] is repo root
    return Path(__file__).resolve().parents[3] / _DEFAULT_QC_MAP_RELATIVE_PATH


def _default_qc_payload() -> dict[str, Any]:
    return {
        "version": 1,
        "description": "Legacy QC experiment ID mapping used for API-only derived classification.",
        "experiments": {
            str(exp_id): {"qc_instrument": instrument}
            for exp_id, instrument in sorted(_DEFAULT_QC_EXPERIMENTS.items())
        },
    }


def _parse_qc_payload(data: Any) -> dict[int, str | None]:
    entries = data.get("experiments") if isinstance(data, dict) else None
    if not isinstance(entries, dict):
        return {}

    resolved: dict[int, str | None] = {}
    for raw_id, raw_value in entries.items():
        try:
            exp_id = int(raw_id)
        except Exception:
            continue

        instrument: str | None = None
        if isinstance(raw_value, str):
            instrument = raw_value.strip() or None
        elif isinstance(raw_value, dict):
            val = raw_value.get("qc_instrument")
            if not isinstance(val, str) or not val.strip():
                val = raw_value.get("instrument")
            if isinstance(val, str):
                instrument = val.strip() or None

        resolved[exp_id] = instrument

    return resolved


@lru_cache(maxsize=1)
def _load_qc_map(path_text: str) -> dict[int, str | None]:
    path = Path(path_text)
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        return _parse_qc_payload(data)

    # If caller explicitly overrides the mapping path, do not auto-create.
    if (os.getenv(_QC_MAP_ENV_VAR) or "").strip():
        return {}

    # Bootstrap a sensible default map into the untracked data/ folder.
    payload = _default_qc_payload()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except Exception:
        pass

    return _parse_qc_payload(payload)


def clear_qc_map_cache() -> None:
    _load_qc_map.cache_clear()


def qc_flags_for_experiment_id(experiment_id: Any) -> dict[str, Any]:
    try:
        exp_id = int(experiment_id)
    except Exception:
        return {"is_qc": False, "qc_instrument": None}

    qc_map = _load_qc_map(str(qc_mapping_path()))
    if exp_id not in qc_map:
        return {"is_qc": False, "qc_instrument": None}

    return {"is_qc": True, "qc_instrument": qc_map.get(exp_id)}
