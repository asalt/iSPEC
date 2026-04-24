from __future__ import annotations

import json
import math
import os
import time
from types import SimpleNamespace
from datetime import UTC, datetime
from pathlib import Path
from string import Template
from typing import Any
from urllib import error, request

from ispec.assistant.connect import get_assistant_session
from ispec.assistant.models import SupportMessage, SupportSession
from ispec.config.paths import resolve_log_dir


DEFAULT_ENV_FILES = (
    ".env",
    ".env.local",
    ".env.vllm",
    ".env.vllm.local",
    ".env.slack",
    ".env.slack.local",
)


_VLLM_BASELINE_ENV_KEYS = (
    "ISPEC_VLLM_URL",
    "ISPEC_VLLM_MODEL",
    "ISPEC_VLLM_TIMEOUT_SECONDS",
    "VLLM_HOST",
    "VLLM_PORT",
    "VLLM_DTYPE",
    "VLLM_MAX_MODEL_LEN",
    "VLLM_MAX_NUM_SEQS",
    "VLLM_MAX_NUM_BATCHED_TOKENS",
    "VLLM_GPU_MEMORY_UTILIZATION",
    "VLLM_ENABLE_CHUNKED_PREFILL",
    "VLLM_USE_V1",
    "VLLM_ATTENTION_BACKEND",
    "VLLM_EXTRA_ARGS",
)


def workspace_root() -> Path:
    return Path(__file__).resolve().parents[4]


def backend_root() -> Path:
    return Path(__file__).resolve().parents[3]


def default_scenario_file() -> Path:
    return backend_root() / "benchmarks" / "support_vllm_scenarios.json"


def default_output_dir() -> Path:
    return Path(resolve_log_dir().path or (Path.home() / ".ispec" / "logs")).expanduser() / "benchmarks"


def _parse_env_line(line: str) -> tuple[str, str] | None:
    text = line.strip()
    if not text or text.startswith("#"):
        return None
    if text.startswith("export "):
        text = text[len("export ") :].strip()
    if "=" not in text:
        return None
    key, value = text.split("=", 1)
    key = key.strip()
    if not key:
        return None
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1]
    return key, value


def _normalize_env_value(key: str, value: str, *, base: Path) -> str:
    raw = str(value or '').strip()
    if not raw:
        return str(value)
    if not key.startswith('ISPEC_') or not key.endswith(('_PATH', '_DIR', '_ROOT')):
        return str(value)
    if '://' in raw and not raw.startswith('sqlite'):
        return raw
    if raw.startswith('sqlite:///'):
        suffix = raw.removeprefix('sqlite:///')
        normalized = Path(suffix).expanduser()
        if not normalized.is_absolute():
            normalized = (base / normalized).resolve()
        return 'sqlite:///' + str(normalized)
    if raw.startswith('sqlite://') and not raw.startswith('sqlite:////'):
        suffix = raw.removeprefix('sqlite://')
        normalized = Path(suffix).expanduser()
        if not normalized.is_absolute():
            normalized = (base / normalized).resolve()
        return 'sqlite:///' + str(normalized)
    normalized = Path(raw).expanduser()
    if not normalized.is_absolute():
        normalized = (base / normalized).resolve()
    return str(normalized)


def load_workspace_env(root: str | Path | None = None) -> dict[str, str]:
    base = Path(root).expanduser().resolve() if root is not None else workspace_root()
    values: dict[str, str] = {}
    for relpath in DEFAULT_ENV_FILES:
        path = base / relpath
        if not path.is_file():
            continue
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except Exception:
            continue
        for line in lines:
            parsed = _parse_env_line(line)
            if parsed is None:
                continue
            key, value = parsed
            values[key] = value
    for key, value in os.environ.items():
        values[key] = value
    return values


def apply_loaded_env(env: dict[str, str], root: str | Path | None = None) -> None:
    base = Path(root).expanduser().resolve() if root is not None else workspace_root()
    for key, value in env.items():
        os.environ[str(key)] = _normalize_env_value(str(key), str(value), base=base)


def support_chat_url(env: dict[str, str], override: str | None = None) -> str:
    raw = (override or env.get("ISPEC_API_URL") or "").strip().rstrip("/")
    if not raw:
        port = (env.get("ISPEC_PORT") or "3001").strip() or "3001"
        raw = f"http://127.0.0.1:{port}"
    if raw.endswith("/api/support/chat"):
        return raw
    if raw.endswith("/api"):
        return raw + "/support/chat"
    return raw + "/api/support/chat"


def backend_openapi_url(env: dict[str, str], override: str | None = None) -> str:
    chat_url = support_chat_url(env, override=override)
    if "/api/support/chat" in chat_url:
        return chat_url.rsplit("/api/support/chat", 1)[0] + "/openapi.json"
    return chat_url.rstrip("/") + "/openapi.json"


def vllm_url(env: dict[str, str]) -> str:
    return (env.get("ISPEC_VLLM_URL") or "http://127.0.0.1:8000").strip().rstrip("/")


def api_key(env: dict[str, str], override: str | None = None) -> str | None:
    key = (override or env.get("ISPEC_API_KEY") or env.get("ISPEC_SLACK_API_KEY") or "").strip()
    return key or None


def state_dir(env: dict[str, str]) -> Path:
    raw = (env.get("ISPEC_STATE_DIR") or "").strip()
    if raw:
        return Path(raw).expanduser()
    return workspace_root() / ".pids"


def _json_load(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _http_json_get(*, url: str, timeout_seconds: float) -> Any:
    req = request.Request(url, method="GET")
    try:
        with request.urlopen(req, timeout=max(1.0, timeout_seconds)) as resp:
            data = resp.read()
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"Request failed: {exc.reason}") from exc
    try:
        return json.loads(data.decode("utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Response was not valid JSON: {exc}") from exc


def _json_post(*, url: str, payload: dict[str, Any], api_key_value: str | None, timeout_seconds: float) -> dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key_value:
        headers["X-API-Key"] = api_key_value
    req = request.Request(url, data=body, headers=headers, method="POST")
    try:
        with request.urlopen(req, timeout=max(1.0, timeout_seconds)) as resp:
            data = resp.read()
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"Request failed: {exc.reason}") from exc
    try:
        return json.loads(data.decode("utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Response was not valid JSON: {exc}") from exc


def _render_template_value(value: Any, variables: dict[str, Any]) -> Any:
    if isinstance(value, str):
        return Template(value).safe_substitute({key: str(item) for key, item in variables.items()})
    if isinstance(value, list):
        return [_render_template_value(item, variables) for item in value]
    if isinstance(value, dict):
        return {str(key): _render_template_value(item, variables) for key, item in value.items()}
    return value


def _string_list(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        text = raw.strip()
        return [text] if text else []
    if isinstance(raw, list):
        out: list[str] = []
        for item in raw:
            if isinstance(item, str) and item.strip():
                out.append(item.strip())
        return out
    raise ValueError("expected string or list of strings")


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    rank = max(0, min(len(ordered) - 1, math.ceil((pct / 100.0) * len(ordered)) - 1))
    return ordered[rank]


def _service_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "path": str(path)}
    payload = _json_load(path.read_text(encoding="utf-8"))
    pid = payload.get("pid")
    running = False
    try:
        if isinstance(pid, int) and pid > 0:
            os.kill(pid, 0)
            running = True
    except Exception:
        running = False
    payload["exists"] = True
    payload["path"] = str(path)
    payload["pid_alive"] = running
    return payload


def load_runtime_baseline(env: dict[str, str]) -> dict[str, Any]:
    root = workspace_root()
    state_root = state_dir(env)
    vllm_settings = {
        key: str(env.get(key) or "").strip()
        for key in _VLLM_BASELINE_ENV_KEYS
        if str(env.get(key) or "").strip()
    }
    return {
        "workspace_root": str(root),
        "backend_root": str(backend_root()),
        "assistant_provider": (env.get("ISPEC_ASSISTANT_PROVIDER") or "").strip() or None,
        "assistant_model": (env.get("ISPEC_VLLM_MODEL") or env.get("ISPEC_OLLAMA_MODEL") or None),
        "vllm_url": vllm_url(env),
        "support_chat_url": support_chat_url(env),
        "backend_openapi_url": backend_openapi_url(env),
        "state_dir": str(state_root),
        "api_server": _service_state(state_root / "api_server.json"),
        "supervisor": _service_state(state_root / "supervisor.json"),
        "vllm_settings": vllm_settings,
    }


def assert_benchmark_baseline(*, env: dict[str, str], allow_supervisor_running: bool = False, allow_non_vllm_provider: bool = False, check_vllm_health: bool = True) -> dict[str, Any]:
    baseline = load_runtime_baseline(env)
    provider = str(baseline.get("assistant_provider") or "").strip().lower()
    if not allow_non_vllm_provider and provider != "vllm":
        raise RuntimeError(f"Benchmark baseline requires ISPEC_ASSISTANT_PROVIDER=vllm; found {provider or 'unset'}.")
    api_state = baseline["api_server"]
    try:
        _http_json_get(url=str(baseline.get("backend_openapi_url") or backend_openapi_url(env)), timeout_seconds=10.0)
        baseline["api_http_ok"] = True
    except Exception as exc:
        baseline["api_http_ok"] = False
        baseline["api_http_error"] = str(exc)
    if not bool(api_state.get("pid_alive")) and not bool(baseline.get("api_http_ok")):
        raise RuntimeError("Benchmark baseline requires the backend API server to be running.")
    supervisor_state = baseline["supervisor"]
    if (not allow_supervisor_running and str(supervisor_state.get("status") or "").strip().lower() == "running" and bool(supervisor_state.get("pid_alive"))):
        raise RuntimeError("Benchmark baseline expects a low-churn run with supervisor stopped or disabled. Stop the supervisor or rerun with --allow-supervisor-running.")
    if check_vllm_health:
        try:
            parsed = _http_json_get(url=vllm_url(env) + "/v1/models", timeout_seconds=10.0)
        except Exception as exc:
            raise RuntimeError(f"Could not reach vLLM at {vllm_url(env)}: {exc}") from exc
        baseline["vllm_models"] = parsed.get("data") if isinstance(parsed, dict) else None
    return baseline


def load_synthetic_benchmark_scenarios(*, path: str | Path, variables: dict[str, Any] | None = None, labels: set[str] | None = None, families: set[str] | None = None) -> list[dict[str, Any]]:
    scenario_path = Path(path).expanduser().resolve()
    payload = json.loads(scenario_path.read_text(encoding="utf-8"))
    scenarios_raw = payload.get("scenarios") if isinstance(payload, dict) else None
    if not isinstance(scenarios_raw, list):
        raise ValueError(f"{scenario_path}: expected a top-level object with a scenarios list")
    render_vars = dict(variables or {})
    loaded: list[dict[str, Any]] = []
    for item in scenarios_raw:
        if not isinstance(item, dict):
            raise ValueError(f"{scenario_path}: each scenario must be an object")
        rendered = _render_template_value(item, render_vars)
        label = str(rendered.get("label") or "").strip()
        family = str(rendered.get("family") or "").strip()
        if not label or not family:
            raise ValueError(f"{scenario_path}: scenario missing label/family")
        if labels and label not in labels:
            continue
        if families and family not in families:
            continue
        turns = rendered.get("turns")
        if not isinstance(turns, list) or not turns:
            raise ValueError(f"{scenario_path}: scenario {label!r} must have a non-empty turns list")
        normalized_turns: list[dict[str, Any]] = []
        for index, turn in enumerate(turns, start=1):
            if not isinstance(turn, dict):
                raise ValueError(f"{scenario_path}: scenario {label!r} turn #{index} must be an object")
            message = str(turn.get("message") or "").strip()
            if not message:
                raise ValueError(f"{scenario_path}: scenario {label!r} turn #{index} missing message")
            normalized_turns.append({"message": message, "expect": turn.get("expect") if isinstance(turn.get("expect"), dict) else None})
        loaded.append({
            "label": label,
            "family": family,
            "tags": _string_list(rendered.get("tags")),
            "notes": str(rendered.get("notes") or "").strip() or None,
            "source_kind": "synthetic",
            "source_path": str(scenario_path),
            "turns": normalized_turns,
        })
    return loaded


def _validate_local_case_messages(messages: Any, *, case_path: Path) -> list[dict[str, Any]]:
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"{case_path}: messages must be a non-empty list")
    out: list[dict[str, Any]] = []
    for index, item in enumerate(messages, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"{case_path}: message #{index} must be an object")
        role = str(item.get("role") or "").strip()
        content = str(item.get("content") or "")
        if role not in {"user", "assistant", "system"}:
            raise ValueError(f"{case_path}: message #{index} invalid role {role!r}")
        normalized = {"role": role, "content": content}
        if isinstance(item.get("assistant_meta"), dict):
            normalized["assistant_meta"] = item["assistant_meta"]
        out.append(normalized)
    return out


def load_local_benchmark_scenarios(*, case_selectors: list[str], root: str | Path | None = None) -> list[dict[str, Any]]:
    case_dir = Path(root).expanduser().resolve() if root is not None else backend_root() / "tests" / "behavioral" / "local"
    if not case_dir.exists():
        raise ValueError(f"Local case directory does not exist: {case_dir}")
    available = sorted(case_dir.glob("*.json"))
    selected_paths: list[Path] = []
    for selector in case_selectors:
        cleaned = str(selector or "").strip()
        if not cleaned:
            continue
        direct = Path(cleaned).expanduser()
        if direct.is_file():
            selected_paths.append(direct.resolve())
            continue
        matches = [path for path in available if cleaned in path.stem or cleaned == path.name]
        if not matches:
            raise ValueError(f"No local behavioral case matched selector {cleaned!r}")
        selected_paths.extend(matches)
    seen: set[str] = set()
    scenarios: list[dict[str, Any]] = []
    for case_path in selected_paths:
        if str(case_path) in seen:
            continue
        seen.add(str(case_path))
        payload = json.loads(case_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"{case_path}: case must be a JSON object")
        label = str(payload.get("label") or case_path.stem).strip()
        tags = _string_list(payload.get("tags"))
        messages = _validate_local_case_messages(payload.get("messages"), case_path=case_path)
        user_turns = [{"message": str(item.get("content") or ""), "expect": None} for item in messages if item.get("role") == "user"]
        if not user_turns:
            raise ValueError(f"{case_path}: case contains no user turns to replay")
        family = tags[0] if tags else "local"
        scenarios.append({
            "label": label,
            "family": family,
            "tags": tags,
            "notes": str(payload.get("notes") or "").strip() or None,
            "source_kind": "local_case",
            "source_path": str(case_path),
            "turns": user_turns,
            "final_expect": payload.get("expect") if isinstance(payload.get("expect"), dict) else None,
        })
    return scenarios


def _snapshot_message(row: SupportMessage) -> SimpleNamespace:
    return SimpleNamespace(
        id=int(getattr(row, 'id')),
        session_pk=int(getattr(row, 'session_pk')),
        role=str(getattr(row, 'role', '') or ''),
        content=str(getattr(row, 'content', '') or ''),
        created_at=getattr(row, 'created_at', None),
        provider=getattr(row, 'provider', None),
        model=getattr(row, 'model', None),
        meta_json=getattr(row, 'meta_json', None),
    )


def _snapshot_session(row: SupportSession) -> SimpleNamespace:
    return SimpleNamespace(
        id=int(getattr(row, 'id')),
        session_id=str(getattr(row, 'session_id', '') or ''),
        user_id=getattr(row, 'user_id', None),
        created_at=getattr(row, 'created_at', None),
        updated_at=getattr(row, 'updated_at', None),
        state_json=getattr(row, 'state_json', None),
    )


def assistant_session_rows(session_id: str) -> tuple[list[SimpleNamespace], SimpleNamespace | None]:
    with get_assistant_session() as db:
        session = db.query(SupportSession).filter(SupportSession.session_id == str(session_id)).first()
        if session is None:
            return [], None
        session_snapshot = _snapshot_session(session)
        messages = db.query(SupportMessage).filter(SupportMessage.session_pk == int(session.id)).order_by(SupportMessage.id.asc()).all()
        message_snapshots = [_snapshot_message(row) for row in messages]
        return message_snapshots, session_snapshot


def latest_assistant_message(session_id: str) -> SimpleNamespace | None:
    messages, _session = assistant_session_rows(session_id)
    assistant_messages = [row for row in messages if str(row.role or "") == "assistant"]
    return assistant_messages[-1] if assistant_messages else None


def wait_for_latest_assistant_message(*, session_id: str, min_message_id: int = 0, timeout_seconds: float = 15.0) -> SimpleNamespace:
    deadline = time.monotonic() + max(1.0, timeout_seconds)
    last_seen: SimpleNamespace | None = None
    while time.monotonic() < deadline:
        row = latest_assistant_message(session_id)
        if row is not None:
            last_seen = row
            if int(row.id) > int(min_message_id):
                return row
        time.sleep(0.1)
    if last_seen is not None:
        return last_seen
    raise RuntimeError(f"No assistant message found for session {session_id!r}")


def assert_case_expectations(case: dict[str, Any], expect: dict[str, Any] | None) -> None:
    if not isinstance(expect, dict) or not expect:
        return
    messages = case.get("messages") if isinstance(case.get("messages"), list) else []
    assistant_messages = [item for item in messages if isinstance(item, dict) and item.get("role") == "assistant"]
    final_assistant = assistant_messages[-1] if assistant_messages else {}
    final_text = str(final_assistant.get("content") or "")
    final_meta = final_assistant.get("assistant_meta") if isinstance(final_assistant.get("assistant_meta"), dict) else {}
    response_contract = final_meta.get("response_contract") if isinstance(final_meta.get("response_contract"), dict) else {}
    reply_interpretation = final_meta.get("reply_interpretation") if isinstance(final_meta.get("reply_interpretation"), dict) else {}
    tool_calls = final_meta.get("tool_calls") if isinstance(final_meta.get("tool_calls"), list) else []
    if "assistant_message_count" in expect:
        assert len(assistant_messages) == int(expect["assistant_message_count"])
    for needle in _string_list(expect.get("final_assistant_contains")):
        assert needle in final_text
    for needle in _string_list(expect.get("final_assistant_not_contains")):
        assert needle not in final_text
    if "response_contract_mode" in expect:
        assert response_contract.get("configured_mode") == expect["response_contract_mode"]
    if "response_contract_selected_contract" in expect:
        assert response_contract.get("selected_contract") == expect["response_contract_selected_contract"]
    for needle in _string_list(expect.get("response_contract_shadow_candidate_contains")):
        shadow_text = str(response_contract.get("shadow_candidate") or "")
        assert needle in shadow_text
    if "reply_interpretation_runtime_kind" in expect:
        assert reply_interpretation.get("runtime_kind") == expect["reply_interpretation_runtime_kind"]
    if "reply_interpretation_runtime_action" in expect:
        assert reply_interpretation.get("runtime_action") == expect["reply_interpretation_runtime_action"]
    if "tool_call_names_include" in expect:
        names = {str(item.get("name") or "").strip() for item in tool_calls if isinstance(item, dict) and str(item.get("name") or "").strip()}
        for name in _string_list(expect.get("tool_call_names_include")):
            assert name in names


def collect_turn_metrics(*, session_id: str, assistant_row: SupportMessage, wall_clock_ms: int) -> dict[str, Any]:
    meta = _json_load(getattr(assistant_row, "meta_json", None))
    llm_trace = meta.get("llm_trace") if isinstance(meta.get("llm_trace"), list) else []
    tool_calls = meta.get("tool_calls") if isinstance(meta.get("tool_calls"), list) else []
    provider_meta_items = [item.get("provider_meta") for item in llm_trace if isinstance(item, dict) and isinstance(item.get("provider_meta"), dict)]
    elapsed_values = [float(item.get("elapsed_ms")) for item in provider_meta_items if isinstance(item.get("elapsed_ms"), (int, float))]
    prompt_stages = [str(item.get("prompt") or "").strip() for item in llm_trace if isinstance(item, dict) and str(item.get("prompt") or "").strip()]
    prompt_families = [str(item.get("prompt_family") or "").strip() for item in llm_trace if isinstance(item, dict) and str(item.get("prompt_family") or "").strip()]
    parser_fallback_used = any(bool(item.get("tool_parser_fallback_used")) for item in provider_meta_items if isinstance(item, dict))
    return {
        "assistant_message_id": int(assistant_row.id),
        "wall_clock_ms": int(wall_clock_ms),
        "assistant_message_chars": len(str(assistant_row.content or "")),
        "model_call_count": len(llm_trace),
        "llm_round_count": max([int(item.get("round") or 0) for item in llm_trace if isinstance(item, dict)], default=0),
        "tool_call_count": len(tool_calls),
        "used_tool_calls": (meta.get("tooling", {}).get("used_tool_calls") if isinstance(meta.get("tooling"), dict) else None),
        "model_elapsed_ms_total": round(sum(elapsed_values), 2) if elapsed_values else 0.0,
        "model_elapsed_ms_max": round(max(elapsed_values), 2) if elapsed_values else 0.0,
        "model_elapsed_ms_values": [round(value, 2) for value in elapsed_values],
        "prompt_stages": prompt_stages,
        "prompt_families": prompt_families,
        "tool_names": [str(item.get("name") or "").strip() for item in tool_calls if isinstance(item, dict) and str(item.get("name") or "").strip()],
        "parser_fallback_used": parser_fallback_used,
        "controller_rule": (meta.get("controller", {}).get("selected_rule_name") if isinstance(meta.get("controller"), dict) else None),
        "response_contract_mode": (meta.get("response_contract", {}).get("configured_mode") if isinstance(meta.get("response_contract"), dict) else None),
        "reply_interpretation_action": (meta.get("reply_interpretation", {}).get("runtime_action") if isinstance(meta.get("reply_interpretation"), dict) else None),
        "session_id": session_id,
        "assistant_text": str(assistant_row.content or ""),
        "assistant_meta": meta,
    }


def summarize_benchmark_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    families: dict[str, dict[str, Any]] = {}
    total_turns = 0
    successful_turns = 0
    for scenario in results:
        family = str(scenario.get("family") or "unknown")
        bucket = families.setdefault(family, {
            "scenario_count": 0,
            "turn_count": 0,
            "successful_turns": 0,
            "wall_clock_ms": [],
            "model_elapsed_ms_total": [],
            "model_call_count": [],
            "tool_call_count": [],
        })
        bucket["scenario_count"] += 1
        turns = scenario.get("turns") if isinstance(scenario.get("turns"), list) else []
        for turn in turns:
            if not isinstance(turn, dict):
                continue
            total_turns += 1
            bucket["turn_count"] += 1
            if bool(turn.get("ok")):
                successful_turns += 1
                bucket["successful_turns"] += 1
            if isinstance(turn.get("wall_clock_ms"), (int, float)):
                bucket["wall_clock_ms"].append(float(turn["wall_clock_ms"]))
            if isinstance(turn.get("model_elapsed_ms_total"), (int, float)):
                bucket["model_elapsed_ms_total"].append(float(turn["model_elapsed_ms_total"]))
            if isinstance(turn.get("model_call_count"), (int, float)):
                bucket["model_call_count"].append(float(turn["model_call_count"]))
            if isinstance(turn.get("tool_call_count"), (int, float)):
                bucket["tool_call_count"].append(float(turn["tool_call_count"]))
    family_summary: dict[str, Any] = {}
    for family, bucket in sorted(families.items()):
        wall_values = bucket.pop("wall_clock_ms")
        model_elapsed_values = bucket.pop("model_elapsed_ms_total")
        model_call_values = bucket.pop("model_call_count")
        tool_call_values = bucket.pop("tool_call_count")
        family_summary[family] = {
            **bucket,
            "wall_clock_ms_p50": _percentile(wall_values, 50),
            "wall_clock_ms_p95": _percentile(wall_values, 95),
            "wall_clock_ms_avg": (sum(wall_values) / len(wall_values)) if wall_values else None,
            "model_elapsed_ms_total_avg": (sum(model_elapsed_values) / len(model_elapsed_values)) if model_elapsed_values else None,
            "model_call_count_avg": (sum(model_call_values) / len(model_call_values)) if model_call_values else None,
            "tool_call_count_avg": (sum(tool_call_values) / len(tool_call_values)) if tool_call_values else None,
        }
    return {
        "scenario_count": len(results),
        "turn_count": total_turns,
        "successful_turns": successful_turns,
        "family_summary": family_summary,
    }


def _format_float(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "-"
    return f"{float(value):.0f}"


def format_benchmark_summary(summary: dict[str, Any]) -> str:
    lines = [
        f"Scenarios: {int(summary.get('scenario_count') or 0)}",
        f"Turns: {int(summary.get('turn_count') or 0)}",
        f"Successful turns: {int(summary.get('successful_turns') or 0)}",
        "",
        "Family	Scenarios	Turns	Success	Wall p50	Wall p95	Model avg	Calls avg	Tools avg",
    ]
    family_summary = summary.get("family_summary") if isinstance(summary.get("family_summary"), dict) else {}
    for family, item in sorted(family_summary.items()):
        lines.append("	".join([
            family,
            str(int(item.get("scenario_count") or 0)),
            str(int(item.get("turn_count") or 0)),
            str(int(item.get("successful_turns") or 0)),
            _format_float(item.get("wall_clock_ms_p50")),
            _format_float(item.get("wall_clock_ms_p95")),
            _format_float(item.get("model_elapsed_ms_total_avg")),
            _format_float(item.get("model_call_count_avg")),
            _format_float(item.get("tool_call_count_avg")),
        ]))
    return "\n".join(lines)


def benchmark_output_path(*, run_label: str) -> Path:
    safe_label = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in str(run_label or "").strip()).strip("-") or "run"
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return default_output_dir() / f"support-vllm-benchmark-{safe_label}-{timestamp}.json"
