from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import threading
import time
import uuid
from collections.abc import Generator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Callable
from zoneinfo import ZoneInfo

import psutil
import requests
from sqlalchemy.orm.attributes import flag_modified

from ispec.agent.connect import get_agent_db_uri, get_agent_session
from ispec.agent.commands import (
    COMMAND_ARCHIVE_AGENT_LOGS,
    COMMAND_COMPACT_SESSION_MEMORY,
    COMMAND_BUILD_SUPPORT_DIGEST,
    COMMAND_POST_SEND_PREPARE,
    COMMAND_RUN_SCHEDULED_ASSISTANT_PROMPT,
    COMMAND_ASSESS_TACKLE_RESULTS,
    COMMAND_RUN_TACKLE_PROMPT,
    COMMAND_DEV_RESTART_SERVICES,
    COMMAND_LEGACY_PUSH_PROJECT_COMMENTS,
    COMMAND_LEGACY_SYNC_ALL,
    COMMAND_ORCHESTRATOR_TICK,
    COMMAND_REVIEW_REPO,
    COMMAND_REVIEW_SUPPORT_SESSION,
    COMMAND_SLACK_POST_MESSAGE,
    COMMAND_SUPPORT_CHAT_TURN,
)
from ispec.agent.models import AgentCommand, AgentEvent, AgentRun, AgentStep
from ispec.agent.policies.primitives.backoff import backoff_exponential_current
from ispec.assistant.compaction import distill_conversation_memory
from ispec.assistant.connect import get_assistant_db_uri, get_assistant_session
from ispec.assistant.controller import enqueue_post_send_prepare_command, support_post_send_thread_key
from ispec.assistant.formatting import split_plan_final
from ispec.assistant.models import (
    SupportMemory,
    SupportMemoryEvidence,
    SupportMessage,
    SupportSession,
    SupportSessionReview,
)
from ispec.assistant.prompting import estimate_tokens_for_messages
from ispec.assistant.response_contracts import response_contract_names
from ispec.assistant.schedules import (
    AssistantSchedule,
    load_assistant_schedules as _load_assistant_schedules,
    normalize_schedule_tool_names as _normalize_schedule_tool_names,
    parse_hhmm as _parse_hhmm,
    parse_weekday as _parse_weekday,
)
from ispec.assistant.service import AssistantReply, _system_prompt_planner, generate_reply
from ispec.assistant.tool_routing import tool_groups_for_available_tools
from ispec.assistant.turn_decision import (
    parse_turn_decision_mode,
    run_turn_decision_pipeline,
    selected_tool_names_from_decision,
)
from ispec.prompt import load_bound_prompt, prompt_binding, prompt_observability_context
from ispec.assistant.tools import (
    TOOL_CALL_PREFIX,
    extract_tool_call_line,
    format_tool_result_message,
    openai_tools_for_user,
    parse_tool_call,
    run_tool,
)
from ispec.concurrency.thread_context import assert_main_thread, main_thread_info, set_main_thread
from ispec.config.paths import (
    resolve_db_location,
    resolve_state_dir,
    resolve_supervisor_pid_file,
    resolve_supervisor_state_file,
)
from ispec.db.connect import get_session
from ispec.db.models import UserRole
from ispec.logging import get_logger
from ispec.omics.connect import get_omics_session
from ispec.schedule.connect import get_schedule_db_uri, get_schedule_session
from ispec.supervisor.inference_broker import InferenceBroker, InferenceRequest

logger = get_logger(__file__)

_DEV_RESTART_ENABLED_ENV = "ISPEC_DEV_RESTART_ENABLED"
_INFERENCE_BROKER_ENABLED_ENV = "ISPEC_SUPERVISOR_INFERENCE_BROKER_ENABLED"


def _supervisor_state_file_path() -> Path:
    resolved = resolve_supervisor_state_file()
    return Path(resolved.path or resolved.value)


def _supervisor_pid_file_path() -> Path:
    resolved = resolve_supervisor_pid_file()
    return Path(resolved.path or resolved.value)


def _write_supervisor_state(payload: dict[str, Any]) -> Path | None:
    path = _supervisor_state_file_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        return path
    except OSError as exc:
        logger.warning("Unable to write supervisor state file %s: %s", path, exc)
        return None


def _write_supervisor_pid() -> Path | None:
    path = _supervisor_pid_file_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        # Keep this file intentionally tiny and human-grep-able.
        path.write_text(f"{os.getpid()}\n")
        return path
    except OSError as exc:
        logger.warning("Unable to write supervisor pid file %s: %s", path, exc)
        return None


def _remove_supervisor_pid(path: Path | None = None) -> None:
    if path is None:
        path = _supervisor_pid_file_path()
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    except OSError as exc:
        logger.warning("Unable to remove supervisor pid file %s: %s", path, exc)


def utcnow() -> datetime:
    return datetime.now(UTC)


_TRUTHY = {"1", "true", "yes", "y", "on"}
_FALSY = {"0", "false", "no", "n", "off"}


def _is_truthy(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in _TRUTHY


def _parse_env_tristate_bool(raw: str | None, *, key: str) -> tuple[bool | None, str | None]:
    """Parse a tri-state boolean env var.

    Returns (value, error):
    - value=True/False when explicitly set
    - value=None when unset/empty/"auto"
    - error is non-null when the value is present but invalid
    """

    if raw is None:
        return None, None
    text = str(raw).strip()
    if not text:
        return None, None
    lowered = text.lower()
    if lowered == "auto":
        return None, None
    if lowered in _TRUTHY:
        return True, None
    if lowered in _FALSY:
        return False, None
    return None, f"Invalid value for {key}: {text!r} (expected 0/1/true/false or unset/auto)."


def _state_dir_is_dev() -> bool:
    resolved = resolve_state_dir()
    raw = (resolved.path or resolved.value or "").strip()
    if not raw:
        return False
    try:
        path = Path(raw).expanduser().resolve()
    except Exception:
        return False
    return path.name == ".pids"


def _assistant_turn_decision_mode() -> str:
    provider = (os.getenv("ISPEC_ASSISTANT_PROVIDER") or "").strip().lower()
    return parse_turn_decision_mode(
        os.getenv("ISPEC_ASSISTANT_TURN_DECISION_MODE"),
        auto_shadow=_state_dir_is_dev() and provider == "vllm",
    )


def _inference_broker_enabled_status() -> tuple[bool, str | None]:
    raw = os.getenv(_INFERENCE_BROKER_ENABLED_ENV)
    parsed, err = _parse_env_tristate_bool(raw, key=_INFERENCE_BROKER_ENABLED_ENV)
    if err:
        return False, err
    if parsed is True:
        return True, None
    if parsed is False:
        return False, f"{_INFERENCE_BROKER_ENABLED_ENV}=0 (forced off)."
    # Auto: enable in dev (.pids) layouts so the supervisor stays responsive
    # while local model inference is in-flight.
    return _state_dir_is_dev(), None


def _dev_restart_auto_enabled(*, tmux_session: str | None = None, make_root: str | None = None) -> tuple[bool, str | None]:
    if shutil.which("tmux") is None:
        return False, "tmux is not installed."

    try:
        from ispec.cli import dev as dev_cli
    except Exception as exc:
        return False, f"Failed importing ispec.cli.dev ({type(exc).__name__})."

    make_root_path: Path | None = None
    if make_root:
        try:
            make_root_path = Path(make_root).expanduser().resolve()
        except Exception:
            make_root_path = None
    if make_root_path is None:
        make_root_path = dev_cli._find_make_root(start=Path(__file__).resolve().parent)  # type: ignore[attr-defined]
    if make_root_path is None:
        return False, "Top-level Makefile not found (run from within the ispec-full repo or pass make_root)."

    state_dir = (os.getenv("ISPEC_STATE_DIR") or "").strip()
    state_dir_path: Path | None = None
    if state_dir:
        try:
            state_dir_path = Path(state_dir).expanduser().resolve()
        except Exception:
            state_dir_path = None
    state_dir_is_dev = bool(
        state_dir_path is not None
        and (state_dir_path.name == ".pids" or state_dir_path == (make_root_path / ".pids").resolve())
    )

    session = dev_cli._tmux_session_name(tmux_session)  # type: ignore[attr-defined]
    tmux_session_exists = dev_cli._tmux_has_session(session)  # type: ignore[attr-defined]

    if not state_dir_is_dev and not tmux_session_exists:
        return (
            False,
            "Auto-detect did not find a dev tmux session (set DEV_TMUX_SESSION) and ISPEC_STATE_DIR is not .pids.",
        )

    return True, None


def _dev_restart_enabled_status(*, tmux_session: str | None = None, make_root: str | None = None) -> tuple[bool, str | None]:
    raw = os.getenv(_DEV_RESTART_ENABLED_ENV)
    parsed, err = _parse_env_tristate_bool(raw, key=_DEV_RESTART_ENABLED_ENV)
    if err:
        return False, err
    if parsed is True:
        return True, None
    if parsed is False:
        return False, f"{_DEV_RESTART_ENABLED_ENV}=0 (forced off)."
    return _dev_restart_auto_enabled(tmux_session=tmux_session, make_root=make_root)


def _clamp_int(value: int, *, min_value: int, max_value: int) -> int:
    if value < min_value:
        return min_value
    if value > max_value:
        return max_value
    return value


def _truncate_text(value: str | None, *, limit: int = 400) -> str:
    text = (value or "").strip()
    if not text or limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def _summarize_command_payload(command_type: str, payload: dict[str, Any]) -> str:
    if not isinstance(payload, dict) or not payload:
        return ""

    parts: list[str] = []
    if command_type == COMMAND_DEV_RESTART_SERVICES:
        services = payload.get("services")
        if isinstance(services, list):
            cleaned = [str(item).strip() for item in services if isinstance(item, str) and item.strip()]
            if cleaned:
                parts.append("services=" + ",".join(cleaned[:6]))
        elif isinstance(services, str) and services.strip():
            parts.append(f"services={services.strip()[:80]}")
    if command_type == COMMAND_SUPPORT_CHAT_TURN:
        chat_request = payload.get("chat_request")
        if isinstance(chat_request, dict):
            queued_session_id = chat_request.get("sessionId")
            if isinstance(queued_session_id, str) and queued_session_id.strip():
                parts.append(f"session_id={queued_session_id.strip()}")
    if command_type == COMMAND_RUN_SCHEDULED_ASSISTANT_PROMPT:
        job = payload.get("job")
        if isinstance(job, dict):
            job_name = job.get("name")
            if isinstance(job_name, str) and job_name.strip():
                parts.append(f"job={job_name.strip()}")
    if command_type == COMMAND_ARCHIVE_AGENT_LOGS:
        older_than_days = _safe_int(payload.get("older_than_days"))
        if older_than_days is not None:
            parts.append(f"older_than_days={older_than_days}")
        batch_size = _safe_int(payload.get("batch_size"))
        if batch_size is not None:
            parts.append(f"batch_size={batch_size}")

    session_id = payload.get("session_id")
    if isinstance(session_id, str) and session_id.strip():
        parts.append(f"session_id={session_id.strip()}")

    target_message_id = payload.get("target_message_id")
    if isinstance(target_message_id, int) and target_message_id > 0:
        parts.append(f"target_message_id={target_message_id}")

    project_id = payload.get("project_id")
    if isinstance(project_id, int) and project_id > 0:
        parts.append(f"project_id={project_id}")

    schedule = payload.get("schedule")
    if isinstance(schedule, dict):
        schedule_name = schedule.get("name")
        if isinstance(schedule_name, str) and schedule_name.strip():
            parts.append(f"schedule={schedule_name.strip()}")
        schedule_key = schedule.get("key")
        if isinstance(schedule_key, str) and schedule_key.strip():
            parts.append(f"schedule_key={schedule_key.strip()}")

    channel = payload.get("channel")
    if isinstance(channel, str) and channel.strip():
        parts.append(f"channel={channel.strip()}")

    return " ".join(parts[:6])


def _parse_json_object(text: str | None) -> dict[str, Any] | None:
    if not text:
        return None
    raw = str(text).strip()
    if not raw:
        return None
    def maybe_repair(candidate: str) -> dict[str, Any] | None:
        try:
            parsed = json.loads(candidate)
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            pass

        # Some models miss a final closing brace/bracket. Attempt a tiny repair
        # pass that only appends closers when strings are properly terminated.
        stack: list[str] = []
        in_string = False
        escape = False
        for ch in candidate:
            if in_string:
                if escape:
                    escape = False
                    continue
                if ch == "\\":
                    escape = True
                    continue
                if ch == "\"":
                    in_string = False
                continue

            if ch == "\"":
                in_string = True
                continue

            if ch in "{[":
                stack.append(ch)
                continue

            if ch in "}]":
                if not stack:
                    continue
                opener = stack[-1]
                if (ch == "}" and opener == "{") or (ch == "]" and opener == "["):
                    stack.pop()
                continue

        if in_string or not stack:
            return None

        closers: list[str] = []
        while stack and len(closers) < 8:
            opener = stack.pop()
            closers.append("}" if opener == "{" else "]")

        repaired = candidate + "".join(closers)
        try:
            parsed = json.loads(repaired)
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            return None

    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
        # Some guided decoding responses arrive as a JSON string that itself
        # contains a serialized object (escaped quotes). Attempt one more pass.
        if isinstance(parsed, str):
            inner = parsed.strip()
            if inner.startswith("{") and inner.endswith("}"):
                repaired = maybe_repair(inner)
                if repaired is not None:
                    return repaired
    except Exception:
        repaired = maybe_repair(raw)
        if repaired is not None:
            return repaired

    start = raw.find("{")
    end = raw.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        parsed = json.loads(raw[start : end + 1])
        if isinstance(parsed, dict):
            return parsed
        if isinstance(parsed, str):
            inner = parsed.strip()
            if inner.startswith("{") and inner.endswith("}"):
                repaired = maybe_repair(inner)
                if repaired is not None:
                    return repaired
        repaired = maybe_repair(raw[start : end + 1])
        if repaired is not None:
            return repaired
        return None
    except Exception:
        repaired = maybe_repair(raw[start : end + 1])
        if repaired is not None:
            return repaired
        return None


def _sqlite_path_from_uri(db_uri: str | None) -> Path | None:
    if not db_uri:
        return None
    raw = str(db_uri).strip()
    if not raw:
        return None
    if raw.startswith("sqlite:///"):
        return Path(raw.removeprefix("sqlite:///")).expanduser()
    if raw.startswith("sqlite://"):
        return Path(raw.removeprefix("sqlite://")).expanduser()
    if "://" in raw:
        return None
    return Path(raw).expanduser()


def _stat_path(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"path": None, "exists": False}
    try:
        resolved = str(path.expanduser().resolve())
    except Exception:
        resolved = str(path)
    try:
        stat = path.stat()
    except FileNotFoundError:
        return {"path": resolved, "exists": False}
    except Exception as exc:
        return {"path": resolved, "exists": False, "error": f"{type(exc).__name__}: {exc}"}
    return {
        "path": resolved,
        "exists": True,
        "size_bytes": int(stat.st_size),
        "mtime": datetime.fromtimestamp(stat.st_mtime, tz=UTC).isoformat(),
    }


def _probe_http(url: str, *, timeout_seconds: float) -> dict[str, Any]:
    started = time.monotonic()
    try:
        resp = requests.get(url, timeout=timeout_seconds)
        elapsed_ms = int((time.monotonic() - started) * 1000)
        result: dict[str, Any] = {
            "ok": 200 <= resp.status_code < 400,
            "url": url,
            "status_code": int(resp.status_code),
            "elapsed_ms": elapsed_ms,
        }
        content_type = (resp.headers.get("content-type") or "").lower()
        if "application/json" in content_type:
            try:
                result["json"] = resp.json()
            except Exception:
                pass
        return result
    except Exception as exc:
        elapsed_ms = int((time.monotonic() - started) * 1000)
        return {
            "ok": False,
            "url": url,
            "elapsed_ms": elapsed_ms,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _check_ispec_backend(*, base_url: str, timeout_seconds: float) -> dict[str, Any]:
    url = base_url.rstrip("/") + "/status"
    result = _probe_http(url, timeout_seconds=timeout_seconds)
    payload = result.get("json")
    if isinstance(payload, dict) and payload.get("ok") is True:
        result["ok"] = True
    return result


def _check_frontend(*, url: str, timeout_seconds: float) -> dict[str, Any]:
    return _probe_http(url, timeout_seconds=timeout_seconds)


def _check_db_files() -> dict[str, Any]:
    core = resolve_db_location("core")
    analysis = resolve_db_location("analysis")
    psm = resolve_db_location("psm")
    assistant_uri = get_assistant_db_uri()
    schedule_uri = get_schedule_db_uri()
    agent_uri = get_agent_db_uri()
    return {
        "ok": True,
        "core_db": _stat_path(_sqlite_path_from_uri(core.uri or core.value)),
        "analysis_db": _stat_path(_sqlite_path_from_uri(analysis.uri or analysis.value)),
        "psm_db": _stat_path(_sqlite_path_from_uri(psm.uri or psm.value)),
        "assistant_db": _stat_path(_sqlite_path_from_uri(assistant_uri)),
        "schedule_db": _stat_path(_sqlite_path_from_uri(schedule_uri)),
        "agent_db": _stat_path(_sqlite_path_from_uri(agent_uri)),
    }


def _diskwatcher_config_json() -> dict[str, Any] | None:
    try:
        completed = subprocess.run(
            ["diskwatcher", "--log-level", "critical", "config", "show", "--json"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except FileNotFoundError:
        return None
    except Exception:
        return None

    raw = (completed.stdout or "").strip()
    if not raw:
        return None
    idx = raw.find("{")
    if idx < 0:
        return None
    try:
        return json.loads(raw[idx:])
    except Exception:
        return None


def _check_diskwatcher() -> dict[str, Any]:
    config = _diskwatcher_config_json()
    if config is None:
        return {"ok": False, "available": False}

    db_file = None
    paths = config.get("paths") if isinstance(config, dict) else None
    if isinstance(paths, dict):
        candidate = paths.get("database_file")
        if isinstance(candidate, str) and candidate.strip():
            db_file = candidate.strip()

    return {
        "ok": True,
        "available": True,
        "database": _stat_path(Path(db_file)) if db_file else {"path": None, "exists": False},
    }


def _check_system_metrics() -> dict[str, Any]:
    vm = psutil.virtual_memory()
    return {
        "ok": True,
        "cpu_percent": float(psutil.cpu_percent(interval=0.2)),
        "mem_total_bytes": int(vm.total),
        "mem_used_bytes": int(vm.used),
        "mem_available_bytes": int(vm.available),
        "mem_percent": float(vm.percent),
    }


def _check_nvidia_smi() -> dict[str, Any]:
    fields = [
        "index",
        "name",
        "uuid",
        "utilization.gpu",
        "utilization.memory",
        "memory.total",
        "memory.used",
        "temperature.gpu",
        "power.draw",
    ]
    cmd = [
        "nvidia-smi",
        f"--query-gpu={','.join(fields)}",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except FileNotFoundError:
        return {"ok": False, "available": False}
    except Exception as exc:
        return {"ok": False, "available": False, "error": f"{type(exc).__name__}: {exc}"}

    if completed.returncode != 0:
        return {
            "ok": False,
            "available": True,
            "error": (completed.stderr or "").strip() or f"nvidia-smi exited {completed.returncode}",
        }

    gpus: list[dict[str, Any]] = []
    for line in (completed.stdout or "").splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != len(fields):
            continue
        row: dict[str, Any] = {}
        for key, value in zip(fields, parts):
            row[key] = value
        gpus.append(row)

    return {"ok": True, "available": True, "gpus": gpus}


def _load_json_dict(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _dump_json(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, str):
        text = value.strip()
        if text.isdigit():
            parsed = int(text)
            return parsed if parsed >= 0 else None
    return None


def _clamp_float(value: float, *, min_value: float, max_value: float) -> float:
    if value < min_value:
        return min_value
    if value > max_value:
        return max_value
    return value


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None
    return None


def _structured_llm_temperature(task: str, *, default: float) -> float:
    global_default = _safe_float(os.getenv("ISPEC_SUPERVISOR_STRUCTURED_TEMPERATURE"))
    value = _safe_float(os.getenv(f"ISPEC_SUPERVISOR_{task}_TEMPERATURE"))
    if value is None:
        value = global_default if global_default is not None else float(default)
    return _clamp_float(float(value), min_value=0.0, max_value=2.0)


def _structured_llm_repair_temperature(task: str, *, default: float) -> float:
    global_default = _safe_float(os.getenv("ISPEC_SUPERVISOR_STRUCTURED_REPAIR_TEMPERATURE"))
    value = _safe_float(os.getenv(f"ISPEC_SUPERVISOR_{task}_REPAIR_TEMPERATURE"))
    if value is None:
        value = global_default if global_default is not None else float(default)
    return _clamp_float(float(value), min_value=0.0, max_value=2.0)


def _structured_llm_max_repairs(task: str, *, default: int) -> int:
    global_default = _safe_int(os.getenv("ISPEC_SUPERVISOR_STRUCTURED_MAX_REPAIR_ATTEMPTS"))
    value = _safe_int(os.getenv(f"ISPEC_SUPERVISOR_{task}_MAX_REPAIR_ATTEMPTS"))
    if value is None:
        value = global_default if global_default is not None else int(default)
    return _clamp_int(int(value), min_value=0, max_value=3)


def _orchestrator_review_failure_cooldown_seconds() -> int:
    raw = (os.getenv("ISPEC_ORCHESTRATOR_REVIEW_FAILURE_COOLDOWN_SECONDS") or "").strip()
    if not raw:
        return 300
    try:
        return _clamp_int(int(raw), min_value=0, max_value=86400)
    except ValueError:
        return 300


def _orchestrator_review_dedupe_lookback() -> int:
    raw = (os.getenv("ISPEC_ORCHESTRATOR_REVIEW_DEDUPE_LOOKBACK") or "").strip()
    if not raw:
        return 200
    try:
        return _clamp_int(int(raw), min_value=20, max_value=5000)
    except ValueError:
        return 200


def _orchestrator_digest_failure_cooldown_seconds() -> int:
    raw = (os.getenv("ISPEC_ORCHESTRATOR_DIGEST_FAILURE_COOLDOWN_SECONDS") or "").strip()
    if not raw:
        return 300
    try:
        return _clamp_int(int(raw), min_value=0, max_value=86400)
    except ValueError:
        return 300


def _orchestrator_digest_dedupe_lookback() -> int:
    raw = (os.getenv("ISPEC_ORCHESTRATOR_DIGEST_DEDUPE_LOOKBACK") or "").strip()
    if not raw:
        return 200
    try:
        return _clamp_int(int(raw), min_value=20, max_value=5000)
    except ValueError:
        return 200


def _orchestrator_review_enqueue_guard(
    *,
    payload: dict[str, Any],
    now: datetime,
    current_command_id: int | None = None,
) -> dict[str, Any] | None:
    session_id = str(payload.get("session_id") or "").strip()
    if not session_id:
        return None
    target_message_id = _safe_int(payload.get("target_message_id"))
    cooldown_seconds = _orchestrator_review_failure_cooldown_seconds()
    lookback = _orchestrator_review_dedupe_lookback()

    candidates: list[dict[str, Any]] = []
    with get_agent_session() as db:
        rows = (
            db.query(AgentCommand)
            .filter(AgentCommand.command_type == COMMAND_REVIEW_SUPPORT_SESSION)
            .order_by(AgentCommand.id.desc())
            .limit(int(lookback))
            .all()
        )
        for row in rows:
            candidates.append(
                {
                    "id": int(row.id or 0),
                    "status": str(row.status or "").strip().lower(),
                    "error": str(row.error or "").strip().lower(),
                    "updated_at": getattr(row, "updated_at", None),
                    "payload": dict(row.payload_json or {}),
                }
            )

    for row in candidates:
        row_id = int(row.get("id") or 0)
        if current_command_id is not None and row_id == int(current_command_id):
            continue

        row_payload = dict(row.get("payload") or {})
        row_session_id = str(row_payload.get("session_id") or "").strip()
        if row_session_id != session_id:
            continue
        row_target_message_id = _safe_int(row_payload.get("target_message_id"))
        same_target = (
            target_message_id is None
            or row_target_message_id is None
            or int(target_message_id) == int(row_target_message_id)
        )
        if not same_target:
            continue

        status = str(row.get("status") or "").strip().lower()
        if status in {"queued", "running"}:
            return {
                "reason": "duplicate_inflight",
                "related_command_id": row_id,
                "related_status": status,
                "session_id": session_id,
                "target_message_id": int(target_message_id) if target_message_id else None,
            }

        if status != "failed" or cooldown_seconds <= 0:
            continue

        error = str(row.get("error") or "").strip().lower()
        if error != "invalid_review_output":
            continue

        updated_at = _as_utc_datetime(row.get("updated_at"))
        if not isinstance(updated_at, datetime):
            continue
        age_seconds = max(0, int((now - updated_at).total_seconds()))
        if age_seconds >= int(cooldown_seconds):
            continue
        return {
            "reason": "cooldown_after_invalid_review_output",
            "related_command_id": row_id,
            "related_status": status,
            "session_id": session_id,
            "target_message_id": int(target_message_id) if target_message_id else None,
            "age_seconds": int(age_seconds),
            "cooldown_seconds": int(cooldown_seconds),
            "retry_after_seconds": int(max(1, int(cooldown_seconds) - age_seconds)),
        }

    return None


def _orchestrator_digest_enqueue_guard(
    *,
    payload: dict[str, Any],
    now: datetime,
    current_command_id: int | None = None,
) -> dict[str, Any] | None:
    cursor_review_id = _safe_int(payload.get("cursor_review_id"))
    if cursor_review_id is None:
        cursor_review_id = _safe_int(payload.get("from_review_id"))
    cooldown_seconds = _orchestrator_digest_failure_cooldown_seconds()
    lookback = _orchestrator_digest_dedupe_lookback()

    candidates: list[dict[str, Any]] = []
    with get_agent_session() as db:
        rows = (
            db.query(AgentCommand)
            .filter(AgentCommand.command_type == COMMAND_BUILD_SUPPORT_DIGEST)
            .order_by(AgentCommand.id.desc())
            .limit(int(lookback))
            .all()
        )
        for row in rows:
            candidates.append(
                {
                    "id": int(row.id or 0),
                    "status": str(row.status or "").strip().lower(),
                    "error": str(row.error or "").strip().lower(),
                    "updated_at": getattr(row, "updated_at", None),
                    "payload": dict(row.payload_json or {}),
                }
            )

    for row in candidates:
        row_id = int(row.get("id") or 0)
        if current_command_id is not None and row_id == int(current_command_id):
            continue

        row_payload = dict(row.get("payload") or {})
        row_cursor = _safe_int(row_payload.get("cursor_review_id"))
        if row_cursor is None:
            row_cursor = _safe_int(row_payload.get("from_review_id"))
        same_cursor = (
            cursor_review_id is None
            or row_cursor is None
            or int(cursor_review_id) == int(row_cursor)
        )
        if not same_cursor:
            continue

        status = str(row.get("status") or "").strip().lower()
        if status in {"queued", "running"}:
            return {
                "reason": "duplicate_inflight",
                "related_command_id": row_id,
                "related_status": status,
                "cursor_review_id": int(cursor_review_id) if cursor_review_id is not None else None,
            }

        if status != "failed" or cooldown_seconds <= 0:
            continue

        error = str(row.get("error") or "").strip().lower()
        if error != "invalid_digest_output":
            continue

        updated_at = _as_utc_datetime(row.get("updated_at"))
        if not isinstance(updated_at, datetime):
            continue
        age_seconds = max(0, int((now - updated_at).total_seconds()))
        if age_seconds >= int(cooldown_seconds):
            continue
        return {
            "reason": "cooldown_after_invalid_digest_output",
            "related_command_id": row_id,
            "related_status": status,
            "cursor_review_id": int(cursor_review_id) if cursor_review_id is not None else None,
            "age_seconds": int(age_seconds),
            "cooldown_seconds": int(cooldown_seconds),
            "retry_after_seconds": int(max(1, int(cooldown_seconds) - age_seconds)),
        }

    return None


def _scalar_row_int(row: Any) -> int:
    if row is None:
        return 0
    if isinstance(row, int):
        return row
    value = getattr(row, "id", None)
    if isinstance(value, int):
        return value
    if isinstance(row, (tuple, list)) and row:
        head = row[0]
        if isinstance(head, int):
            return head
        try:
            return int(head)
        except Exception:
            return 0
    try:
        return int(row)
    except Exception:
        return 0


@dataclass(frozen=True)
class ClaimedCommand:
    id: int
    command_type: str
    payload: dict[str, Any]
    attempts: int
    max_attempts: int


@dataclass(frozen=True)
class CommandExecution:
    ok: bool
    result: dict[str, Any]
    error: str | None = None
    prompt: dict[str, Any] | None = None
    response: dict[str, Any] | None = None
    defer_seconds: int | None = None


LLMTask = Generator[InferenceRequest, AssistantReply, CommandExecution]


def _run_llm_task_sync(task: LLMTask) -> CommandExecution:
    """Run a generator-style LLM task synchronously (blocking).

    The same task can also be driven asynchronously by the supervisor inference
    broker. Keeping a single task definition prevents drift between "dev" and
    "prod" behavior.
    """

    reply: AssistantReply | None = None
    while True:
        try:
            request = task.send(reply) if reply is not None else next(task)
        except StopIteration as stop:
            value = stop.value
            if isinstance(value, CommandExecution):
                return value
            return CommandExecution(
                ok=False,
                result={"ok": False, "error": "LLM task returned invalid result type."},
                error="invalid_llm_task_result",
            )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            return CommandExecution(ok=False, result={"ok": False, "error": error}, error=error)

        if not isinstance(request, InferenceRequest):
            return CommandExecution(
                ok=False,
                result={"ok": False, "error": "LLM task yielded invalid request type."},
                error="invalid_llm_task_request",
            )

        reply = generate_reply(
            messages=request.messages,
            tools=request.tools,
            tool_choice=request.tool_choice,
            stage=request.stage,  # type: ignore[arg-type]
            vllm_extra_body=request.vllm_extra_body,
            observability_context={
                "surface": "supervisor_task",
                "stage": request.stage,
                **(request.observability_context or {}),
            },
        )


def _claim_next_command(
    *,
    agent_id: str,
    run_id: str,
    exclude_command_types: set[str] | None = None,
) -> ClaimedCommand | None:
    assert_main_thread("supervisor._claim_next_command")
    now = utcnow()
    with get_agent_session() as db:
        query = (
            db.query(AgentCommand)
            .filter(AgentCommand.status == "queued")
            .filter(AgentCommand.available_at <= now)
        )
        if exclude_command_types:
            query = query.filter(~AgentCommand.command_type.in_(sorted({str(x) for x in exclude_command_types if x})))
        cmd = query.order_by(AgentCommand.priority.desc(), AgentCommand.id.asc()).first()
        if cmd is None:
            return None

        cmd.status = "running"
        cmd.claimed_at = now
        cmd.claimed_by_agent_id = agent_id
        cmd.claimed_by_run_id = run_id
        cmd.started_at = now
        cmd.updated_at = now
        cmd.attempts = int(cmd.attempts or 0) + 1

        db.commit()
        db.refresh(cmd)

        return ClaimedCommand(
            id=int(cmd.id),
            command_type=str(cmd.command_type or ""),
            payload=dict(cmd.payload_json or {}),
            attempts=int(cmd.attempts or 0),
            max_attempts=int(cmd.max_attempts or 0),
        )


def _finish_command(
    *,
    command_id: int,
    ok: bool,
    result: dict[str, Any] | None,
    error: str | None,
) -> None:
    assert_main_thread("supervisor._finish_command")
    now = utcnow()
    with get_agent_session() as db:
        cmd = db.query(AgentCommand).filter(AgentCommand.id == int(command_id)).first()
        if cmd is None:
            return
        cmd.status = "succeeded" if ok else "failed"
        cmd.ended_at = now
        cmd.updated_at = now
        cmd.error = error
        if result is not None:
            cmd.result_json = dict(result)
        db.commit()


def _touch_command_updated_at(*, command_id: int) -> None:
    """Bump `agent_command.updated_at` for a running command.

    This prevents long-running work (e.g. model inference) from being recovered
    as "stale" while it is legitimately in-flight.
    """

    assert_main_thread("supervisor._touch_command_updated_at")
    now = utcnow()
    with get_agent_session() as db:
        cmd = db.query(AgentCommand).filter(AgentCommand.id == int(command_id)).first()
        if cmd is None:
            return
        status = str(getattr(cmd, "status", "") or "").strip().lower()
        if status != "running":
            return
        cmd.updated_at = now
        db.commit()


def _defer_command(
    *,
    command_id: int,
    delay_seconds: int,
    result: dict[str, Any] | None,
    error: str | None,
) -> None:
    assert_main_thread("supervisor._defer_command")
    now = utcnow()
    available_at = now + timedelta(seconds=max(1, int(delay_seconds)))
    with get_agent_session() as db:
        cmd = db.query(AgentCommand).filter(AgentCommand.id == int(command_id)).first()
        if cmd is None:
            return
        cmd.status = "queued"
        cmd.available_at = available_at
        cmd.claimed_at = None
        cmd.claimed_by_agent_id = None
        cmd.claimed_by_run_id = None
        cmd.started_at = None
        cmd.ended_at = None
        cmd.updated_at = now
        cmd.error = error
        if result is not None:
            cmd.result_json = dict(result)
        db.commit()


def _enqueue_command(
    *,
    command_type: str,
    payload: dict[str, Any] | None = None,
    priority: int = 0,
    available_at: datetime | None = None,
    max_attempts: int = 3,
) -> int | None:
    assert_main_thread("supervisor._enqueue_command")
    now = utcnow()
    cmd_type = (command_type or "").strip()
    if not cmd_type:
        return None
    with get_agent_session() as db:
        row = AgentCommand(
            command_type=cmd_type,
            status="queued",
            priority=int(priority),
            created_at=now,
            updated_at=now,
            available_at=available_at or now,
            attempts=0,
            max_attempts=int(max_attempts),
            payload_json=dict(payload or {}),
            result_json={},
        )
        db.add(row)
        db.commit()
        db.refresh(row)
        return int(row.id)


def _running_command_stale_seconds() -> int:
    raw = (os.getenv("ISPEC_SUPERVISOR_RUNNING_STALE_SECONDS") or "").strip()
    if not raw:
        return 300
    try:
        return _clamp_int(int(raw), min_value=30, max_value=7 * 86400)
    except ValueError:
        return 300


def _command_last_activity_at(cmd: AgentCommand) -> datetime | None:
    for field in ("updated_at", "started_at", "claimed_at", "created_at"):
        value = getattr(cmd, field, None)
        if isinstance(value, datetime):
            return value
    return None


def _as_utc_datetime(value: datetime | None) -> datetime | None:
    if not isinstance(value, datetime):
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _recover_stale_running_commands(*, agent_id: str, run_id: str) -> dict[str, Any]:
    assert_main_thread("supervisor._recover_stale_running_commands")
    now = utcnow()
    stale_seconds = _running_command_stale_seconds()
    stale_before = now - timedelta(seconds=stale_seconds)

    recovered = 0
    skipped_not_stale = 0
    skipped_current_run = 0
    skipped_other_agent = 0
    skipped_recent_running_run = 0
    running_total = 0

    with get_agent_session() as db:
        running_rows = (
            db.query(AgentCommand)
            .filter(AgentCommand.status == "running")
            .order_by(AgentCommand.id.asc())
            .all()
        )
        running_total = len(running_rows)

        claimed_run_ids = sorted(
            {
                str(row.claimed_by_run_id).strip()
                for row in running_rows
                if isinstance(row.claimed_by_run_id, str) and row.claimed_by_run_id.strip()
            }
        )
        run_state: dict[str, tuple[str, datetime | None]] = {}
        if claimed_run_ids:
            run_rows = (
                db.query(AgentRun.run_id, AgentRun.status, AgentRun.updated_at)
                .filter(AgentRun.run_id.in_(claimed_run_ids))
                .all()
            )
            for claimed_run_id, status, updated_at in run_rows:
                run_state[str(claimed_run_id)] = (str(status or "").strip().lower(), updated_at)

        for cmd in running_rows:
            claimed_agent_id = str(cmd.claimed_by_agent_id or "").strip()
            if claimed_agent_id and claimed_agent_id != agent_id:
                skipped_other_agent += 1
                continue

            last_activity = _as_utc_datetime(_command_last_activity_at(cmd))
            if isinstance(last_activity, datetime) and last_activity > stale_before:
                skipped_not_stale += 1
                continue

            claimed_run_id = str(cmd.claimed_by_run_id or "").strip()
            if claimed_run_id and claimed_run_id == run_id:
                skipped_current_run += 1
                continue

            if claimed_run_id:
                status_updated = run_state.get(claimed_run_id)
                if isinstance(status_updated, tuple):
                    claimed_status, claimed_updated_at = status_updated
                    claimed_updated_at_utc = _as_utc_datetime(claimed_updated_at)
                    if (
                        claimed_status == "running"
                        and isinstance(claimed_updated_at_utc, datetime)
                        and claimed_updated_at_utc > stale_before
                    ):
                        skipped_recent_running_run += 1
                        continue

            previous = {
                "claimed_by_agent_id": claimed_agent_id or None,
                "claimed_by_run_id": claimed_run_id or None,
                "claimed_at": cmd.claimed_at.isoformat() if isinstance(cmd.claimed_at, datetime) else None,
                "started_at": cmd.started_at.isoformat() if isinstance(cmd.started_at, datetime) else None,
                "updated_at": cmd.updated_at.isoformat() if isinstance(cmd.updated_at, datetime) else None,
                "error": _truncate_text(cmd.error, limit=240) or None,
            }

            cmd.status = "queued"
            cmd.available_at = now
            cmd.claimed_at = None
            cmd.claimed_by_agent_id = None
            cmd.claimed_by_run_id = None
            cmd.started_at = None
            cmd.ended_at = None
            cmd.updated_at = now
            cmd.error = "recovered_stale_running_command"

            result_payload = dict(cmd.result_json or {})
            result_payload["stale_recovery"] = {
                "recovered_at": now.isoformat(),
                "stale_seconds": int(stale_seconds),
                "previous": previous,
            }
            cmd.result_json = result_payload
            recovered += 1

        if recovered > 0:
            db.commit()

    return {
        "ok": True,
        "running_total": int(running_total),
        "recovered": int(recovered),
        "stale_seconds": int(stale_seconds),
        "skipped_not_stale": int(skipped_not_stale),
        "skipped_current_run": int(skipped_current_run),
        "skipped_other_agent": int(skipped_other_agent),
        "skipped_recent_running_run": int(skipped_recent_running_run),
    }


def _compact_session_memory(payload: dict[str, Any]) -> tuple[bool, dict[str, Any], str | None]:
    provider = (os.getenv("ISPEC_ASSISTANT_PROVIDER") or "").strip().lower()
    if provider != "vllm":
        return (
            False,
            {"ok": False, "error": "Memory compaction requires ISPEC_ASSISTANT_PROVIDER=vllm."},
            "provider_not_vllm",
        )

    session_id = str(payload.get("session_id") or "").strip()
    session_pk = _safe_int(payload.get("session_pk"))
    if not session_id and session_pk is None:
        return False, {"ok": False, "error": "Missing session_id/session_pk."}, "missing_session"

    requested_target = _safe_int(payload.get("target_message_id")) or 0
    if requested_target <= 0:
        return False, {"ok": False, "error": "Missing target_message_id."}, "missing_target"

    keep_last = _safe_int(payload.get("keep_last")) or 6
    if keep_last < 1:
        keep_last = 1

    batch_size = _safe_int(os.getenv("ISPEC_ASSISTANT_COMPACTION_BATCH_SIZE")) or 20
    if batch_size < 1:
        batch_size = 1
    if batch_size > 50:
        batch_size = 50

    with get_assistant_session() as db:
        session = None
        if session_pk is not None:
            session = db.query(SupportSession).filter(SupportSession.id == int(session_pk)).first()
        if session is None and session_id:
            session = db.query(SupportSession).filter(SupportSession.session_id == session_id).first()
        if session is None:
            return (
                False,
                {"ok": False, "error": "Support session not found.", "session_id": session_id},
                "session_not_found",
            )

        state = _load_json_dict(getattr(session, "state_json", None))
        memory_up_to_id = _safe_int(state.get("conversation_memory_up_to_id")) or 0

        boundary_rows = (
            db.query(SupportMessage.id)
            .filter(SupportMessage.session_pk == session.id)
            .order_by(SupportMessage.id.desc())
            .limit(keep_last + 1)
            .all()
        )
        current_target = 0
        if len(boundary_rows) > keep_last:
            current_target = _scalar_row_int(boundary_rows[keep_last])

        target_id = max(requested_target, current_target) if current_target else requested_target
        if target_id <= memory_up_to_id:
            return True, {
                "ok": True,
                "noop": True,
                "session_id": session.session_id,
                "memory_up_to_id": memory_up_to_id,
                "target_id": target_id,
            }, None

        rows = (
            db.query(SupportMessage)
            .filter(SupportMessage.session_pk == session.id)
            .filter(SupportMessage.id > memory_up_to_id)
            .filter(SupportMessage.id <= target_id)
            .order_by(SupportMessage.id.asc())
            .all()
        )
        if not rows:
            return (
                False,
                {
                    "ok": False,
                    "error": "Support messages not available for compaction yet.",
                    "session_id": session.session_id,
                    "memory_up_to_id": memory_up_to_id,
                    "target_id": target_id,
                },
                "messages_not_ready",
            )
        previous_memory = state.get("conversation_memory") if isinstance(state.get("conversation_memory"), dict) else None
        messages = [
            {"id": int(row.id), "role": row.role, "content": row.content}
            for row in rows
            if row.role in {"user", "assistant", "system"} and row.content
        ]

    if not messages:
        with get_assistant_session() as db:
            session = None
            if session_pk is not None:
                session = db.query(SupportSession).filter(SupportSession.id == int(session_pk)).first()
            if session is None and session_id:
                session = db.query(SupportSession).filter(SupportSession.session_id == session_id).first()
            if session is None:
                return (
                    False,
                    {"ok": False, "error": "Support session not found.", "session_id": session_id},
                    "session_not_found",
                )

            state = _load_json_dict(getattr(session, "state_json", None))
            state["conversation_memory_up_to_id"] = int(target_id)
            state["conversation_memory_updated_at"] = utcnow().isoformat()
            state["conversation_memory_version"] = 1
            requested = _safe_int(state.get("conversation_memory_requested_up_to_id")) or 0
            if requested <= target_id:
                state["conversation_memory_requested_up_to_id"] = int(target_id)
            session.state_json = _dump_json(state)
            session.updated_at = utcnow()
        return True, {"ok": True, "noop": True, "target_id": target_id, "message_count": 0}, None

    batches: list[dict[str, Any]] = []
    mem = previous_memory
    reply_meta: list[dict[str, Any]] = []
    for idx in range(0, len(messages), batch_size):
        batch = messages[idx : idx + batch_size]
        distill = distill_conversation_memory(
            previous_memory=mem,
            new_messages=[{"role": item["role"], "content": item["content"]} for item in batch],
            generate_reply_fn=generate_reply,
        )
        reply_meta.append(
            {
                "provider": distill.reply.provider,
                "model": distill.reply.model,
                "ok": bool(distill.reply.ok),
                "meta": distill.reply.meta,
                "error": distill.reply.error,
            }
        )
        if distill.memory is None or not distill.reply.ok:
            error = distill.reply.error or "Memory distillation failed."
            with get_assistant_session() as db:
                session = None
                if session_pk is not None:
                    session = db.query(SupportSession).filter(SupportSession.id == int(session_pk)).first()
                if session is None and session_id:
                    session = db.query(SupportSession).filter(SupportSession.session_id == session_id).first()
                if session is not None:
                    state = _load_json_dict(getattr(session, "state_json", None))
                    state["conversation_memory_last_error"] = error
                    state["conversation_memory_requested_up_to_id"] = int(
                        _safe_int(state.get("conversation_memory_up_to_id")) or 0
                    )
                    state["conversation_memory_updated_at"] = utcnow().isoformat()
                    session.state_json = _dump_json(state)
                    session.updated_at = utcnow()
            return False, {"ok": False, "error": error, "reply": reply_meta[-1]}, error

        mem = distill.memory
        batches.append({"start_id": int(batch[0]["id"]), "end_id": int(batch[-1]["id"]), "count": len(batch)})

    with get_assistant_session() as db:
        session = None
        if session_pk is not None:
            session = db.query(SupportSession).filter(SupportSession.id == int(session_pk)).first()
        if session is None and session_id:
            session = db.query(SupportSession).filter(SupportSession.session_id == session_id).first()
        if session is None:
            return (
                False,
                {"ok": False, "error": "Support session not found.", "session_id": session_id},
                "session_not_found",
            )

        state = _load_json_dict(getattr(session, "state_json", None))
        state["conversation_memory"] = mem or {}
        state["conversation_memory_version"] = int((mem or {}).get("schema_version") or 1)
        state["conversation_memory_up_to_id"] = int(target_id)
        state["conversation_memory_updated_at"] = utcnow().isoformat()
        state.pop("conversation_memory_last_error", None)

        requested = _safe_int(state.get("conversation_memory_requested_up_to_id")) or 0
        if requested <= target_id:
            state["conversation_memory_requested_up_to_id"] = int(target_id)

        memory_row = SupportMemory(
            session_pk=int(session.id),
            user_id=int(session.user_id) if session.user_id is not None else 0,
            kind="summary",
            key="conversation_memory",
            value_json=_dump_json(mem or {}),
        )
        db.add(memory_row)
        db.flush()

        evidence_ids: set[int] = set()
        evidence_weights: dict[int, float] = {}
        for batch in batches:
            end_id = _safe_int(batch.get("end_id"))
            if end_id is None or end_id <= 0:
                continue
            evidence_ids.add(int(end_id))
            weight = float(_safe_int(batch.get("count")) or 1)
            evidence_weights[int(end_id)] = max(evidence_weights.get(int(end_id), 0.0), weight)
        evidence_ids.add(int(target_id))

        triggered_by = _safe_int(payload.get("triggered_by_message_id"))
        if triggered_by is not None and triggered_by > 0:
            evidence_ids.add(int(triggered_by))

        for message_id in sorted(evidence_ids):
            db.add(
                SupportMemoryEvidence(
                    memory_id=int(memory_row.id),
                    message_id=int(message_id),
                    weight=float(evidence_weights.get(int(message_id), 1.0)),
                )
            )

        session.state_json = _dump_json(state)
        session.updated_at = utcnow()

    return True, {
        "ok": True,
        "session_id": session_id,
        "target_id": target_id,
        "message_count": len(messages),
        "batch_size": batch_size,
        "batches": batches,
        "llm": reply_meta[-1] if reply_meta else None,
        "llm_batches": reply_meta,
    }, None


_ORCHESTRATOR_STATE_VERSION = 1
_ORCHESTRATOR_DECISION_VERSION = 1
_SESSION_REVIEW_VERSION = 1
_SUPPORT_DIGEST_VERSION = 1
_REPO_REVIEW_VERSION = 1
_TACKLE_ASSESS_VERSION = 1
_TACKLE_PROMPT_VERSION = 1
_ORCHESTRATOR_DEFAULT_BASE_SECONDS = 60
_ORCHESTRATOR_BACKOFF_MAX_EXP = 6
_SCHEDULER_STATE_VERSION = 1


def _orchestrator_enabled() -> bool:
    parsed, err = _parse_env_tristate_bool(
        os.getenv("ISPEC_ORCHESTRATOR_ENABLED"),
        key="ISPEC_ORCHESTRATOR_ENABLED",
    )
    if err:
        logger.warning("%s", err)
        parsed = None
    if parsed is True:
        return True
    if parsed is False:
        return False
    provider = (os.getenv("ISPEC_ASSISTANT_PROVIDER") or "").strip().lower()
    return provider == "vllm"


def _orchestrator_tick_min_seconds() -> int:
    raw = (os.getenv("ISPEC_ORCHESTRATOR_MIN_SECONDS") or "").strip()
    if not raw:
        return 30
    try:
        return max(10, int(raw))
    except ValueError:
        return 30


def _orchestrator_tick_max_seconds() -> int:
    raw = (os.getenv("ISPEC_ORCHESTRATOR_MAX_SECONDS") or "").strip()
    if not raw:
        # Default to an hour so the system stays quiet when idle, while allowing
        # event-driven "pokes" (API + queued work) to bring ticks forward.
        return 3600
    try:
        return max(30, int(raw))
    except ValueError:
        return 3600


def _orchestrator_max_commands_per_tick() -> int:
    raw = (os.getenv("ISPEC_ORCHESTRATOR_MAX_COMMANDS_PER_TICK") or "").strip()
    if not raw:
        return 2
    try:
        return _clamp_int(int(raw), min_value=0, max_value=6)
    except ValueError:
        return 2


def _orchestrator_base_tick_seconds(*, min_tick_seconds: int, max_tick_seconds: int) -> int:
    raw = (os.getenv("ISPEC_ORCHESTRATOR_BASE_SECONDS") or "").strip()
    base = _ORCHESTRATOR_DEFAULT_BASE_SECONDS
    if raw:
        try:
            base = int(raw)
        except ValueError:
            base = _ORCHESTRATOR_DEFAULT_BASE_SECONDS
    base = max(int(min_tick_seconds), int(base))
    return _clamp_int(base, min_value=int(min_tick_seconds), max_value=int(max_tick_seconds))


def _orchestrator_idle_backoff_seconds(*, base_seconds: int, idle_streak: int) -> int:
    """Exponential backoff for idle orchestrator ticks.

    We treat the first idle tick as ``base_seconds`` and then double for each
    consecutive idle tick.
    """

    return int(
        backoff_exponential_current(
            int(idle_streak),
            base_seconds=float(base_seconds),
            start_step=1,
            max_exp=int(_ORCHESTRATOR_BACKOFF_MAX_EXP),
        )
    )


def _orchestrator_error_backoff_seconds(*, base_seconds: int, error_streak: int) -> int:
    """Exponential backoff for invalid orchestrator outputs."""

    return int(
        backoff_exponential_current(
            max(0, int(error_streak)),
            base_seconds=float(base_seconds),
            start_step=0,
            max_exp=int(_ORCHESTRATOR_BACKOFF_MAX_EXP),
        )
    )


def _load_orchestrator_state(run: AgentRun) -> dict[str, Any]:
    summary = run.summary_json if isinstance(run.summary_json, dict) else {}
    state = summary.get("orchestrator")
    if not isinstance(state, dict):
        state = {}
    if int(state.get("schema_version") or 0) != _ORCHESTRATOR_STATE_VERSION:
        state = {"schema_version": _ORCHESTRATOR_STATE_VERSION}
    if not isinstance(state.get("recent_thoughts"), list):
        state["recent_thoughts"] = []
    if not isinstance(state.get("recent_model_thoughts"), list):
        state["recent_model_thoughts"] = []
    if not isinstance(state.get("ticks"), int):
        state["ticks"] = 0
    if not isinstance(state.get("idle_streak"), int):
        state["idle_streak"] = 0
    if not isinstance(state.get("error_streak"), int):
        state["error_streak"] = 0
    return state


def _save_orchestrator_state(*, run: AgentRun, state: dict[str, Any]) -> None:
    summary = run.summary_json if isinstance(run.summary_json, dict) else {}
    summary = dict(summary)
    summary["orchestrator"] = dict(state)
    run.summary_json = summary
    flag_modified(run, "summary_json")


def _load_scheduler_state(run: AgentRun) -> dict[str, Any]:
    summary = run.summary_json if isinstance(run.summary_json, dict) else {}
    state = summary.get("scheduler")
    if not isinstance(state, dict):
        state = {}
    if int(state.get("schema_version") or 0) != _SCHEDULER_STATE_VERSION:
        state = {"schema_version": _SCHEDULER_STATE_VERSION}
    if not isinstance(state.get("slack"), dict):
        state["slack"] = {}
    if not isinstance(state.get("assistant_jobs"), dict):
        state["assistant_jobs"] = {}
    if not isinstance(state.get("legacy_sync"), dict):
        state["legacy_sync"] = {}
    if not isinstance(state.get("agent_log_archive"), dict):
        state["agent_log_archive"] = {}
    return state


def _save_scheduler_state(*, run: AgentRun, state: dict[str, Any]) -> None:
    summary = run.summary_json if isinstance(run.summary_json, dict) else {}
    summary = dict(summary)
    summary["scheduler"] = dict(state)
    run.summary_json = summary
    flag_modified(run, "summary_json")


def _assistant_snapshot(*, assistant_db) -> dict[str, Any]:
    try:
        total_sessions = int(assistant_db.query(SupportSession.id).count())
    except Exception:
        total_sessions = 0
    try:
        total_messages = int(assistant_db.query(SupportMessage.id).count())
    except Exception:
        total_messages = 0
    try:
        total_reviews = int(assistant_db.query(SupportSessionReview.id).count())
    except Exception:
        total_reviews = 0
    latest_review_id = 0
    latest_review_at: str | None = None
    try:
        latest_review = (
            assistant_db.query(SupportSessionReview)
            .order_by(SupportSessionReview.id.desc())
            .first()
        )
        if latest_review is not None:
            latest_review_id = int(latest_review.id or 0)
            latest_review_at = latest_review.updated_at.isoformat() if latest_review.updated_at else None
    except Exception:
        latest_review_id = 0
        latest_review_at = None

    sessions: list[SupportSession] = (
        assistant_db.query(SupportSession)
        .order_by(SupportSession.updated_at.desc(), SupportSession.id.desc())
        .limit(12)
        .all()
    )

    session_pks = [int(session.id) for session in sessions]
    review_by_session_pk: dict[int, dict[str, Any]] = {}
    if session_pks:
        review_rows = (
            assistant_db.query(
                SupportSessionReview.session_pk,
                SupportSessionReview.target_message_id,
                SupportSessionReview.updated_at,
            )
            .filter(SupportSessionReview.session_pk.in_(session_pks))
            .order_by(
                SupportSessionReview.session_pk.asc(),
                SupportSessionReview.target_message_id.desc(),
                SupportSessionReview.updated_at.desc(),
                SupportSessionReview.id.desc(),
            )
            .all()
        )
        for session_pk, target_id, updated_at in review_rows:
            key = int(session_pk)
            if key in review_by_session_pk:
                continue
            review_by_session_pk[key] = {
                "reviewed_up_to_id": int(target_id or 0),
                "review_updated_at": updated_at.isoformat() if updated_at else None,
            }

    items: list[dict[str, Any]] = []
    needs_review: list[dict[str, Any]] = []
    for session in sessions:
        last_message = (
            assistant_db.query(SupportMessage)
            .filter(SupportMessage.session_pk == session.id)
            .order_by(SupportMessage.id.desc())
            .first()
        )
        last_assistant = (
            assistant_db.query(SupportMessage)
            .filter(SupportMessage.session_pk == session.id)
            .filter(SupportMessage.role == "assistant")
            .order_by(SupportMessage.id.desc())
            .first()
        )
        last_user = (
            assistant_db.query(SupportMessage)
            .filter(SupportMessage.session_pk == session.id)
            .filter(SupportMessage.role == "user")
            .order_by(SupportMessage.id.desc())
            .first()
        )
        msg_count = int(
            assistant_db.query(SupportMessage.id).filter(SupportMessage.session_pk == session.id).count()
        )
        state = _load_json_dict(getattr(session, "state_json", None))
        review_info = review_by_session_pk.get(int(session.id))
        if isinstance(review_info, dict):
            reviewed_up_to = _safe_int(review_info.get("reviewed_up_to_id")) or 0
            review_updated_at = review_info.get("review_updated_at")
        else:
            reviewed_up_to = _safe_int(state.get("conversation_review_up_to_id")) or 0
            review_updated_at = state.get("conversation_review_updated_at")
        memory_up_to = _safe_int(state.get("conversation_memory_up_to_id")) or 0
        last_id = int(last_message.id) if last_message is not None else 0
        last_assistant_id = int(last_assistant.id) if last_assistant is not None else 0
        session_item = {
            "session_id": str(session.session_id),
            "user_id": int(session.user_id) if session.user_id is not None else None,
            "updated_at": session.updated_at.isoformat() if getattr(session, "updated_at", None) else None,
            "message_count": msg_count,
            "last_message_id": last_id,
            "last_assistant_message_id": last_assistant_id,
            "last_user_message": _truncate_text(getattr(last_user, "content", None), limit=240),
            "last_message_role": getattr(last_message, "role", None),
            "reviewed_up_to_id": reviewed_up_to,
            "review_updated_at": review_updated_at,
            "memory_up_to_id": memory_up_to,
        }
        items.append(session_item)
        # Only review completed assistant turns. If the most recent message is a
        # user message, the assistant is likely still responding.
        if (
            getattr(last_message, "role", None) == "assistant"
            and last_assistant_id > reviewed_up_to
            and last_assistant_id > 0
        ):
            needs_review.append(session_item)

    return {
        "ok": True,
        "sessions_total": total_sessions,
        "messages_total": total_messages,
        "reviews_total": total_reviews,
        "latest_review_id": latest_review_id,
        "latest_review_at": latest_review_at,
        "recent_sessions": items,
        "sessions_needing_review": needs_review,
    }


def _orchestrator_decision_schema(*, max_commands: int) -> dict[str, Any]:
    allowed_commands = [COMMAND_REVIEW_SUPPORT_SESSION, COMMAND_BUILD_SUPPORT_DIGEST, COMMAND_REVIEW_REPO]
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "schema_version": {"type": "integer", "enum": [_ORCHESTRATOR_DECISION_VERSION]},
            "thoughts": {"type": "string", "maxLength": 2000},
            "next_tick_seconds": {"type": "integer", "minimum": 10, "maximum": 86400},
            "commands": {
                "type": "array",
                "maxItems": max(0, int(max_commands)),
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "command_type": {"type": "string", "enum": allowed_commands},
                        "priority": {"type": "integer", "minimum": -10, "maximum": 10},
                        "delay_seconds": {"type": "integer", "minimum": 0, "maximum": 86400},
                        "payload": {"type": "object"},
                    },
                    "required": ["command_type", "priority", "delay_seconds", "payload"],
                },
            },
        },
        "required": ["schema_version", "thoughts", "next_tick_seconds", "commands"],
    }


@prompt_binding("supervisor.orchestrator.system")
def _orchestrator_system_prompt(*, max_commands: int) -> str:
    return load_bound_prompt(_orchestrator_system_prompt, values={"max_commands": max_commands}).text

def _validate_orchestrator_decision(
    decision: dict[str, Any] | None,
    *,
    min_tick_seconds: int,
    max_tick_seconds: int,
    max_commands: int,
) -> dict[str, Any] | None:
    if not isinstance(decision, dict):
        return None

    thoughts = _truncate_text(decision.get("thoughts"), limit=2000)
    raw_next = _safe_int(decision.get("next_tick_seconds")) or 0
    next_tick_seconds = _clamp_int(raw_next if raw_next > 0 else min_tick_seconds, min_value=min_tick_seconds, max_value=max_tick_seconds)

    commands_raw = decision.get("commands")
    if not isinstance(commands_raw, list):
        commands_raw = []
    commands: list[dict[str, Any]] = []
    for item in commands_raw[: max(0, max_commands)]:
        if not isinstance(item, dict):
            continue
        cmd_type = str(item.get("command_type") or "").strip()
        if cmd_type not in {COMMAND_REVIEW_SUPPORT_SESSION, COMMAND_BUILD_SUPPORT_DIGEST, COMMAND_REVIEW_REPO}:
            continue
        payload = item.get("payload")
        if not isinstance(payload, dict):
            payload = {}
        delay_seconds = _safe_int(item.get("delay_seconds")) or 0
        delay_seconds = _clamp_int(delay_seconds, min_value=0, max_value=86400)
        priority = _safe_int(item.get("priority")) or 0
        priority = _clamp_int(priority, min_value=-10, max_value=10)
        commands.append(
            {
                "command_type": cmd_type,
                "priority": int(priority),
                "delay_seconds": int(delay_seconds),
                "payload": dict(payload),
            }
        )

    return {
        "schema_version": _ORCHESTRATOR_DECISION_VERSION,
        "thoughts": thoughts,
        "next_tick_seconds": int(next_tick_seconds),
        "commands": commands,
    }


def _orchestrator_command_alias(command_type: str) -> str:
    mapping = {
        COMMAND_REVIEW_SUPPORT_SESSION: "session_review",
        COMMAND_BUILD_SUPPORT_DIGEST: "support_digest",
        COMMAND_REVIEW_REPO: "repo_review",
    }
    key = str(command_type or "").strip()
    return mapping.get(key, key or "unknown")


def _orchestrator_count_commands(items: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        command_type = str(item.get("command_type") or "").strip()
        if not command_type:
            continue
        alias = _orchestrator_command_alias(command_type)
        counts[alias] = int(counts.get(alias, 0) or 0) + 1
    return counts


def _orchestrator_action_summary(
    *,
    decision_commands: list[dict[str, Any]],
    scheduled: list[dict[str, Any]],
    skipped: list[dict[str, Any]],
) -> dict[str, Any]:
    requested_counts = _orchestrator_count_commands(decision_commands)
    scheduled_counts = _orchestrator_count_commands(scheduled)
    skipped_counts = _orchestrator_count_commands(skipped)

    skipped_reason_counts: dict[str, int] = {}
    for item in skipped:
        if not isinstance(item, dict):
            continue
        guard = item.get("guard")
        reason = guard.get("reason") if isinstance(guard, dict) else None
        reason_key = str(reason or "").strip() or "unspecified"
        skipped_reason_counts[reason_key] = int(skipped_reason_counts.get(reason_key, 0) or 0) + 1

    requested_total = int(sum(int(v) for v in requested_counts.values()))
    scheduled_total = int(sum(int(v) for v in scheduled_counts.values()))
    skipped_total = int(sum(int(v) for v in skipped_counts.values()))

    summary_parts: list[str] = []
    if scheduled_total > 0:
        detail = ", ".join(
            f"{name}={count}"
            for name, count in sorted(scheduled_counts.items(), key=lambda pair: pair[0])
        )
        summary_parts.append(f"Scheduled {scheduled_total} command(s) ({detail}).")
    else:
        summary_parts.append("No commands scheduled this tick.")

    if skipped_total > 0:
        reason_detail = ", ".join(
            f"{reason}={count}"
            for reason, count in sorted(skipped_reason_counts.items(), key=lambda pair: pair[0])
        )
        summary_parts.append(f"Skipped {skipped_total} command(s) ({reason_detail}).")

    if requested_total > 0 and scheduled_total == 0 and skipped_total == 0:
        summary_parts.append(f"Decision requested {requested_total} command(s), but none were accepted.")

    summary = _truncate_text(" ".join(summary_parts).strip(), limit=600)

    return {
        "schema_version": 1,
        "requested": requested_counts,
        "scheduled": scheduled_counts,
        "skipped": skipped_counts,
        "skip_reasons": skipped_reason_counts,
        "totals": {
            "requested": requested_total,
            "scheduled": scheduled_total,
            "skipped": skipped_total,
        },
        "summary": summary,
    }


def _schedule_orchestrator_tick(*, delay_seconds: int, current_command_id: int | None = None) -> int | None:
    if not _orchestrator_enabled():
        return None
    now = utcnow()
    delay_seconds = max(10, int(delay_seconds))
    available_at = now + timedelta(seconds=delay_seconds)
    with get_agent_session() as db:
        existing = (
            db.query(AgentCommand)
            .filter(AgentCommand.command_type == COMMAND_ORCHESTRATOR_TICK)
            .filter(AgentCommand.status == "queued")
            .filter(AgentCommand.available_at > now)
            .order_by(AgentCommand.available_at.asc())
            .first()
        )
        if existing is not None:
            try:
                delta_seconds = abs(
                    float((existing.available_at - available_at).total_seconds())
                )
            except Exception:
                delta_seconds = 0.0
            if delta_seconds >= 1.0:
                existing.available_at = available_at
                existing.updated_at = now
                payload = dict(existing.payload_json or {})
                payload["source"] = payload.get("source") or "self_schedule"
                payload["previous_command_id"] = (
                    int(current_command_id) if current_command_id else None
                )
                payload["rescheduled_at"] = now.isoformat()
                existing.payload_json = payload
            return int(existing.id)
    return _enqueue_command(
        command_type=COMMAND_ORCHESTRATOR_TICK,
        payload={"source": "self_schedule", "previous_command_id": int(current_command_id) if current_command_id else None},
        priority=-5,
        available_at=available_at,
    )


def _task_orchestrator_tick(
    *,
    payload: dict[str, Any],
    agent_id: str,
    run_id: str,
    command_id: int,
) -> LLMTask:
    if not _orchestrator_enabled():
        return CommandExecution(ok=True, result={"ok": True, "disabled": True})

    provider = (os.getenv("ISPEC_ASSISTANT_PROVIDER") or "").strip().lower()
    if provider != "vllm":
        return CommandExecution(
            ok=False,
            result={"ok": False, "error": "Orchestrator requires ISPEC_ASSISTANT_PROVIDER=vllm."},
            error="provider_not_vllm",
        )

    min_tick = _orchestrator_tick_min_seconds()
    max_tick = _orchestrator_tick_max_seconds()
    max_commands = _orchestrator_max_commands_per_tick()
    base_tick = _orchestrator_base_tick_seconds(min_tick_seconds=min_tick, max_tick_seconds=max_tick)

    state_for_prompt: dict[str, Any] = {}
    prev_idle_streak = 0
    prev_error_streak = 0
    with get_agent_session() as agent_db:
        run = agent_db.query(AgentRun).filter(AgentRun.run_id == run_id).one()
        state = _load_orchestrator_state(run)
        prev_idle_streak = int(state.get("idle_streak") or 0)
        prev_error_streak = int(state.get("error_streak") or 0)
        state["ticks"] = int(state.get("ticks") or 0) + 1
        state["last_tick_at"] = utcnow().isoformat()
        recent_thoughts = [str(x) for x in state.get("recent_thoughts") if isinstance(x, str)]
        state["recent_thoughts"] = recent_thoughts[-10:]
        state_for_prompt = {
            "schema_version": int(state.get("schema_version") or 0),
            "ticks": int(state.get("ticks") or 0),
            "idle_streak": int(state.get("idle_streak") or 0),
            "error_streak": int(state.get("error_streak") or 0),
            "last_thought": _truncate_text(
                state.get("last_thought") if isinstance(state.get("last_thought"), str) else None,
                limit=600,
            ),
            "recent_thoughts": [_truncate_text(text, limit=300) for text in recent_thoughts[-5:]],
            "last_action_summary": _truncate_text(
                state.get("last_action_summary") if isinstance(state.get("last_action_summary"), str) else None,
                limit=400,
            ),
            "last_tick_at": state.get("last_tick_at"),
            "next_tick_seconds": state.get("next_tick_seconds"),
            "next_tick_command_id": state.get("next_tick_command_id"),
            "digest_last_review_id": int(state.get("digest_last_review_id") or 0),
            "digest_last_at": state.get("digest_last_at"),
        }
        _save_orchestrator_state(run=run, state=state)
        agent_db.commit()

    with get_assistant_session() as assistant_db:
        assistant_snapshot = _assistant_snapshot(assistant_db=assistant_db)

    prompt_snapshot: dict[str, Any] = dict(assistant_snapshot or {})
    # The orchestrator only needs a tiny slice of assistant state. Keeping the
    # prompt small also prevents "echo the input JSON" failures that can be
    # truncated by max_tokens and become invalid JSON.
    prompt_snapshot.pop("recent_sessions", None)
    if isinstance(prompt_snapshot.get("sessions_needing_review"), list):
        prompt_snapshot["sessions_needing_review"] = prompt_snapshot["sessions_needing_review"][:3]

    context = {
        "schema_version": 1,
        "agent": {"agent_id": agent_id, "run_id": run_id, "command_id": int(command_id)},
        "orchestrator": state_for_prompt,
        "assistant": prompt_snapshot,
        "requested": dict(payload or {}),
    }

    schema = _orchestrator_decision_schema(max_commands=max_commands)
    prompt = load_bound_prompt(_orchestrator_system_prompt, values={"max_commands": max_commands})
    messages = [
        {"role": "system", "content": prompt.text},
        {"role": "user", "content": json.dumps(context, ensure_ascii=False)},
    ]
    reply = yield InferenceRequest(
        messages=messages,
        tools=None,
        vllm_extra_body={"structured_outputs": {"json": schema}, "temperature": 0, "max_tokens": 800},
        observability_context=prompt_observability_context(prompt, extra={"task": "orchestrator_tick"}),
    )

    parsed = _parse_json_object(reply.content)
    validated = _validate_orchestrator_decision(
        parsed if isinstance(parsed, dict) else None,
        min_tick_seconds=min_tick,
        max_tick_seconds=max_tick,
        max_commands=max_commands,
    )

    llm_invalid = validated is None
    if validated is None:
        # Recover from invalid model output by continuing with a safe default.
        validated = {
            "schema_version": _ORCHESTRATOR_DECISION_VERSION,
            "thoughts": "",
            "next_tick_seconds": int(min_tick),
            "commands": [],
        }

    sessions_needing_review = (
        assistant_snapshot.get("sessions_needing_review") if isinstance(assistant_snapshot, dict) else None
    )
    if not isinstance(sessions_needing_review, list):
        sessions_needing_review = []
    sessions_needing_review_count = len(sessions_needing_review)

    if validated is not None and sessions_needing_review:
        wants_session_review = any(
            isinstance(item, dict) and item.get("command_type") == COMMAND_REVIEW_SUPPORT_SESSION
            for item in (validated.get("commands") or [])
        )
        if not wants_session_review:
            newest = sessions_needing_review[0] if sessions_needing_review else None
            session_id = newest.get("session_id") if isinstance(newest, dict) else None
            review_target_id = newest.get("last_assistant_message_id") if isinstance(newest, dict) else None
            if review_target_id is None:
                review_target_id = newest.get("last_message_id") if isinstance(newest, dict) else None
            if isinstance(session_id, str) and session_id.strip():
                payload_review: dict[str, Any] = {"session_id": session_id.strip()}
                if isinstance(review_target_id, int) and review_target_id > 0:
                    payload_review["target_message_id"] = int(review_target_id)
                elif isinstance(review_target_id, str) and review_target_id.strip().isdigit():
                    payload_review["target_message_id"] = int(review_target_id.strip())

                forced = {
                    "command_type": COMMAND_REVIEW_SUPPORT_SESSION,
                    "priority": 0,
                    "delay_seconds": 0,
                    "payload": payload_review,
                }
                existing = validated.get("commands") if isinstance(validated.get("commands"), list) else []
                validated["commands"] = [forced, *existing][: max(0, int(max_commands))]
                if not (validated.get("thoughts") or "").strip():
                    validated["thoughts"] = f"Reviewing latest support session: {session_id.strip()}."

    if validated is not None and not sessions_needing_review:
        latest_review_id = (
            _safe_int(assistant_snapshot.get("latest_review_id")) if isinstance(assistant_snapshot, dict) else None
        )
        latest_review_id = int(latest_review_id or 0)
        digest_last_review_id = _safe_int(state_for_prompt.get("digest_last_review_id"))
        digest_last_review_id = int(digest_last_review_id or 0)
        existing_commands = validated.get("commands") if isinstance(validated.get("commands"), list) else []
        if latest_review_id > digest_last_review_id and not existing_commands:
            validated["commands"] = [
                {
                    "command_type": COMMAND_BUILD_SUPPORT_DIGEST,
                    "priority": -1,
                    "delay_seconds": 0,
                    "payload": {"cursor_review_id": digest_last_review_id},
                }
            ]
            if not (validated.get("thoughts") or "").strip():
                validated["thoughts"] = "Building a digest for newly reviewed support sessions."

    scheduled: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    if validated is not None:
        now = utcnow()
        for cmd in validated["commands"]:
            command_type = str(cmd.get("command_type") or "").strip()
            payload_cmd = dict(cmd.get("payload") or {})
            guard: dict[str, Any] | None = None
            if command_type == COMMAND_REVIEW_SUPPORT_SESSION:
                guard = _orchestrator_review_enqueue_guard(
                    payload=payload_cmd,
                    now=now,
                    current_command_id=int(command_id) if command_id is not None else None,
                )
            elif command_type == COMMAND_BUILD_SUPPORT_DIGEST:
                guard = _orchestrator_digest_enqueue_guard(
                    payload=payload_cmd,
                    now=now,
                    current_command_id=int(command_id) if command_id is not None else None,
                )
            if isinstance(guard, dict):
                skipped.append(
                    {
                        "command_type": command_type,
                        "priority": int(cmd.get("priority") or 0),
                        "delay_seconds": int(cmd.get("delay_seconds") or 0),
                        "payload": payload_cmd,
                        "guard": guard,
                    }
                )
                continue

            delay_seconds = int(cmd.get("delay_seconds") or 0)
            scheduled_at = now + timedelta(seconds=delay_seconds)
            enqueued_id = _enqueue_command(
                command_type=command_type,
                payload=payload_cmd,
                priority=int(cmd.get("priority") or 0),
                available_at=scheduled_at,
            )
            scheduled.append(
                {
                    "id": enqueued_id,
                    "command_type": command_type,
                    "available_at": scheduled_at.isoformat(),
                    "priority": cmd.get("priority"),
                }
            )

        action_summary = _orchestrator_action_summary(
            decision_commands=list(validated.get("commands") or []),
            scheduled=scheduled,
            skipped=skipped,
        )

        requested_tick_seconds = int(validated.get("next_tick_seconds") or min_tick)
        tick_reason = "model" if not llm_invalid else "invalid_output_backoff"
        new_error_streak = 0 if not llm_invalid else prev_error_streak + 1
        has_inflight_duplicates = any(
            isinstance(item, dict)
            and isinstance(item.get("guard"), dict)
            and item["guard"].get("reason") == "duplicate_inflight"
            for item in skipped
        )
        new_idle_streak = 0 if scheduled or has_inflight_duplicates else prev_idle_streak + 1
        tick_seconds = requested_tick_seconds

        if scheduled and sessions_needing_review_count > 1:
            tick_seconds = min_tick
            tick_reason = "review_backlog"
        elif scheduled:
            tick_seconds = min(requested_tick_seconds, base_tick)
            tick_reason = "work"
        elif has_inflight_duplicates:
            tick_seconds = min(requested_tick_seconds, base_tick)
            tick_reason = "work_inflight"
        else:
            if llm_invalid:
                tick_seconds = _orchestrator_error_backoff_seconds(base_seconds=base_tick, error_streak=new_error_streak)
                tick_reason = "invalid_output_backoff"
            else:
                tick_seconds = max(
                    requested_tick_seconds,
                    _orchestrator_idle_backoff_seconds(base_seconds=base_tick, idle_streak=new_idle_streak),
                )
                tick_reason = "idle_backoff"
        tick_seconds = _clamp_int(int(tick_seconds), min_value=min_tick, max_value=max_tick)
        scheduled_tick_id = _schedule_orchestrator_tick(delay_seconds=tick_seconds, current_command_id=command_id)

        with get_agent_session() as agent_db:
            run = agent_db.query(AgentRun).filter(AgentRun.run_id == run_id).one()
            state = _load_orchestrator_state(run)
            # Store action-aligned thoughts for status reporting; keep the raw model
            # thought separately for debugging.
            model_thought = str(validated.get("thoughts") or "").strip()
            action_thought = ""
            if isinstance(action_summary, dict):
                action_thought = str(action_summary.get("summary") or "").strip()
            state["last_model_thought"] = model_thought
            state["last_thought"] = action_thought or model_thought or ""
            state["recent_thoughts"] = (
                [*([str(x) for x in state.get("recent_thoughts") if isinstance(x, str)]), state["last_thought"]]
            )[-10:]
            state["recent_model_thoughts"] = (
                [*([str(x) for x in state.get("recent_model_thoughts") if isinstance(x, str)]), model_thought]
            )[-10:]
            state["last_action_summary"] = action_summary.get("summary") if isinstance(action_summary, dict) else None
            state["last_action"] = action_summary if isinstance(action_summary, dict) else {}
            state["last_action_at"] = utcnow().isoformat()
            state["idle_streak"] = int(new_idle_streak)
            state["error_streak"] = int(new_error_streak)
            state["next_tick_seconds"] = int(tick_seconds)
            state["next_tick_command_id"] = scheduled_tick_id
            state["next_tick_reason"] = tick_reason
            state["base_tick_seconds"] = int(base_tick)
            state["requested_tick_seconds"] = int(requested_tick_seconds)
            _save_orchestrator_state(run=run, state=state)
            agent_db.commit()

        return CommandExecution(
            ok=True,
            result={
                "ok": True,
                "decision": validated,
                "scheduled": scheduled,
                "skipped": skipped,
                "action_summary": action_summary,
                "scheduled_tick_command_id": scheduled_tick_id,
                "assistant_snapshot": assistant_snapshot,
                "severity": "warning" if llm_invalid else "info",
                "llm": {
                    "provider": reply.provider,
                    "model": reply.model,
                    "ok": bool(reply.ok),
                    "meta": reply.meta,
                    "invalid_output_recovered": bool(llm_invalid),
                },
            },
            prompt={"messages": messages, "structured_outputs": {"json": schema}},
            response={"raw": reply.content, "parsed": parsed, "validated": validated, "llm_invalid": llm_invalid},
        )

    raise RuntimeError("Orchestrator reached an unexpected state.")


def _run_orchestrator_tick(*, payload: dict[str, Any], agent_id: str, run_id: str, command_id: int) -> CommandExecution:
    return _run_llm_task_sync(
        _task_orchestrator_tick(payload=payload, agent_id=agent_id, run_id=run_id, command_id=command_id)
    )


def _session_review_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "schema_version": {"type": "integer", "enum": [_SESSION_REVIEW_VERSION]},
            "session_id": {"type": "string", "maxLength": 256},
            "target_message_id": {"type": "integer", "minimum": 1},
            "summary": {"type": "string", "maxLength": 2000},
            "issues": {
                "type": "array",
                "maxItems": 20,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "severity": {"type": "string", "enum": ["info", "warning", "error"]},
                        "category": {
                            "type": "string",
                            "enum": ["tool_use", "accuracy", "ux", "bug", "data", "security", "other"],
                        },
                        "description": {"type": "string", "maxLength": 400},
                        "evidence_message_ids": {
                            "type": "array",
                            "items": {"type": "integer", "minimum": 1},
                            "maxItems": 10,
                        },
                        "suggested_fix": {"type": "string", "maxLength": 400},
                    },
                    "required": ["severity", "category", "description"],
                },
            },
            "repo_search_queries": {
                "type": "array",
                "maxItems": 10,
                "items": {"type": "string", "maxLength": 120},
            },
            "followups": {
                "type": "array",
                "maxItems": 20,
                "items": {"type": "string", "maxLength": 240},
            },
        },
        "required": ["schema_version", "session_id", "target_message_id", "summary", "issues", "repo_search_queries", "followups"],
    }


@prompt_binding("supervisor.session_review.system")
def _session_review_system_prompt() -> str:
    return load_bound_prompt(_session_review_system_prompt).text

@prompt_binding("supervisor.session_review.repair")
def _session_review_repair_system_prompt() -> str:
    return load_bound_prompt(_session_review_repair_system_prompt).text

def _coerce_session_review_output(
    parsed: dict[str, Any] | None,
    *,
    session_id: str,
    target_message_id: int,
) -> dict[str, Any] | None:
    if not isinstance(parsed, dict):
        return None

    if int(parsed.get("schema_version") or 0) == _SESSION_REVIEW_VERSION:
        return parsed

    def normalize_issue(raw_issue: Any) -> dict[str, Any] | None:
        if not isinstance(raw_issue, dict):
            return None
        severity = str(raw_issue.get("severity") or "").strip().lower()
        if severity not in {"info", "warning", "error"}:
            severity = "info"
        category = str(raw_issue.get("category") or "").strip().lower()
        if category not in {"tool_use", "accuracy", "ux", "bug", "data", "security", "other"}:
            category = "other"
        description = raw_issue.get("description")
        if not isinstance(description, str) or not description.strip():
            description = raw_issue.get("message")
        if not isinstance(description, str) or not description.strip():
            description = raw_issue.get("text")
        if not isinstance(description, str) or not description.strip():
            return None

        issue: dict[str, Any] = {
            "severity": severity,
            "category": category,
            "description": _truncate_text(description, limit=400),
        }

        evidence_ids: list[int] = []
        raw_evidence = raw_issue.get("evidence_message_ids")
        if raw_evidence is None:
            raw_evidence = raw_issue.get("evidence")
        if isinstance(raw_evidence, list):
            for item in raw_evidence[:10]:
                value = _safe_int(item)
                if value is not None and value > 0:
                    evidence_ids.append(int(value))
        if evidence_ids:
            issue["evidence_message_ids"] = evidence_ids

        suggested_fix = raw_issue.get("suggested_fix")
        if suggested_fix is None:
            suggested_fix = raw_issue.get("fix")
        if isinstance(suggested_fix, str) and suggested_fix.strip():
            issue["suggested_fix"] = _truncate_text(suggested_fix, limit=400)
        return issue

    def normalize_review_dict(review: dict[str, Any]) -> dict[str, Any]:
        summary = ""
        if isinstance(review.get("summary"), str):
            summary = _truncate_text(review.get("summary"), limit=2000)
        if not summary and isinstance(review.get("review_summary"), str):
            summary = _truncate_text(review.get("review_summary"), limit=2000)

        issues: list[dict[str, Any]] = []
        raw_issues = review.get("issues")
        if isinstance(raw_issues, list):
            for item in raw_issues[:20]:
                normalized = normalize_issue(item)
                if normalized is not None:
                    issues.append(normalized)

        def add_issue_list(key: str, *, category: str, severity: str) -> None:
            raw_items = review.get(key)
            if isinstance(raw_items, str) and raw_items.strip():
                raw_items = [raw_items]
            if not isinstance(raw_items, list):
                return
            for item in raw_items:
                if not isinstance(item, str):
                    continue
                text = item.strip()
                if not text:
                    continue
                issues.append(
                    {
                        "severity": severity,
                        "category": category,
                        "description": _truncate_text(text, limit=400),
                    }
                )

        if not issues:
            add_issue_list("missed_tool_opportunities", category="tool_use", severity="info")
            add_issue_list("incorrect_claims", category="accuracy", severity="warning")
            add_issue_list("confusing_UX_guidance", category="ux", severity="info")
            add_issue_list("bugs", category="bug", severity="warning")

        followups: list[str] = []
        raw_followups = review.get("followups")
        if raw_followups is None:
            raw_followups = review.get("follow-ups")
        if isinstance(raw_followups, str) and raw_followups.strip():
            raw_followups = [raw_followups]
        if isinstance(raw_followups, list):
            for item in raw_followups[:20]:
                if not isinstance(item, str):
                    continue
                text = item.strip()
                if not text:
                    continue
                followups.append(_truncate_text(text, limit=240))

        repo_queries: list[str] = []
        raw_queries = review.get("repo_search_queries")
        if raw_queries is None:
            raw_queries = review.get("repo_search_query")
        if raw_queries is None:
            raw_queries = review.get("repo_search")
        if isinstance(raw_queries, str) and raw_queries.strip():
            raw_queries = [raw_queries]
        if isinstance(raw_queries, list):
            for item in raw_queries[:10]:
                if not isinstance(item, str):
                    continue
                text = item.strip()
                if not text:
                    continue
                repo_queries.append(_truncate_text(text, limit=120))

        if not summary:
            summary = _truncate_text(
                f"Auto-parsed review ({len(issues)} issues, {len(followups)} followups).",
                limit=2000,
            )

        return {
            "schema_version": _SESSION_REVIEW_VERSION,
            "session_id": session_id,
            "target_message_id": int(target_message_id),
            "summary": summary,
            "issues": issues[:20],
            "repo_search_queries": repo_queries[:10],
            "followups": followups[:20],
        }

    # Some models produce a top-level "review_notes" object or omit the
    # schema_version entirely. Accept a small set of variants and coerce them.
    review: dict[str, Any] | None = None
    for key in ("review", "review_notes", "notes", "review_note"):
        candidate = parsed.get(key)
        if isinstance(candidate, dict):
            review = candidate
            break
    if review is None:
        review = parsed

    if not isinstance(review, dict):
        return None
    return normalize_review_dict(review)


def _controller_root_state(state: dict[str, Any]) -> dict[str, Any]:
    raw = state.get("controller")
    root = dict(raw) if isinstance(raw, dict) else {}
    state["controller"] = root
    return root


def _controller_support_post_send_state(state: dict[str, Any]) -> dict[str, Any]:
    root = _controller_root_state(state)
    raw = root.get("support_post_send")
    support_state = dict(raw) if isinstance(raw, dict) else {}
    root["support_post_send"] = support_state
    return support_state


def _controller_set_support_prepared_followup_state(
    state: dict[str, Any],
    prepared: dict[str, Any] | None,
) -> None:
    root = _controller_root_state(state)
    if isinstance(prepared, dict) and prepared:
        root["prepared_followup"] = prepared
    else:
        root.pop("prepared_followup", None)


def _controller_prepared_followup_from_review(
    *,
    review: dict[str, Any],
    target_message_id: int,
    prepared_at: datetime,
) -> dict[str, Any]:
    raw_issues = review.get("issues")
    issues: list[str] = []
    if isinstance(raw_issues, list):
        for item in raw_issues[:3]:
            if not isinstance(item, dict):
                continue
            description = item.get("description")
            if isinstance(description, str) and description.strip():
                issues.append(_truncate_text(description.strip(), limit=240))

    raw_followups = review.get("followups")
    followups: list[str] = []
    if isinstance(raw_followups, str) and raw_followups.strip():
        raw_followups = [raw_followups]
    if isinstance(raw_followups, list):
        for item in raw_followups[:3]:
            if not isinstance(item, str):
                continue
            text = item.strip()
            if text:
                followups.append(_truncate_text(text, limit=240))

    summary = review.get("summary") if isinstance(review.get("summary"), str) else ""
    return {
        "schema_version": 1,
        "for_message_id": int(target_message_id),
        "prepared_at": prepared_at.isoformat(),
        "summary": _truncate_text(summary, limit=600),
        "issues": issues,
        "followups": followups,
    }


def _run_support_post_send_prepare(
    *,
    payload: dict[str, Any],
    agent_id: str,
    run_id: str,
    command_id: int | None = None,
) -> CommandExecution:
    review_execution = _run_support_session_review(
        payload=payload,
        agent_id=agent_id,
        run_id=run_id,
        command_id=command_id,
    )
    if not review_execution.ok:
        return review_execution

    review_result = dict(review_execution.result or {})
    review = review_result.get("review") if isinstance(review_result.get("review"), dict) else {}
    session_id = str(review_result.get("session_id") or payload.get("session_id") or "").strip()
    session_pk = _safe_int(payload.get("session_pk"))
    target_message_id = _safe_int(review_result.get("target_message_id")) or _safe_int(payload.get("target_message_id"))
    if not session_id and session_pk is None:
        return CommandExecution(
            ok=False,
            result={"ok": False, "error": "Missing session for post-send prepare."},
            error="missing_session",
            prompt=review_execution.prompt,
            response=review_execution.response,
        )
    if target_message_id is None or target_message_id <= 0:
        return CommandExecution(
            ok=False,
            result={"ok": False, "error": "Missing target_message_id for post-send prepare."},
            error="missing_target_message_id",
            prompt=review_execution.prompt,
            response=review_execution.response,
        )

    prepared_at = utcnow()
    prepared_followup = _controller_prepared_followup_from_review(
        review=review,
        target_message_id=int(target_message_id),
        prepared_at=prepared_at,
    )

    next_enqueue_meta: dict[str, Any] | None = None
    with get_assistant_session() as assistant_db:
        session = None
        if session_pk is not None:
            session = assistant_db.query(SupportSession).filter(SupportSession.id == int(session_pk)).first()
        if session is None and session_id:
            session = assistant_db.query(SupportSession).filter(SupportSession.session_id == session_id).first()
        if session is None:
            return CommandExecution(
                ok=False,
                result={
                    "ok": False,
                    "error": "Support session not found for post-send prepare.",
                    "session_id": session_id or None,
                },
                error="session_not_found",
                prompt=review_execution.prompt,
                response=review_execution.response,
            )

        state = _load_json_dict(getattr(session, "state_json", None))
        support_state = _controller_support_post_send_state(state)
        support_state["thread_key"] = str(payload.get("thread_key") or support_post_send_thread_key(session.session_id))
        support_state["last_completed_target_message_id"] = int(target_message_id)
        support_state["last_completed_at"] = prepared_at.isoformat()
        support_state.pop("queued_command_id", None)
        if _safe_int(support_state.get("queued_target_message_id")) == int(target_message_id):
            support_state.pop("queued_target_message_id", None)
        _controller_set_support_prepared_followup_state(state, prepared_followup)

        pending_target_message_id = _safe_int(support_state.get("pending_target_message_id"))
        if pending_target_message_id is not None and pending_target_message_id > int(target_message_id):
            with get_agent_session() as agent_db:
                enqueue_result = enqueue_post_send_prepare_command(
                    agent_db=agent_db,
                    source="support_chat",
                    thread_key=str(support_state.get("thread_key") or ""),
                    payload={
                        "session_id": session.session_id,
                        "session_pk": int(session.id),
                        "target_message_id": int(pending_target_message_id),
                        "requested_at": prepared_at.isoformat(),
                    },
                    priority=1,
                    current_command_id=command_id,
                )
            next_enqueue_meta = enqueue_result.as_meta()
            if enqueue_result.enqueued and enqueue_result.command_id is not None:
                support_state["queued_target_message_id"] = int(pending_target_message_id)
                support_state["queued_command_id"] = int(enqueue_result.command_id)
                support_state["last_enqueued_at"] = prepared_at.isoformat()
                support_state.pop("pending_target_message_id", None)
                support_state.pop("pending_requested_at", None)

        session.state_json = _dump_json(state)
        session.updated_at = prepared_at
        assistant_db.commit()

    result_payload = dict(review_result)
    result_payload["prepared_followup"] = prepared_followup
    result_payload["thread_key"] = str(payload.get("thread_key") or support_post_send_thread_key(session_id))
    if next_enqueue_meta is not None:
        result_payload["next_enqueue"] = next_enqueue_meta
    return CommandExecution(
        ok=True,
        result=result_payload,
        prompt=review_execution.prompt,
        response=review_execution.response,
    )


def _run_post_send_prepare(
    *,
    payload: dict[str, Any],
    agent_id: str,
    run_id: str,
    command_id: int | None = None,
) -> CommandExecution:
    source = str(payload.get("source") or "").strip()
    if source == "support_chat":
        return _run_support_post_send_prepare(
            payload=payload,
            agent_id=agent_id,
            run_id=run_id,
            command_id=command_id,
        )
    return CommandExecution(
        ok=True,
        result={
            "ok": True,
            "noop": True,
            "source": source or None,
            "reason": "unsupported_source",
        },
    )


def _run_support_chat_turn(
    *,
    payload: dict[str, Any],
    agent_id: str,
    run_id: str,
    command_id: int | None = None,
) -> CommandExecution:
    request_payload = payload.get("chat_request")
    if not isinstance(request_payload, dict):
        return CommandExecution(
            ok=False,
            result={"ok": False, "error": "Missing chat_request payload."},
            error="missing_chat_request",
        )

    user_id = _safe_int(payload.get("user_id"))

    try:
        from ispec.api.routes import support as support_routes
    except Exception as exc:
        return CommandExecution(
            ok=False,
            result={"ok": False, "error": f"Unable to import support route: {type(exc).__name__}"},
            error="chat_route_import_error",
        )

    try:
        chat_request = support_routes.ChatRequest.model_validate(dict(request_payload))
    except Exception:
        return CommandExecution(
            ok=False,
            result={"ok": False, "error": "Invalid chat_request payload."},
            error="invalid_chat_request",
        )

    queue_meta = dict(chat_request.meta or {})
    queue_meta["_queue_force_inline"] = True
    queue_meta["_queue_agent_id"] = str(agent_id)
    queue_meta["_queue_run_id"] = str(run_id)
    if command_id is not None:
        queue_meta["_queue_command_id"] = int(command_id)
    chat_request = chat_request.model_copy(update={"meta": queue_meta})

    try:
        with get_session() as core_db:
            user = None
            if user_id is not None:
                if int(user_id) == 0:
                    from ispec.api.security import _ApiKeyServiceUser

                    # Queue-mode support chats can legitimately carry the synthetic
                    # API-key assistant user (id=0). Preserve that internal posture
                    # instead of looking for a real auth_user row.
                    user = _ApiKeyServiceUser()
                else:
                    from ispec.db.models import AuthUser

                    user = core_db.query(AuthUser).filter(AuthUser.id == int(user_id)).first()
                    if user is None:
                        return CommandExecution(
                            ok=False,
                            result={"ok": False, "error": f"User not found for queued chat turn (user_id={user_id})."},
                            error="chat_user_not_found",
                        )

            with (
                get_assistant_session() as assistant_db,
                get_agent_session() as agent_db,
                get_omics_session() as omics_db,
                get_schedule_session() as schedule_db,
            ):
                response_obj = support_routes.chat(
                    payload=chat_request,
                    request=None,
                    assistant_db=assistant_db,
                    agent_db=agent_db,
                    core_db=core_db,
                    omics_db=omics_db,
                    schedule_db=schedule_db,
                    user=user,
                )
    except Exception as exc:
        detail = getattr(exc, "detail", None)
        detail_text = str(detail).strip() if detail is not None else ""
        if not detail_text:
            detail_text = f"{type(exc).__name__}: {exc}"
        return CommandExecution(
            ok=False,
            result={"ok": False, "error": detail_text},
            error="queued_chat_turn_failed",
        )

    response_payload: dict[str, Any]
    if hasattr(response_obj, "model_dump"):
        response_payload = response_obj.model_dump(mode="json")
    elif isinstance(response_obj, dict):
        response_payload = dict(response_obj)
    else:
        response_payload = {"sessionId": chat_request.sessionId, "message": str(response_obj)}

    return CommandExecution(
        ok=True,
        result={
            "ok": True,
            "chat_response": response_payload,
            "session_id": chat_request.sessionId,
            "queued_command_id": int(command_id) if command_id is not None else None,
            "user_id": int(user_id) if user_id is not None else None,
        },
    )


def _task_scheduled_assistant_prompt(
    *,
    payload: dict[str, Any],
    agent_id: str,
    run_id: str,
    command_id: int | None = None,
) -> LLMTask:
    job_payload = payload.get("job") if isinstance(payload.get("job"), dict) else payload
    job_name = str(job_payload.get("name") or "").strip()
    prompt_text = _truncate_text(str(job_payload.get("prompt") or "").strip(), limit=6000)
    allowed_tools_raw = job_payload.get("allowed_tools")
    allowed_tool_names = set(_normalize_schedule_tool_names(allowed_tools_raw))
    required_tool = str(job_payload.get("required_tool") or "").strip() or None
    if required_tool and required_tool not in allowed_tool_names:
        allowed_tool_names.add(required_tool)

    max_tool_calls = _safe_int(job_payload.get("max_tool_calls")) or 4
    max_tool_calls = _clamp_int(int(max_tool_calls), min_value=1, max_value=12)

    if not prompt_text:
        return CommandExecution(
            ok=False,
            result={"ok": False, "error": "Missing scheduled assistant prompt.", "job_name": job_name or None},
            error="missing_scheduled_assistant_prompt",
        )
    if not allowed_tool_names:
        return CommandExecution(
            ok=False,
            result={"ok": False, "error": "No allowed_tools configured.", "job_name": job_name or None},
            error="missing_scheduled_assistant_tools",
        )

    scheduled_user = _ScheduledAssistantToolUser()
    tool_schemas = _filter_openai_tools_by_name(
        tools=openai_tools_for_user(scheduled_user),
        allowed_names=allowed_tool_names,
    )
    provided_tool_names = {
        name
        for name in (_openai_tool_name_from_spec(tool) for tool in tool_schemas)
        if isinstance(name, str) and name
    }

    if not provided_tool_names:
        return CommandExecution(
            ok=False,
            result={
                "ok": False,
                "error": "No configured scheduled tools are currently available.",
                "job_name": job_name or None,
                "allowed_tools": sorted(allowed_tool_names),
            },
            error="scheduled_assistant_tools_unavailable",
        )
    if required_tool and required_tool not in provided_tool_names:
        return CommandExecution(
            ok=False,
            result={
                "ok": False,
                "error": f"Required tool {required_tool} is unavailable.",
                "job_name": job_name or None,
                "allowed_tools": sorted(allowed_tool_names),
                "available_tools": sorted(provided_tool_names),
            },
            error="scheduled_assistant_required_tool_unavailable",
        )

    provider = (os.getenv("ISPEC_ASSISTANT_PROVIDER") or "").strip().lower()
    turn_decision_mode = _assistant_turn_decision_mode()
    turn_decision_groups = tool_groups_for_available_tools(provided_tool_names)
    turn_decision_result = None
    turn_decision_runtime_applied = False
    if provider == "vllm" and turn_decision_mode != "off":
        turn_decision_result = run_turn_decision_pipeline(
            generate_reply_fn=generate_reply,
            mode=turn_decision_mode,
            source="scheduled_assistant",
            user_message=prompt_text,
            last_assistant_message=None,
            focused_project_id=None,
            referenced_project_ids=[],
            groups=turn_decision_groups,
            response_modes=["single"],
            contract_caps=list(response_contract_names()),
            extra_context={
                "job_name": job_name or None,
                "required_tool": required_tool,
                "allowed_tools": sorted(provided_tool_names),
            },
        )
        turn_decision_runtime_applied = bool(
            turn_decision_result.ok and turn_decision_mode == "own" and turn_decision_result.decision is not None
        )
    if turn_decision_runtime_applied and turn_decision_result and turn_decision_result.decision is not None:
        selected_tool_names = selected_tool_names_from_decision(
            groups=turn_decision_groups,
            decision=turn_decision_result.decision,
            always_include=[required_tool] if required_tool else None,
        )
        tool_schemas = _filter_openai_tools_by_name(
            tools=tool_schemas,
            allowed_names=selected_tool_names,
        )
        provided_tool_names = {
            name
            for name in (_openai_tool_name_from_spec(tool) for tool in tool_schemas)
            if isinstance(name, str) and name
        }

    system_prompt_rendered = _render_scheduled_assistant_system_prompt(
        allowed_tool_names=provided_tool_names,
        required_tool=required_tool,
    )
    system_prompt = system_prompt_rendered.text
    context_payload: dict[str, Any] = {
        "schema_version": 1,
        "job": {
            "name": job_name or None,
            "schedule": payload.get("schedule") if isinstance(payload.get("schedule"), dict) else None,
            "required_tool": required_tool,
            "allowed_tools": sorted(provided_tool_names),
            "max_tool_calls": int(max_tool_calls),
        },
        "agent": {
            "agent_id": agent_id,
            "run_id": run_id,
            "command_id": int(command_id) if command_id is not None else None,
        },
    }

    tool_messages: list[dict[str, Any]] = []
    tool_calls: list[dict[str, Any]] = []
    llm_trace: list[dict[str, Any]] = []
    used_tool_calls = 0
    required_tool_calls = 0
    queued_command_ids: list[int] = []
    raw_reply = ""
    final_text = ""
    forced_required_tool_round = False
    max_rounds = max(4, int(max_tool_calls) + 3)
    turn_decision_meta = turn_decision_result.as_meta() if turn_decision_result is not None else None

    def execute_tool(tool_name: str, tool_args: dict[str, Any], *, protocol: str) -> dict[str, Any]:
        nonlocal used_tool_calls, required_tool_calls

        if required_tool and tool_name == required_tool and required_tool_calls >= 1:
            return {
                "ok": False,
                "tool": tool_name,
                "error": "Required final tool already executed once for this scheduled job.",
            }
        if used_tool_calls >= max_tool_calls:
            return {
                "ok": False,
                "tool": tool_name,
                "error": "Tool call limit exceeded; no further tools executed.",
            }

        used_tool_calls += 1
        with (
            get_session() as core_db,
            get_assistant_session() as assistant_db,
            get_agent_session() as agent_db,
            get_omics_session() as omics_db,
            get_schedule_session() as schedule_db,
        ):
            tool_payload = run_tool(
                name=tool_name,
                args=tool_args,
                core_db=core_db,
                assistant_db=assistant_db,
                agent_db=agent_db,
                schedule_db=schedule_db,
                omics_db=omics_db,
                user=scheduled_user,
                api_schema=None,
                user_message=prompt_text,
            )

        if required_tool and tool_name == required_tool and bool(tool_payload.get("ok")):
            required_tool_calls += 1
            result_obj = tool_payload.get("result")
            if isinstance(result_obj, dict):
                command_id_obj = _safe_int(result_obj.get("command_id"))
                if command_id_obj is not None and command_id_obj > 0:
                    queued_command_ids.append(int(command_id_obj))

        tool_calls.append(
            {
                "name": tool_name,
                "arguments": dict(tool_args or {}),
                "ok": bool(tool_payload.get("ok")),
                "error": tool_payload.get("error"),
                "result_preview": _tool_result_preview(tool_payload),
                "protocol": protocol,
            }
        )
        return tool_payload

    for llm_round in range(1, max_rounds + 1):
        context_payload["agent"] = {
            **(context_payload.get("agent") if isinstance(context_payload.get("agent"), dict) else {}),
            "round": int(llm_round),
        }
        context_message = _scheduled_assistant_context_message(payload=context_payload)
        tools_for_call = tool_schemas
        tool_choice: str | dict[str, Any] | None = None
        if forced_required_tool_round and required_tool:
            tools_for_call = [
                tool
                for tool in tool_schemas
                if _openai_tool_name_from_spec(tool) == required_tool
            ]
            tool_choice = {"type": "function", "function": {"name": required_tool}}
        elif (
            llm_round == 1
            and turn_decision_runtime_applied
            and turn_decision_result
            and turn_decision_result.decision is not None
            and turn_decision_result.decision.tool_plan.preferred_first_tool in provided_tool_names
        ):
            tool_choice = {
                "type": "function",
                "function": {"name": str(turn_decision_result.decision.tool_plan.preferred_first_tool)},
            }

        messages_for_llm: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "system", "content": context_message},
            {"role": "user", "content": prompt_text},
            *tool_messages,
        ]

        reply = yield InferenceRequest(
            messages=messages_for_llm,
            tools=tools_for_call,
            tool_choice=tool_choice,
            stage="planner",
            observability_context=prompt_observability_context(
                system_prompt_rendered,
                extra={"task": "scheduled_assistant", "job_name": job_name or None},
            ),
        )

        raw_reply = reply.content or ""
        trace_step: dict[str, Any] = {
            "round": int(llm_round),
            "tool_choice": tool_choice,
            "required_tool_round": bool(forced_required_tool_round),
            "provider": reply.provider,
            "model": reply.model,
            "reply_preview": _truncate_text(raw_reply, limit=400),
        }
        if isinstance(reply.meta, dict):
            trace_step["provider_meta"] = {
                "elapsed_ms": reply.meta.get("elapsed_ms"),
                "usage": reply.meta.get("usage"),
                "fallback": reply.meta.get("fallback"),
            }

        if reply.tool_calls:
            tool_messages.append(
                {
                    "role": "assistant",
                    "content": raw_reply,
                    "tool_calls": reply.tool_calls,
                }
            )
            executed_tools: list[str] = []
            for tool_call in reply.tool_calls:
                if not isinstance(tool_call, dict):
                    continue
                call_id = str(tool_call.get("id") or tool_call.get("tool_call_id") or f"call_{used_tool_calls + 1}")
                func_obj = tool_call.get("function") if isinstance(tool_call.get("function"), dict) else {}
                tool_name = str(func_obj.get("name") or "").strip()
                args_raw = func_obj.get("arguments")
                try:
                    if isinstance(args_raw, str):
                        parsed_args = json.loads(args_raw) if args_raw.strip() else {}
                    elif isinstance(args_raw, dict):
                        parsed_args = args_raw
                    else:
                        parsed_args = {}
                except Exception:
                    parsed_args = {}
                if not isinstance(parsed_args, dict):
                    parsed_args = {}
                if not tool_name:
                    continue

                executed_tools.append(tool_name)
                tool_payload = execute_tool(tool_name, parsed_args, protocol="openai")
                tool_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": json.dumps(tool_payload, ensure_ascii=False),
                    }
                )
                tool_messages.append(
                    {"role": "system", "content": format_tool_result_message(tool_name, tool_payload)}
                )

            if executed_tools:
                trace_step["executed_tools"] = executed_tools
            llm_trace.append(trace_step)
            forced_required_tool_round = False
            continue

        tool_call = parse_tool_call(raw_reply)
        if tool_call is not None:
            tool_name, tool_args = tool_call
            tool_payload = execute_tool(tool_name, tool_args, protocol="line")
            tool_call_line = extract_tool_call_line(raw_reply) or raw_reply.strip()
            tool_messages.extend(
                [
                    {"role": "assistant", "content": tool_call_line},
                    {"role": "system", "content": format_tool_result_message(tool_name, tool_payload)},
                ]
            )
            trace_step["executed_tools"] = [tool_name]
            llm_trace.append(trace_step)
            forced_required_tool_round = False
            continue

        _, parsed_final = split_plan_final(raw_reply)
        final_text = parsed_final.strip() or raw_reply.strip()
        trace_step["final_preview"] = _truncate_text(final_text, limit=400)
        llm_trace.append(trace_step)

        if required_tool and required_tool_calls <= 0:
            if forced_required_tool_round:
                return CommandExecution(
                    ok=False,
                    result={
                        "ok": False,
                        "error": f"Required tool {required_tool} was not called.",
                        "job_name": job_name or None,
                        "final_text": final_text,
                        "tool_calls": tool_calls,
                        "required_tool": required_tool,
                        "turn_decision": turn_decision_meta,
                    },
                    error="scheduled_assistant_required_tool_not_called",
                    prompt={
                        "system_prompt": system_prompt,
                        "context": context_payload,
                        "required_tool": required_tool,
                    },
                    response={"llm_trace": llm_trace, "raw_reply": _truncate_text(raw_reply, limit=2000)},
                )

            tool_messages.append({"role": "assistant", "content": raw_reply})
            tool_messages.append(
                {
                    "role": "system",
                    "content": (
                        f"You must now call {required_tool} exactly once.\n"
                        "Use confirm=true.\n"
                        "Put the finalized staff-facing message into the tool call.\n"
                        "Do not return FINAL until the tool call succeeds.\n"
                        "Draft message to send:\n"
                        f"{_truncate_text(final_text, limit=3500)}"
                    ),
                }
            )
            forced_required_tool_round = True
            continue

        return CommandExecution(
            ok=True,
            result={
                "ok": True,
                "job_name": job_name or None,
                "required_tool": required_tool,
                "required_tool_called": bool(required_tool_calls > 0) if required_tool else None,
                "required_tool_call_count": int(required_tool_calls),
                "queued_command_ids": queued_command_ids,
                "used_tool_calls": int(used_tool_calls),
                "tool_calls": tool_calls,
                "turn_decision": turn_decision_meta,
                "final_text": final_text,
                "llm": {
                    "provider": reply.provider,
                    "model": reply.model,
                    "meta": reply.meta,
                },
            },
            prompt={
                "system_prompt": system_prompt,
                "context": context_payload,
                "allowed_tools": sorted(provided_tool_names),
            },
            response={
                "llm_trace": llm_trace,
                "raw_reply": _truncate_text(raw_reply, limit=2000),
            },
        )

    return CommandExecution(
        ok=False,
        result={
            "ok": False,
            "error": "Scheduled assistant prompt exceeded max rounds.",
            "job_name": job_name or None,
            "required_tool": required_tool,
            "required_tool_called": bool(required_tool_calls > 0) if required_tool else None,
            "tool_calls": tool_calls,
            "turn_decision": turn_decision_meta,
            "final_text": final_text,
        },
        error="scheduled_assistant_max_rounds_exceeded",
        prompt={
            "system_prompt": system_prompt,
            "context": context_payload,
            "allowed_tools": sorted(provided_tool_names),
        },
        response={"llm_trace": llm_trace, "raw_reply": _truncate_text(raw_reply, limit=2000)},
    )


def _run_scheduled_assistant_prompt(
    *,
    payload: dict[str, Any],
    agent_id: str,
    run_id: str,
    command_id: int | None = None,
) -> CommandExecution:
    return _run_llm_task_sync(
        _task_scheduled_assistant_prompt(
            payload=payload,
            agent_id=agent_id,
            run_id=run_id,
            command_id=command_id,
        )
    )


def _openai_tool_name_from_spec(tool: dict[str, Any]) -> str | None:
    if not isinstance(tool, dict):
        return None
    func_obj = tool.get("function")
    if not isinstance(func_obj, dict):
        return None
    name = func_obj.get("name")
    if not isinstance(name, str) or not name.strip():
        return None
    return name.strip()


def _filter_openai_tools_by_name(*, tools: list[dict[str, Any]], allowed_names: set[str]) -> list[dict[str, Any]]:
    if not allowed_names:
        return []
    filtered: list[dict[str, Any]] = []
    for tool in tools:
        name = _openai_tool_name_from_spec(tool)
        if name and name in allowed_names:
            filtered.append(tool)
    return filtered


def _tool_result_preview(payload: dict[str, Any], *, max_chars: int = 2000) -> str | None:
    result = payload.get("result")
    if result is None:
        return None
    try:
        rendered = json.dumps(result, ensure_ascii=False, separators=(",", ":"))
    except Exception:
        return None
    if max_chars > 0 and len(rendered) > max_chars:
        return rendered[:max_chars] + "…"
    return rendered


def _scheduled_assistant_context_message(*, payload: dict[str, Any]) -> str:
    return "SCHEDULED_ASSISTANT_CONTEXT v1:\n" + json.dumps(payload, ensure_ascii=False)


@prompt_binding("supervisor.scheduled_assistant.system")
def _scheduled_assistant_system_prompt(*, allowed_tool_names: set[str], required_tool: str | None) -> str:
    return _render_scheduled_assistant_system_prompt(
        allowed_tool_names=allowed_tool_names,
        required_tool=required_tool,
    ).text


def _render_scheduled_assistant_system_prompt(
    *,
    allowed_tool_names: set[str],
    required_tool: str | None,
):
    required_tool_block = ""
    if required_tool:
        required_tool_block = "\n".join(
            [
                f"- After gathering the needed information, call {required_tool} exactly once.",
                "- Do not stop after drafting the message; complete the required final tool call.",
                "- If the required tool succeeds, finish with a short FINAL confirming the action.",
            ]
        )
    return load_bound_prompt(
        _scheduled_assistant_system_prompt,
        values={
            "planner_prompt": _system_prompt_planner(
                tools_available=bool(allowed_tool_names),
                response_format="single",
                tool_names=allowed_tool_names,
            ).strip(),
            "required_tool_block": required_tool_block,
        },
    )


def _task_support_session_review(
    *,
    payload: dict[str, Any],
    agent_id: str,
    run_id: str,
    command_id: int | None = None,
) -> LLMTask:
    provider = (os.getenv("ISPEC_ASSISTANT_PROVIDER") or "").strip().lower()
    if provider != "vllm":
        return CommandExecution(
            ok=False,
            result={"ok": False, "error": "Session review requires ISPEC_ASSISTANT_PROVIDER=vllm."},
            error="provider_not_vllm",
        )

    session_id = str(payload.get("session_id") or "").strip()
    session_pk = _safe_int(payload.get("session_pk"))
    if not session_id and session_pk is None:
        return CommandExecution(
            ok=False,
            result={"ok": False, "error": "Missing session_id/session_pk."},
            error="missing_session",
        )

    max_messages = _clamp_int(_safe_int(payload.get("max_messages")) or 40, min_value=10, max_value=120)

    with get_assistant_session() as assistant_db:
        session = None
        if session_pk is not None:
            session = assistant_db.query(SupportSession).filter(SupportSession.id == int(session_pk)).first()
        if session is None and session_id:
            session = (
                assistant_db.query(SupportSession).filter(SupportSession.session_id == session_id).first()
            )
        if session is None:
            return CommandExecution(
                ok=False,
                result={"ok": False, "error": "Support session not found.", "session_id": session_id},
                error="session_not_found",
            )

        state = _load_json_dict(getattr(session, "state_json", None))
        latest_review = (
            assistant_db.query(SupportSessionReview)
            .filter(SupportSessionReview.session_pk == int(session.id))
            .order_by(SupportSessionReview.target_message_id.desc(), SupportSessionReview.id.desc())
            .first()
        )
        reviewed_up_to = int(latest_review.target_message_id) if latest_review is not None else 0

        last_message = (
            assistant_db.query(SupportMessage)
            .filter(SupportMessage.session_pk == session.id)
            .order_by(SupportMessage.id.desc())
            .first()
        )
        if last_message is None:
            return CommandExecution(ok=True, result={"ok": True, "noop": True, "empty": True})
        last_assistant = (
            assistant_db.query(SupportMessage)
            .filter(SupportMessage.session_pk == session.id)
            .filter(SupportMessage.role == "assistant")
            .order_by(SupportMessage.id.desc())
            .first()
        )
        if last_assistant is None:
            return CommandExecution(
                ok=True,
                result={
                    "ok": True,
                    "noop": True,
                    "session_id": session.session_id,
                    "reason": "no_assistant_messages",
                },
            )

        requested_target_id = _safe_int(payload.get("target_message_id"))
        if requested_target_id is None or requested_target_id <= 0:
            target_id = int(last_assistant.id)
        else:
            candidate = (
                assistant_db.query(SupportMessage)
                .filter(SupportMessage.session_pk == session.id)
                .filter(SupportMessage.role == "assistant")
                .filter(SupportMessage.id <= int(requested_target_id))
                .order_by(SupportMessage.id.desc())
                .first()
            )
            if candidate is None:
                return CommandExecution(
                    ok=True,
                    result={
                        "ok": True,
                        "noop": True,
                        "session_id": session.session_id,
                        "reason": "target_before_first_assistant",
                        "requested_target_message_id": int(requested_target_id),
                        "last_assistant_message_id": int(last_assistant.id),
                    },
                )
            target_id = int(candidate.id)
        if target_id <= reviewed_up_to:
            return CommandExecution(
                ok=True,
                result={
                    "ok": True,
                    "noop": True,
                    "session_id": session.session_id,
                    "reviewed_up_to_id": reviewed_up_to,
                    "target_id": target_id,
                },
            )

        rows = (
            assistant_db.query(SupportMessage)
            .filter(SupportMessage.session_pk == session.id)
            .filter(SupportMessage.id <= target_id)
            .order_by(SupportMessage.id.desc())
            .limit(max_messages)
            .all()
        )
        rows.reverse()
        transcript: list[dict[str, Any]] = []
        for row in rows:
            transcript.append(
                {
                    "id": int(row.id),
                    "role": row.role,
                    "content": _truncate_text(row.content, limit=800),
                }
            )

        context = {
            "schema_version": 1,
            "agent": {"agent_id": agent_id, "run_id": run_id},
            "session": {
                "id": session.session_id,
                "pk": int(session.id),
                "user_id": int(session.user_id) if session.user_id is not None else None,
                "reviewed_up_to_id": reviewed_up_to,
                "target_message_id": target_id,
                "state": {
                    "conversation_memory": state.get("conversation_memory") if isinstance(state.get("conversation_memory"), dict) else None,
                    "conversation_summary": _truncate_text(state.get("conversation_summary"), limit=1200) if isinstance(state.get("conversation_summary"), str) else None,
                },
            },
            "transcript": transcript,
        }

        schema = _session_review_schema()
        prompt = load_bound_prompt(_session_review_system_prompt)
        messages = [
            {"role": "system", "content": prompt.text},
            {"role": "user", "content": json.dumps(context, ensure_ascii=False)},
        ]
        temperature = _structured_llm_temperature("SESSION_REVIEW", default=0.0)
        repair_temperature = _structured_llm_repair_temperature("SESSION_REVIEW", default=0.25)
        max_repairs = _structured_llm_max_repairs("SESSION_REVIEW", default=1)

        vllm_extra_body = {"structured_outputs": {"json": schema}, "temperature": temperature, "max_tokens": 1200}
        reply = yield InferenceRequest(messages=messages, tools=None, vllm_extra_body=vllm_extra_body, observability_context=prompt_observability_context(prompt, extra={"task": "session_review"}))

        attempts: list[dict[str, Any]] = []
        parsed = _parse_json_object(reply.content)
        coerced = _coerce_session_review_output(
            parsed if isinstance(parsed, dict) else None,
            session_id=session.session_id,
            target_message_id=int(target_id),
        )
        attempts.append(
            {
                "attempt": 1,
                "temperature": float(temperature),
                "raw": reply.content,
                "parsed": parsed,
                "coerced": coerced,
                "llm": {"provider": reply.provider, "model": reply.model, "meta": reply.meta},
            }
        )

        recovered = False
        if (not isinstance(coerced, dict) or int(coerced.get("schema_version") or 0) != _SESSION_REVIEW_VERSION) and (
            max_repairs > 0
        ):
            repair_prompt = load_bound_prompt(_session_review_repair_system_prompt)
            repair_messages = [
                {"role": "system", "content": repair_prompt.text},
                {"role": "user", "content": json.dumps({"context": context, "previous_output": reply.content}, ensure_ascii=False)},
            ]
            repair_body = {"structured_outputs": {"json": schema}, "temperature": repair_temperature, "max_tokens": 1200}
            repair_reply = yield InferenceRequest(messages=repair_messages, tools=None, vllm_extra_body=repair_body, observability_context=prompt_observability_context(repair_prompt, extra={"task": "session_review_repair"}))
            parsed2 = _parse_json_object(repair_reply.content)
            coerced2 = _coerce_session_review_output(
                parsed2 if isinstance(parsed2, dict) else None,
                session_id=session.session_id,
                target_message_id=int(target_id),
            )
            attempts.append(
                {
                    "attempt": 2,
                    "temperature": float(repair_temperature),
                    "raw": repair_reply.content,
                    "parsed": parsed2,
                    "coerced": coerced2,
                    "llm": {"provider": repair_reply.provider, "model": repair_reply.model, "meta": repair_reply.meta},
                }
            )
            if isinstance(coerced2, dict) and int(coerced2.get("schema_version") or 0) == _SESSION_REVIEW_VERSION:
                reply = repair_reply
                parsed = parsed2
                coerced = coerced2
                recovered = True

        if not isinstance(coerced, dict) or int(coerced.get("schema_version") or 0) != _SESSION_REVIEW_VERSION:
            return CommandExecution(
                ok=False,
                result={
                    "ok": False,
                    "error": "Invalid review output.",
                    "raw": _truncate_text(reply.content, limit=1200),
                    "attempts": len(attempts),
                    "llm": {"provider": reply.provider, "model": reply.model, "meta": reply.meta},
                },
                error="invalid_review_output",
                prompt={"messages": messages, "structured_outputs": {"json": schema}, "vllm_extra_body": vllm_extra_body},
                response={"attempts": attempts},
            )

        record = (
            assistant_db.query(SupportSessionReview)
            .filter(SupportSessionReview.session_pk == int(session.id))
            .filter(SupportSessionReview.target_message_id == int(target_id))
            .first()
        )
        if record is None:
            record = SupportSessionReview(
                session_pk=int(session.id),
                target_message_id=int(target_id),
            )
            assistant_db.add(record)

        record.schema_version = int(coerced.get("schema_version") or _SESSION_REVIEW_VERSION)
        record.review_json = coerced
        record.agent_id = agent_id
        record.run_id = run_id
        record.command_id = int(command_id) if command_id is not None else None

        assistant_db.commit()

        return CommandExecution(
            ok=True,
            result={
                "ok": True,
                "session_id": session.session_id,
                "target_message_id": target_id,
                "review": coerced,
                "llm": {
                    "provider": reply.provider,
                    "model": reply.model,
                    "meta": reply.meta,
                    "invalid_output_recovered": bool(recovered),
                    "attempts": len(attempts),
                },
            },
            prompt={"messages": messages, "structured_outputs": {"json": schema}, "vllm_extra_body": vllm_extra_body},
            response={"attempts": attempts},
        )


def _run_support_session_review(
    *,
    payload: dict[str, Any],
    agent_id: str,
    run_id: str,
    command_id: int | None = None,
) -> CommandExecution:
    return _run_llm_task_sync(
        _task_support_session_review(payload=payload, agent_id=agent_id, run_id=run_id, command_id=command_id)
    )


def _structured_llm_input_token_budget(task: str, *, default: int) -> int:
    value = _safe_int(os.getenv(f"ISPEC_SUPERVISOR_{task}_MAX_INPUT_TOKENS"))
    if value is None:
        value = _safe_int(os.getenv("ISPEC_SUPERVISOR_MAX_INPUT_TOKENS")) or int(default)
    return _clamp_int(int(value), min_value=512, max_value=32768)


def _support_digest_prompt_review_item(item: dict[str, Any], *, compact_level: int = 0) -> dict[str, Any]:
    review = item.get("review") if isinstance(item.get("review"), dict) else {}
    level = max(0, int(compact_level))
    summary_limit = 320 if level <= 0 else (180 if level == 1 else 96)
    issue_limit = 180 if level <= 0 else (100 if level == 1 else 72)
    followup_limit = 180 if level <= 0 else (100 if level == 1 else 72)
    query_limit = 120 if level <= 0 else 80
    max_issue_items = 3 if level <= 0 else (2 if level == 1 else 1)
    max_followups = 3 if level <= 0 else (2 if level == 1 else 1)
    max_queries = 3 if level <= 0 else (1 if level == 1 else 0)

    summary = review.get("summary") if isinstance(review.get("summary"), str) else ""
    summary = _truncate_text(summary, limit=summary_limit)

    issues: list[str] = []
    raw_issues = review.get("issues")
    if isinstance(raw_issues, list):
        for raw_issue in raw_issues[:max_issue_items]:
            if not isinstance(raw_issue, dict):
                continue
            description = raw_issue.get("description")
            if not isinstance(description, str) or not description.strip():
                continue
            issues.append(_truncate_text(description.strip(), limit=issue_limit))

    followups: list[str] = []
    raw_followups = review.get("followups")
    if isinstance(raw_followups, str) and raw_followups.strip():
        raw_followups = [raw_followups]
    if isinstance(raw_followups, list):
        for raw_followup in raw_followups[:max_followups]:
            if not isinstance(raw_followup, str):
                continue
            text = raw_followup.strip()
            if not text:
                continue
            followups.append(_truncate_text(text, limit=followup_limit))

    repo_search_queries: list[str] = []
    raw_queries = review.get("repo_search_queries")
    if isinstance(raw_queries, str) and raw_queries.strip():
        raw_queries = [raw_queries]
    if isinstance(raw_queries, list):
        for raw_query in raw_queries[:max_queries]:
            if not isinstance(raw_query, str):
                continue
            text = raw_query.strip()
            if not text:
                continue
            repo_search_queries.append(_truncate_text(text, limit=query_limit))

    issues_count = len(raw_issues) if isinstance(raw_issues, list) else 0
    return {
        "review_id": int(item.get("review_id") or 0),
        "session_id": str(item.get("session_id") or "").strip(),
        "user_id": int(item.get("user_id")) if isinstance(item.get("user_id"), int) else None,
        "target_message_id": int(item.get("target_message_id") or 0),
        "updated_at": item.get("updated_at"),
        "summary": summary,
        "issues_count": int(max(0, issues_count)),
        "top_issues": issues,
        "followups": followups,
        "repo_search_queries": repo_search_queries,
    }


def _fit_support_digest_reviews_to_budget(
    *,
    review_items: list[dict[str, Any]],
    agent_id: str,
    run_id: str,
    command_id: int | None,
    cursor_review_id: int,
    max_input_tokens: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    selected = list(review_items)
    compact_level = 0
    prompt_items = [_support_digest_prompt_review_item(item, compact_level=compact_level) for item in selected]

    while True:
        context = {
            "schema_version": 1,
            "agent": {"agent_id": agent_id, "run_id": run_id, "command_id": int(command_id) if command_id else None},
            "cursor_review_id": int(cursor_review_id),
            "reviews": prompt_items,
        }
        messages = [
            {"role": "system", "content": _support_digest_system_prompt()},
            {"role": "user", "content": json.dumps(context, ensure_ascii=False)},
        ]
        estimated_tokens = estimate_tokens_for_messages(messages)
        if estimated_tokens <= int(max_input_tokens) or len(selected) <= 1:
            if estimated_tokens <= int(max_input_tokens) or len(selected) <= 1 and compact_level >= 2:
                return selected, prompt_items, int(estimated_tokens)
            compact_level += 1
            prompt_items = [
                _support_digest_prompt_review_item(item, compact_level=compact_level)
                for item in selected
            ]
            continue
        selected = selected[:-1]
        compact_level = 0
        prompt_items = [_support_digest_prompt_review_item(item, compact_level=compact_level) for item in selected]


def _support_digest_schema(*, max_sessions: int) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "schema_version": {"type": "integer", "enum": [_SUPPORT_DIGEST_VERSION]},
            "from_review_id": {"type": "integer", "minimum": 0},
            "to_review_id": {"type": "integer", "minimum": 0},
            "summary": {"type": "string", "maxLength": 2000},
            "highlights": {
                "type": "array",
                "maxItems": 20,
                "items": {"type": "string", "maxLength": 240},
            },
            "followups": {
                "type": "array",
                "maxItems": 20,
                "items": {"type": "string", "maxLength": 240},
            },
            "sessions": {
                "type": "array",
                "maxItems": max(0, int(max_sessions)),
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "session_id": {"type": "string", "maxLength": 256},
                        "user_id": {"type": ["integer", "null"]},
                        "review_id": {"type": "integer", "minimum": 1},
                        "target_message_id": {"type": "integer", "minimum": 1},
                        "summary": {"type": "string", "maxLength": 600},
                        "issues_count": {"type": "integer", "minimum": 0, "maximum": 50},
                        "followups": {
                            "type": "array",
                            "maxItems": 10,
                            "items": {"type": "string", "maxLength": 240},
                        },
                    },
                    "required": [
                        "session_id",
                        "user_id",
                        "review_id",
                        "target_message_id",
                        "summary",
                        "issues_count",
                        "followups",
                    ],
                },
            },
        },
        "required": [
            "schema_version",
            "from_review_id",
            "to_review_id",
            "summary",
            "highlights",
            "followups",
            "sessions",
        ],
    }


@prompt_binding("supervisor.support_digest.system")
def _support_digest_system_prompt() -> str:
    return load_bound_prompt(_support_digest_system_prompt).text

@prompt_binding("supervisor.support_digest.repair")
def _support_digest_repair_system_prompt() -> str:
    return load_bound_prompt(_support_digest_repair_system_prompt).text

def _coerce_support_digest_output(
    parsed: dict[str, Any] | None,
    *,
    from_review_id: int,
    to_review_id: int,
) -> dict[str, Any] | None:
    if not isinstance(parsed, dict):
        return None

    digest: dict[str, Any] | None = None
    for key in ("digest", "report", "summary_notes"):
        candidate = parsed.get(key)
        if isinstance(candidate, dict):
            digest = candidate
            break
    if digest is None:
        digest = parsed

    if not isinstance(digest, dict):
        return None

    expected_keys = {"summary", "highlights", "followups", "sessions", "from_review_id", "to_review_id"}
    if not any(key in digest for key in expected_keys):
        return None

    summary = digest.get("summary") if isinstance(digest.get("summary"), str) else ""
    summary = _truncate_text(summary, limit=2000)
    if not summary:
        summary = _truncate_text("Support digest generated.", limit=2000)

    highlights: list[str] = []
    raw_highlights = digest.get("highlights")
    if isinstance(raw_highlights, list):
        for item in raw_highlights:
            if not isinstance(item, str):
                continue
            text = item.strip()
            if not text:
                continue
            highlights.append(_truncate_text(text, limit=240))

    followups: list[str] = []
    raw_followups = digest.get("followups")
    if isinstance(raw_followups, str) and raw_followups.strip():
        raw_followups = [raw_followups]
    if isinstance(raw_followups, list):
        for item in raw_followups:
            if not isinstance(item, str):
                continue
            text = item.strip()
            if not text:
                continue
            followups.append(_truncate_text(text, limit=240))

    sessions: list[dict[str, Any]] = []
    raw_sessions = digest.get("sessions")
    if isinstance(raw_sessions, list):
        for item in raw_sessions:
            if not isinstance(item, dict):
                continue
            session_id = item.get("session_id")
            if not isinstance(session_id, str) or not session_id.strip():
                continue
            review_id = _safe_int(item.get("review_id")) or 0
            target_message_id = _safe_int(item.get("target_message_id")) or 0
            if review_id <= 0 or target_message_id <= 0:
                continue
            user_id_raw = item.get("user_id")
            user_id = int(user_id_raw) if isinstance(user_id_raw, int) and user_id_raw > 0 else None
            item_summary = item.get("summary") if isinstance(item.get("summary"), str) else ""
            item_summary = _truncate_text(item_summary, limit=600)
            issues_count = _safe_int(item.get("issues_count")) or 0
            issues_count = _clamp_int(int(issues_count), min_value=0, max_value=50)
            item_followups: list[str] = []
            followups_raw = item.get("followups")
            if isinstance(followups_raw, str) and followups_raw.strip():
                followups_raw = [followups_raw]
            if isinstance(followups_raw, list):
                for value in followups_raw:
                    if not isinstance(value, str):
                        continue
                    text = value.strip()
                    if not text:
                        continue
                    item_followups.append(_truncate_text(text, limit=240))
            sessions.append(
                {
                    "session_id": session_id.strip(),
                    "user_id": user_id,
                    "review_id": int(review_id),
                    "target_message_id": int(target_message_id),
                    "summary": item_summary,
                    "issues_count": int(issues_count),
                    "followups": item_followups[:10],
                }
            )

    return {
        "schema_version": _SUPPORT_DIGEST_VERSION,
        "from_review_id": int(from_review_id),
        "to_review_id": int(to_review_id),
        "summary": summary,
        "highlights": highlights[:20],
        "followups": followups[:20],
        "sessions": sessions,
    }


def _task_support_digest(
    *,
    payload: dict[str, Any],
    agent_id: str,
    run_id: str,
    command_id: int | None = None,
) -> LLMTask:
    provider = (os.getenv("ISPEC_ASSISTANT_PROVIDER") or "").strip().lower()
    if provider != "vllm":
        return CommandExecution(
            ok=False,
            result={"ok": False, "error": "Support digest requires ISPEC_ASSISTANT_PROVIDER=vllm."},
            error="provider_not_vllm",
        )

    cursor_review_id = _safe_int(payload.get("cursor_review_id"))
    if cursor_review_id is None:
        cursor_review_id = _safe_int(payload.get("from_review_id"))
    cursor_review_id = int(cursor_review_id or 0)

    max_reviews = _clamp_int(_safe_int(payload.get("max_reviews")) or 20, min_value=1, max_value=100)
    max_input_tokens = _structured_llm_input_token_budget("SUPPORT_DIGEST", default=5200)

    review_items: list[dict[str, Any]] = []
    with get_assistant_session() as assistant_db:
        rows = (
            assistant_db.query(
                SupportSessionReview.id,
                SupportSessionReview.target_message_id,
                SupportSessionReview.updated_at,
                SupportSessionReview.review_json,
                SupportSession.session_id,
                SupportSession.user_id,
            )
            .join(SupportSession, SupportSessionReview.session_pk == SupportSession.id)
            .filter(SupportSessionReview.id > cursor_review_id)
            .order_by(SupportSessionReview.id.asc())
            .limit(max_reviews)
            .all()
        )
        for review_id, target_message_id, updated_at, review_json, session_id, user_id in rows:
            review_id_int = int(review_id or 0)
            if review_id_int <= 0:
                continue
            target_message_id_int = int(target_message_id or 0)
            review_items.append(
                {
                    "review_id": review_id_int,
                    "session_id": str(session_id),
                    "user_id": int(user_id) if user_id is not None else None,
                    "target_message_id": target_message_id_int,
                    "updated_at": updated_at.isoformat() if updated_at else None,
                    "review": review_json if isinstance(review_json, dict) else None,
                }
            )

    if not review_items:
        return CommandExecution(
            ok=True,
            result={
                "ok": True,
                "noop": True,
                "cursor_review_id": int(cursor_review_id),
                "max_reviews": int(max_reviews),
            },
        )

    review_items_prompt, prompt_review_items, prompt_input_tokens = _fit_support_digest_reviews_to_budget(
        review_items=review_items,
        agent_id=agent_id,
        run_id=run_id,
        command_id=command_id,
        cursor_review_id=int(cursor_review_id),
        max_input_tokens=int(max_input_tokens),
    )
    if len(review_items_prompt) < len(review_items):
        logger.info(
            "Trimmed support digest batch from %s to %s review(s) to fit prompt budget est_tokens=%s budget=%s",
            len(review_items),
            len(review_items_prompt),
            int(prompt_input_tokens),
            int(max_input_tokens),
        )
    review_items = list(review_items_prompt)
    evidence_message_ids = [
        int(item["target_message_id"])
        for item in review_items
        if int(item.get("target_message_id") or 0) > 0
    ]

    from_review_id = int(review_items[0]["review_id"])
    to_review_id = int(review_items[-1]["review_id"])

    context = {
        "schema_version": 1,
        "agent": {"agent_id": agent_id, "run_id": run_id, "command_id": int(command_id) if command_id else None},
        "cursor_review_id": int(cursor_review_id),
        "reviews": prompt_review_items,
    }

    schema = _support_digest_schema(max_sessions=len(review_items))
    prompt = load_bound_prompt(_support_digest_system_prompt)
    messages = [
        {"role": "system", "content": prompt.text},
        {"role": "user", "content": json.dumps(context, ensure_ascii=False)},
    ]
    temperature = _structured_llm_temperature("SUPPORT_DIGEST", default=0.0)
    repair_temperature = _structured_llm_repair_temperature("SUPPORT_DIGEST", default=0.25)
    max_repairs = _structured_llm_max_repairs("SUPPORT_DIGEST", default=1)

    vllm_extra_body = {"structured_outputs": {"json": schema}, "temperature": temperature, "max_tokens": 1200}
    reply = yield InferenceRequest(messages=messages, tools=None, vllm_extra_body=vllm_extra_body, observability_context=prompt_observability_context(prompt, extra={"task": "support_digest"}))

    attempts: list[dict[str, Any]] = []
    parsed = _parse_json_object(reply.content)
    coerced = _coerce_support_digest_output(
        parsed if isinstance(parsed, dict) else None,
        from_review_id=from_review_id,
        to_review_id=to_review_id,
    )
    attempts.append(
        {
            "attempt": 1,
            "temperature": float(temperature),
            "raw": reply.content,
            "parsed": parsed,
            "coerced": coerced,
            "llm": {"provider": reply.provider, "model": reply.model, "meta": reply.meta},
        }
    )

    recovered = False
    if not isinstance(coerced, dict) and max_repairs > 0:
        repair_prompt = load_bound_prompt(_support_digest_repair_system_prompt)
        repair_messages = [
            {"role": "system", "content": repair_prompt.text},
            {"role": "user", "content": json.dumps({"context": context, "previous_output": reply.content}, ensure_ascii=False)},
        ]
        repair_body = {"structured_outputs": {"json": schema}, "temperature": repair_temperature, "max_tokens": 1200}
        repair_reply = yield InferenceRequest(messages=repair_messages, tools=None, vllm_extra_body=repair_body, observability_context=prompt_observability_context(repair_prompt, extra={"task": "support_digest_repair"}))
        parsed2 = _parse_json_object(repair_reply.content)
        coerced2 = _coerce_support_digest_output(
            parsed2 if isinstance(parsed2, dict) else None,
            from_review_id=from_review_id,
            to_review_id=to_review_id,
        )
        attempts.append(
            {
                "attempt": 2,
                "temperature": float(repair_temperature),
                "raw": repair_reply.content,
                "parsed": parsed2,
                "coerced": coerced2,
                "llm": {"provider": repair_reply.provider, "model": repair_reply.model, "meta": repair_reply.meta},
            }
        )
        if isinstance(coerced2, dict):
            reply = repair_reply
            parsed = parsed2
            coerced = coerced2
            recovered = True

    if not isinstance(coerced, dict):
        return CommandExecution(
            ok=False,
            result={
                "ok": False,
                "error": "Invalid support digest output.",
                "cursor_review_id": int(cursor_review_id),
                "from_review_id": from_review_id,
                "to_review_id": to_review_id,
                "raw": _truncate_text(reply.content, limit=1200),
                "attempts": len(attempts),
                "llm": {"provider": reply.provider, "model": reply.model, "meta": reply.meta},
            },
            error="invalid_digest_output",
            prompt={"messages": messages, "structured_outputs": {"json": schema}, "vllm_extra_body": vllm_extra_body},
            response={"attempts": attempts},
        )

    digest_id: int | None = None
    with get_assistant_session() as assistant_db:
        record = SupportMemory(
            session_pk=None,
            user_id=0,
            kind="digest",
            key="global",
            value_json=_dump_json(coerced),
        )
        assistant_db.add(record)
        assistant_db.flush()
        digest_id = int(record.id)

        for message_id in sorted({int(x) for x in evidence_message_ids if int(x) > 0}):
            assistant_db.add(
                SupportMemoryEvidence(
                    memory_id=int(record.id),
                    message_id=int(message_id),
                    weight=1.0,
                )
            )
        assistant_db.commit()

    if digest_id is not None:
        try:
            with get_agent_session() as agent_db:
                run = agent_db.query(AgentRun).filter(AgentRun.run_id == run_id).one()
                state = _load_orchestrator_state(run)
                state["digest_last_review_id"] = int(to_review_id)
                state["digest_last_at"] = utcnow().isoformat()
                _save_orchestrator_state(run=run, state=state)
                agent_db.commit()
        except Exception:
            logger.exception("Failed to persist digest cursor in orchestrator state (run_id=%s)", run_id)

    return CommandExecution(
        ok=True,
        result={
            "ok": True,
            "digest_id": int(digest_id) if digest_id is not None else None,
            "cursor_review_id": int(cursor_review_id),
                "from_review_id": from_review_id,
                "to_review_id": to_review_id,
                "review_count": len(review_items),
                "reviews_available": len(rows),
                "reviews_deferred": max(0, len(rows) - len(review_items)),
                "input_tokens_estimate": int(prompt_input_tokens),
                "digest": coerced,
                "llm": {
                    "provider": reply.provider,
                    "model": reply.model,
                "meta": reply.meta,
                "invalid_output_recovered": bool(recovered),
                "attempts": len(attempts),
            },
        },
        prompt={"messages": messages, "structured_outputs": {"json": schema}, "vllm_extra_body": vllm_extra_body},
        response={"attempts": attempts},
    )


def _run_support_digest(
    *,
    payload: dict[str, Any],
    agent_id: str,
    run_id: str,
    command_id: int | None = None,
) -> CommandExecution:
    return _run_llm_task_sync(
        _task_support_digest(payload=payload, agent_id=agent_id, run_id=run_id, command_id=command_id)
    )


_REPO_REVIEW_MAX_FILE_BYTES = 250_000
_REPO_REVIEW_ALLOW_SUFFIXES = {".py", ".ts", ".tsx", ".js", ".jsx", ".vue", ".md", ".txt"}
_REPO_REVIEW_DENY_DIRS = {".git", "node_modules", "__pycache__", ".pytest_cache", ".mypy_cache", ".venv", "venv"}


def _repo_root() -> Path | None:
    raw = (os.getenv("ISPEC_ASSISTANT_REPO_ROOT") or "").strip()
    if raw:
        candidate = Path(raw).expanduser()
        if candidate.exists():
            try:
                return candidate.resolve()
            except Exception:
                return candidate
    start = Path(__file__).resolve()
    for parent in start.parents:
        if (parent / "Makefile").exists() and (parent / "iSPEC").exists():
            return parent
    return None


def _repo_review_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "schema_version": {"type": "integer", "enum": [_REPO_REVIEW_VERSION]},
            "summary": {"type": "string", "maxLength": 2000},
            "findings": {
                "type": "array",
                "maxItems": 20,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "severity": {"type": "string", "enum": ["info", "warning", "error"]},
                        "path": {"type": "string", "maxLength": 260},
                        "line": {"type": "integer", "minimum": 1},
                        "title": {"type": "string", "maxLength": 160},
                        "recommendation": {"type": "string", "maxLength": 600},
                    },
                    "required": ["severity", "path", "title", "recommendation"],
                },
            },
            "next_steps": {
                "type": "array",
                "maxItems": 20,
                "items": {"type": "string", "maxLength": 240},
            },
        },
        "required": ["schema_version", "summary", "findings", "next_steps"],
    }


@prompt_binding("supervisor.repo_review.system")
def _repo_review_system_prompt() -> str:
    return load_bound_prompt(_repo_review_system_prompt).text

@prompt_binding("supervisor.repo_review.repair")
def _repo_review_repair_system_prompt() -> str:
    return load_bound_prompt(_repo_review_repair_system_prompt).text

def _coerce_repo_review_output(parsed: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(parsed, dict):
        return None

    if int(parsed.get("schema_version") or 0) == _REPO_REVIEW_VERSION:
        return parsed

    review_obj = parsed.get("review")
    if isinstance(review_obj, dict) and not any(key in parsed for key in ("summary", "findings", "next_steps")):
        issues_raw = review_obj.get("issues")
        recommendations_raw = review_obj.get("recommendations")
        findings_from_issues: list[dict[str, Any]] = []

        severity_map = {"low": "info", "medium": "warning", "high": "error", "info": "info", "warning": "warning", "error": "error"}

        if isinstance(issues_raw, list):
            for item in issues_raw:
                if not isinstance(item, dict):
                    continue
                path = item.get("file") if isinstance(item.get("file"), str) else item.get("path")
                if not isinstance(path, str) or not path.strip():
                    continue
                title = item.get("issue") if isinstance(item.get("issue"), str) else item.get("title")
                description = item.get("description") if isinstance(item.get("description"), str) else None
                recommendation = item.get("recommendation") if isinstance(item.get("recommendation"), str) else description
                if not isinstance(title, str) or not title.strip():
                    title = "Finding"
                if not isinstance(recommendation, str) or not recommendation.strip():
                    recommendation = "Review this location."
                severity_raw = item.get("severity")
                severity_key = str(severity_raw or "").strip().lower()
                severity = severity_map.get(severity_key, "info")
                finding: dict[str, Any] = {
                    "severity": severity,
                    "path": path.strip(),
                    "title": _truncate_text(title, limit=160),
                    "recommendation": _truncate_text(recommendation, limit=600),
                }
                line = item.get("line")
                line_int = _safe_int(line) if line is not None else None
                if line_int is not None and line_int > 0:
                    finding["line"] = int(line_int)
                findings_from_issues.append(finding)

        next_steps_from_recs: list[str] = []
        if isinstance(recommendations_raw, str) and recommendations_raw.strip():
            recommendations_raw = [recommendations_raw]
        if isinstance(recommendations_raw, list):
            for item in recommendations_raw:
                if not isinstance(item, str):
                    continue
                text = item.strip()
                if not text:
                    continue
                next_steps_from_recs.append(_truncate_text(text, limit=240))

        summary = _truncate_text(
            f"Auto-parsed repo review ({len(findings_from_issues)} findings, {len(next_steps_from_recs)} next steps).",
            limit=2000,
        )
        return {
            "schema_version": _REPO_REVIEW_VERSION,
            "summary": summary,
            "findings": findings_from_issues[:20],
            "next_steps": next_steps_from_recs[:20],
        }

    summary = parsed.get("summary") if isinstance(parsed.get("summary"), str) else ""
    findings_raw = parsed.get("findings")
    next_steps_raw = parsed.get("next_steps")

    findings: list[dict[str, Any]] = []
    if isinstance(findings_raw, list):
        for item in findings_raw:
            if not isinstance(item, dict):
                continue
            severity = item.get("severity")
            if not isinstance(severity, str) or severity not in {"info", "warning", "error"}:
                severity = "info"
            path = item.get("path")
            title = item.get("title")
            recommendation = item.get("recommendation")
            if not isinstance(path, str) or not path.strip():
                continue
            if not isinstance(title, str) or not title.strip():
                continue
            if not isinstance(recommendation, str) or not recommendation.strip():
                continue
            coerced: dict[str, Any] = {
                "severity": severity,
                "path": path.strip(),
                "title": _truncate_text(title, limit=160),
                "recommendation": _truncate_text(recommendation, limit=600),
            }
            line = item.get("line")
            line_int = _safe_int(line) if line is not None else None
            if line_int is not None and line_int > 0:
                coerced["line"] = int(line_int)
            findings.append(coerced)

    next_steps: list[str] = []
    if isinstance(next_steps_raw, str) and next_steps_raw.strip():
        next_steps_raw = [next_steps_raw]
    if isinstance(next_steps_raw, list):
        for item in next_steps_raw:
            if not isinstance(item, str):
                continue
            text = item.strip()
            if not text:
                continue
            next_steps.append(_truncate_text(text, limit=240))

    if not summary:
        summary = _truncate_text(
            f"Auto-parsed repo review ({len(findings)} findings, {len(next_steps)} next steps).",
            limit=2000,
        )

    return {
        "schema_version": _REPO_REVIEW_VERSION,
        "summary": _truncate_text(summary, limit=2000),
        "findings": findings[:20],
        "next_steps": next_steps[:20],
    }


def _repo_path_allowed(path: Path) -> bool:
    if path.suffix and path.suffix.lower() not in _REPO_REVIEW_ALLOW_SUFFIXES:
        return False
    parts = set(path.parts)
    if _REPO_REVIEW_DENY_DIRS.intersection(parts):
        return False
    return True


def _repo_rg_search(*, repo_root: Path, query: str, path: str, limit: int) -> list[dict[str, Any]]:
    cmd = ["rg", "-n", "--no-heading", "--max-columns", "300", "--ignore-case", "--fixed-strings", "--", query, path]
    try:
        completed = subprocess.run(
            cmd,
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except FileNotFoundError:
        return []
    except Exception:
        return []
    if completed.returncode not in {0, 1}:
        return []

    matches: list[dict[str, Any]] = []
    for raw_line in (completed.stdout or "").splitlines():
        line = raw_line.rstrip("\n")
        if not line:
            continue
        parts = line.split(":", 2)
        if len(parts) < 3:
            continue
        file_part, line_part, text_part = parts[0], parts[1], parts[2]
        try:
            line_no = int(line_part)
        except ValueError:
            continue
        matches.append({"path": file_part, "line": line_no, "text": text_part[:300]})
        if len(matches) >= limit:
            break
    return matches


def _repo_read_snippet(*, repo_root: Path, rel_path: str, line: int, context_lines: int = 3) -> str | None:
    candidate = (repo_root / rel_path).resolve()
    try:
        candidate.relative_to(repo_root)
    except Exception:
        return None
    if not candidate.exists() or not candidate.is_file():
        return None
    if not _repo_path_allowed(candidate):
        return None
    try:
        stat = candidate.stat()
    except Exception:
        stat = None
    if stat is not None and int(stat.st_size) > _REPO_REVIEW_MAX_FILE_BYTES:
        return None

    try:
        raw = candidate.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None
    lines = raw.splitlines()
    line = max(1, int(line))
    start = max(1, line - context_lines)
    end = min(len(lines), line + context_lines)
    snippet_lines = lines[start - 1 : end]
    return "\n".join(snippet_lines)


def _task_repo_review(*, payload: dict[str, Any], agent_id: str, run_id: str) -> LLMTask:
    provider = (os.getenv("ISPEC_ASSISTANT_PROVIDER") or "").strip().lower()
    if provider != "vllm":
        return CommandExecution(
            ok=False,
            result={"ok": False, "error": "Repo review requires ISPEC_ASSISTANT_PROVIDER=vllm."},
            error="provider_not_vllm",
        )

    repo_root = _repo_root()
    if repo_root is None:
        return CommandExecution(ok=False, result={"ok": False, "error": "Repo root not found."}, error="repo_root_not_found")

    paths_raw = payload.get("paths")
    if isinstance(paths_raw, list):
        paths = [str(p) for p in paths_raw if isinstance(p, str) and p.strip()]
    else:
        paths = []
    if not paths:
        paths = ["iSPEC/src", "mspc-data-entry-ui/app/src"]

    queries_raw = payload.get("queries")
    if isinstance(queries_raw, list):
        queries = [str(q) for q in queries_raw if isinstance(q, str) and q.strip()]
    else:
        queries = []
    if not queries:
        queries = ["TODO", "FIXME"]

    limit = _clamp_int(_safe_int(payload.get("limit")) or 12, min_value=1, max_value=40)

    matches: list[dict[str, Any]] = []
    for query in queries[:5]:
        for path in paths[:6]:
            matches.extend(_repo_rg_search(repo_root=repo_root, query=query, path=path, limit=limit))
            if len(matches) >= limit:
                break
        if len(matches) >= limit:
            break
    matches = matches[:limit]

    snippets: list[dict[str, Any]] = []
    for match in matches[:20]:
        rel = str(match.get("path") or "")
        line_no = int(match.get("line") or 1)
        snippet = _repo_read_snippet(repo_root=repo_root, rel_path=rel, line=line_no, context_lines=3)
        if snippet:
            snippets.append({"path": rel, "line": line_no, "match": match.get("text"), "snippet": snippet})

    context = {
        "schema_version": 1,
        "agent": {"agent_id": agent_id, "run_id": run_id},
        "repo": {"root": str(repo_root), "paths": paths, "queries": queries},
        "matches": snippets,
    }

    schema = _repo_review_schema()
    prompt = load_bound_prompt(_repo_review_system_prompt)
    messages = [
        {"role": "system", "content": prompt.text},
        {"role": "user", "content": json.dumps(context, ensure_ascii=False)},
    ]
    temperature = _structured_llm_temperature("REPO_REVIEW", default=0.0)
    repair_temperature = _structured_llm_repair_temperature("REPO_REVIEW", default=0.25)
    max_repairs = _structured_llm_max_repairs("REPO_REVIEW", default=1)

    vllm_extra_body = {"structured_outputs": {"json": schema}, "temperature": temperature, "max_tokens": 1200}
    reply = yield InferenceRequest(messages=messages, tools=None, vllm_extra_body=vllm_extra_body, observability_context=prompt_observability_context(prompt, extra={"task": "repo_review"}))

    attempts: list[dict[str, Any]] = []
    parsed = _parse_json_object(reply.content)
    coerced = _coerce_repo_review_output(parsed if isinstance(parsed, dict) else None)
    attempts.append(
        {
            "attempt": 1,
            "temperature": float(temperature),
            "raw": reply.content,
            "parsed": parsed,
            "coerced": coerced,
            "llm": {"provider": reply.provider, "model": reply.model, "meta": reply.meta},
        }
    )

    recovered = False
    if (not isinstance(coerced, dict) or int(coerced.get("schema_version") or 0) != _REPO_REVIEW_VERSION) and (
        max_repairs > 0
    ):
        repair_prompt = load_bound_prompt(_repo_review_repair_system_prompt)
        repair_messages = [
            {"role": "system", "content": repair_prompt.text},
            {"role": "user", "content": json.dumps({"context": context, "previous_output": reply.content}, ensure_ascii=False)},
        ]
        repair_body = {"structured_outputs": {"json": schema}, "temperature": repair_temperature, "max_tokens": 1200}
        repair_reply = yield InferenceRequest(messages=repair_messages, tools=None, vllm_extra_body=repair_body, observability_context=prompt_observability_context(repair_prompt, extra={"task": "repo_review_repair"}))
        parsed2 = _parse_json_object(repair_reply.content)
        coerced2 = _coerce_repo_review_output(parsed2 if isinstance(parsed2, dict) else None)
        attempts.append(
            {
                "attempt": 2,
                "temperature": float(repair_temperature),
                "raw": repair_reply.content,
                "parsed": parsed2,
                "coerced": coerced2,
                "llm": {"provider": repair_reply.provider, "model": repair_reply.model, "meta": repair_reply.meta},
            }
        )
        if isinstance(coerced2, dict) and int(coerced2.get("schema_version") or 0) == _REPO_REVIEW_VERSION:
            reply = repair_reply
            parsed = parsed2
            coerced = coerced2
            recovered = True

    if not isinstance(coerced, dict) or int(coerced.get("schema_version") or 0) != _REPO_REVIEW_VERSION:
        return CommandExecution(
            ok=False,
            result={
                "ok": False,
                "error": "Invalid repo review output.",
                "repo_root": str(repo_root),
                "paths": paths,
                "queries": queries,
                "matches": snippets,
                "raw": _truncate_text(reply.content, limit=1200),
                "attempts": len(attempts),
                "llm": {"provider": reply.provider, "model": reply.model, "meta": reply.meta},
            },
            error="invalid_repo_review_output",
            prompt={"messages": messages, "structured_outputs": {"json": schema}, "vllm_extra_body": vllm_extra_body},
            response={"attempts": attempts},
        )

    return CommandExecution(
        ok=True,
        result={
            "ok": True,
            "repo_root": str(repo_root),
            "paths": paths,
            "queries": queries,
            "matches": snippets,
            "review": coerced,
            "llm": {
                "provider": reply.provider,
                "model": reply.model,
                "meta": reply.meta,
                "invalid_output_recovered": bool(recovered),
                "attempts": len(attempts),
            },
        },
        prompt={"messages": messages, "structured_outputs": {"json": schema}, "vllm_extra_body": vllm_extra_body},
        response={"attempts": attempts},
    )


def _run_repo_review(*, payload: dict[str, Any], agent_id: str, run_id: str) -> CommandExecution:
    return _run_llm_task_sync(_task_repo_review(payload=payload, agent_id=agent_id, run_id=run_id))


def _prune_json_for_prompt(
    value: Any,
    *,
    max_depth: int = 5,
    max_list_items: int = 80,
    max_dict_items: int = 120,
    max_str_chars: int = 3000,
) -> Any:
    """Reduce large payloads so we can safely include them in LLM prompts.

    This is intentionally conservative: we keep structure but trim very large
    strings and collections.
    """

    if max_depth <= 0:
        if value is None or isinstance(value, (int, float, bool)):
            return value
        if isinstance(value, str):
            return _truncate_text(value, limit=max_str_chars)
        return "<truncated>"

    if value is None or isinstance(value, (int, float, bool)):
        return value

    if isinstance(value, str):
        return _truncate_text(value, limit=max_str_chars)

    if isinstance(value, list):
        items = [_prune_json_for_prompt(item, max_depth=max_depth - 1, max_list_items=max_list_items, max_dict_items=max_dict_items, max_str_chars=max_str_chars) for item in value[: max(0, int(max_list_items))]]
        if len(value) > int(max_list_items):
            items.append(f"<truncated {len(value) - int(max_list_items)} more items>")
        return items

    if isinstance(value, dict):
        pruned: dict[str, Any] = {}
        keys = [k for k in value.keys() if isinstance(k, str)]
        for key in keys[: max(0, int(max_dict_items))]:
            pruned[key] = _prune_json_for_prompt(
                value.get(key),
                max_depth=max_depth - 1,
                max_list_items=max_list_items,
                max_dict_items=max_dict_items,
                max_str_chars=max_str_chars,
            )
        if len(keys) > int(max_dict_items):
            pruned["_truncated_keys"] = int(len(keys) - int(max_dict_items))
        return pruned

    # Fallback: represent unknown objects by type name.
    return str(value)


def _tackle_assess_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "schema_version": {"type": "integer", "enum": [_TACKLE_ASSESS_VERSION]},
            "project_id": {"type": ["integer", "null"], "minimum": 1},
            "summary": {"type": "string", "maxLength": 2000},
            "findings": {
                "type": "array",
                "maxItems": 25,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "severity": {"type": "string", "enum": ["info", "warning", "error"]},
                        "topic": {"type": "string", "enum": ["pca", "limma", "telemetry", "io", "other"]},
                        "description": {"type": "string", "maxLength": 500},
                        "evidence": {"type": ["string", "null"], "maxLength": 500},
                    },
                    "required": ["severity", "topic", "description", "evidence"],
                },
            },
            "next_steps": {
                "type": "array",
                "maxItems": 25,
                "items": {"type": "string", "maxLength": 280},
            },
            "questions": {
                "type": "array",
                "maxItems": 25,
                "items": {"type": "string", "maxLength": 280},
            },
        },
        "required": ["schema_version", "project_id", "summary", "findings", "next_steps", "questions"],
    }


@prompt_binding("supervisor.tackle_assess.system")
def _tackle_assess_system_prompt() -> str:
    return load_bound_prompt(_tackle_assess_system_prompt).text

@prompt_binding("supervisor.tackle_assess.repair")
def _tackle_assess_repair_system_prompt() -> str:
    return load_bound_prompt(_tackle_assess_repair_system_prompt).text

def _coerce_tackle_assess_output(parsed: dict[str, Any] | None, *, project_id: int | None) -> dict[str, Any] | None:
    if not isinstance(parsed, dict):
        return None

    assess: dict[str, Any] | None = None
    for key in ("assessment", "report", "result"):
        candidate = parsed.get(key)
        if isinstance(candidate, dict):
            assess = candidate
            break
    if assess is None:
        assess = parsed

    if not isinstance(assess, dict):
        return None

    expected_keys = {"summary", "findings", "next_steps", "questions", "project_id"}
    if not any(key in assess for key in expected_keys):
        return None

    summary = assess.get("summary") if isinstance(assess.get("summary"), str) else ""
    summary = _truncate_text(summary, limit=2000)
    if not summary:
        summary = _truncate_text("Assessment generated.", limit=2000)

    parsed_project = _safe_int(assess.get("project_id"))
    if parsed_project is not None and parsed_project <= 0:
        parsed_project = None
    effective_project = parsed_project if parsed_project is not None else project_id

    findings_raw = assess.get("findings")
    if isinstance(findings_raw, dict):
        findings_raw = [findings_raw]
    findings: list[dict[str, Any]] = []
    if isinstance(findings_raw, list):
        for item in findings_raw:
            if not isinstance(item, dict):
                continue
            severity = item.get("severity")
            if not isinstance(severity, str) or severity not in {"info", "warning", "error"}:
                severity = "info"
            topic = item.get("topic")
            if not isinstance(topic, str) or topic not in {"pca", "limma", "telemetry", "io", "other"}:
                topic = "other"
            description = item.get("description")
            if not isinstance(description, str) or not description.strip():
                continue
            evidence = item.get("evidence")
            if evidence is not None and not isinstance(evidence, str):
                evidence = None
            findings.append(
                {
                    "severity": severity,
                    "topic": topic,
                    "description": _truncate_text(description.strip(), limit=500),
                    "evidence": _truncate_text(evidence.strip(), limit=500) if isinstance(evidence, str) and evidence.strip() else None,
                }
            )

    next_steps_raw = assess.get("next_steps")
    if isinstance(next_steps_raw, str) and next_steps_raw.strip():
        next_steps_raw = [next_steps_raw]
    next_steps: list[str] = []
    if isinstance(next_steps_raw, list):
        for item in next_steps_raw:
            if not isinstance(item, str):
                continue
            text = item.strip()
            if not text:
                continue
            next_steps.append(_truncate_text(text, limit=280))

    questions_raw = assess.get("questions")
    if isinstance(questions_raw, str) and questions_raw.strip():
        questions_raw = [questions_raw]
    questions: list[str] = []
    if isinstance(questions_raw, list):
        for item in questions_raw:
            if not isinstance(item, str):
                continue
            text = item.strip()
            if not text:
                continue
            questions.append(_truncate_text(text, limit=280))

    return {
        "schema_version": _TACKLE_ASSESS_VERSION,
        "project_id": int(effective_project) if isinstance(effective_project, int) and effective_project > 0 else None,
        "summary": summary,
        "findings": findings[:25],
        "next_steps": next_steps[:25],
        "questions": questions[:25],
    }


def _task_tackle_results_assess(
    *,
    payload: dict[str, Any],
    agent_id: str,
    run_id: str,
    command_id: int,
) -> LLMTask:
    project_id = _safe_int(payload.get("project_id"))
    if project_id is not None and project_id <= 0:
        project_id = None

    pruned_payload = _prune_json_for_prompt(payload)
    context = {
        "schema_version": 1,
        "agent": {"agent_id": agent_id, "run_id": run_id, "command_id": int(command_id)},
        "project_id": int(project_id) if project_id is not None else None,
        "payload": pruned_payload,
    }

    schema = _tackle_assess_schema()
    prompt = load_bound_prompt(_tackle_assess_system_prompt)
    messages = [
        {"role": "system", "content": prompt.text},
        {"role": "user", "content": json.dumps(context, ensure_ascii=False)},
    ]
    temperature = _structured_llm_temperature("TACKLE_ASSESS", default=0.0)
    repair_temperature = _structured_llm_repair_temperature("TACKLE_ASSESS", default=0.25)
    max_repairs = _structured_llm_max_repairs("TACKLE_ASSESS", default=1)

    vllm_extra_body = {"structured_outputs": {"json": schema}, "temperature": temperature, "max_tokens": 1200}
    reply = yield InferenceRequest(messages=messages, tools=None, vllm_extra_body=vllm_extra_body, observability_context=prompt_observability_context(prompt, extra={"task": "tackle_assess"}))

    attempts: list[dict[str, Any]] = []
    parsed = _parse_json_object(reply.content)
    coerced = _coerce_tackle_assess_output(parsed if isinstance(parsed, dict) else None, project_id=project_id)
    attempts.append(
        {
            "attempt": 1,
            "temperature": float(temperature),
            "raw": reply.content,
            "parsed": parsed,
            "coerced": coerced,
            "llm": {"provider": reply.provider, "model": reply.model, "meta": reply.meta},
        }
    )

    recovered = False
    if (not isinstance(coerced, dict) or int(coerced.get("schema_version") or 0) != _TACKLE_ASSESS_VERSION) and (
        max_repairs > 0
    ):
        repair_prompt = load_bound_prompt(_tackle_assess_repair_system_prompt)
        repair_messages = [
            {"role": "system", "content": repair_prompt.text},
            {"role": "user", "content": json.dumps({"context": context, "previous_output": reply.content}, ensure_ascii=False)},
        ]
        repair_body = {"structured_outputs": {"json": schema}, "temperature": repair_temperature, "max_tokens": 1200}
        repair_reply = yield InferenceRequest(messages=repair_messages, tools=None, vllm_extra_body=repair_body, observability_context=prompt_observability_context(repair_prompt, extra={"task": "tackle_assess_repair"}))
        parsed2 = _parse_json_object(repair_reply.content)
        coerced2 = _coerce_tackle_assess_output(parsed2 if isinstance(parsed2, dict) else None, project_id=project_id)
        attempts.append(
            {
                "attempt": 2,
                "temperature": float(repair_temperature),
                "raw": repair_reply.content,
                "parsed": parsed2,
                "coerced": coerced2,
                "llm": {"provider": repair_reply.provider, "model": repair_reply.model, "meta": repair_reply.meta},
            }
        )
        if isinstance(coerced2, dict) and int(coerced2.get("schema_version") or 0) == _TACKLE_ASSESS_VERSION:
            reply = repair_reply
            parsed = parsed2
            coerced = coerced2
            recovered = True

    if not isinstance(coerced, dict) or int(coerced.get("schema_version") or 0) != _TACKLE_ASSESS_VERSION:
        return CommandExecution(
            ok=False,
            result={
                "ok": False,
                "error": "Invalid tackle assessment output.",
                "project_id": int(project_id) if project_id is not None else None,
                "raw": _truncate_text(reply.content, limit=1200),
                "attempts": len(attempts),
                "llm": {"provider": reply.provider, "model": reply.model, "meta": reply.meta},
            },
            error="invalid_tackle_assess_output",
            prompt={"messages": messages, "structured_outputs": {"json": schema}, "vllm_extra_body": vllm_extra_body},
            response={"attempts": attempts},
        )

    return CommandExecution(
        ok=True,
        result={
            "ok": True,
            "project_id": int(project_id) if project_id is not None else None,
            "assessment": coerced,
            "llm": {
                "provider": reply.provider,
                "model": reply.model,
                "meta": reply.meta,
                "invalid_output_recovered": bool(recovered),
                "attempts": len(attempts),
            },
        },
        prompt={"messages": messages, "structured_outputs": {"json": schema}, "vllm_extra_body": vllm_extra_body},
        response={"attempts": attempts},
    )


def _run_tackle_results_assess(
    *,
    payload: dict[str, Any],
    agent_id: str,
    run_id: str,
    command_id: int,
) -> CommandExecution:
    return _run_llm_task_sync(
        _task_tackle_results_assess(payload=payload, agent_id=agent_id, run_id=run_id, command_id=command_id)
    )


@prompt_binding("supervisor.tackle_prompt.system")
def _tackle_prompt_system_prompt() -> str:
    return load_bound_prompt(_tackle_prompt_system_prompt).text

def _task_tackle_prompt_freeform(
    *,
    payload: dict[str, Any],
    agent_id: str,
    run_id: str,
    command_id: int,
) -> LLMTask:
    prompt_text = payload.get("prompt")
    if not isinstance(prompt_text, str) or not prompt_text.strip():
        prompt_text = payload.get("prompt_text")
    if not isinstance(prompt_text, str) or not prompt_text.strip():
        prompt_text = payload.get("message")
    prompt = (prompt_text or "").strip()
    if not prompt:
        return CommandExecution(
            ok=False,
            result={"ok": False, "error": "Missing prompt text (payload.prompt)."},
            error="missing_prompt",
        )

    project_id = _safe_int(payload.get("project_id"))
    if project_id is not None and project_id <= 0:
        project_id = None

    # Optional structured context that tackle can send alongside the prompt.
    extra_context = payload.get("context")
    if not isinstance(extra_context, dict):
        extra_context = {}
    pruned_context = _prune_json_for_prompt(extra_context)

    user_parts: list[str] = []
    if project_id is not None:
        user_parts.append(f"project_id: {int(project_id)}")
    if pruned_context:
        user_parts.append("context_json:\n" + json.dumps(pruned_context, ensure_ascii=False))
    user_parts.append(prompt)
    user_content = "\n\n".join(user_parts).strip()

    prompt = load_bound_prompt(_tackle_prompt_system_prompt)
    messages = [
        {"role": "system", "content": prompt.text},
        {"role": "user", "content": user_content},
    ]

    # Keep output bounded unless the caller explicitly asks for a larger cap.
    max_tokens = _safe_int(payload.get("max_tokens"))
    if max_tokens is None:
        max_tokens = 900
    max_tokens = _clamp_int(int(max_tokens), min_value=64, max_value=2400)

    vllm_extra_body = {"max_tokens": int(max_tokens)}
    reply = yield InferenceRequest(messages=messages, tools=None, vllm_extra_body=vllm_extra_body, observability_context=prompt_observability_context(prompt, extra={"task": "tackle_prompt_freeform"}))
    if not reply.ok:
        return CommandExecution(
            ok=False,
            result={"ok": False, "error": reply.error or "vLLM request failed.", "llm": {"meta": reply.meta}},
            error="tackle_prompt_llm_failed",
            prompt={"messages": messages, "vllm_extra_body": vllm_extra_body},
            response={"raw": reply.content, "llm": {"provider": reply.provider, "model": reply.model, "meta": reply.meta}},
        )

    output_text = _truncate_text(reply.content, limit=20_000)
    return CommandExecution(
        ok=True,
        result={
            "ok": True,
            "project_id": int(project_id) if project_id is not None else None,
            "output_text": output_text,
            "llm": {"provider": reply.provider, "model": reply.model, "meta": reply.meta},
        },
        prompt={"messages": messages, "vllm_extra_body": vllm_extra_body},
        response={"raw": output_text},
    )


def _run_tackle_prompt_freeform(
    *,
    payload: dict[str, Any],
    agent_id: str,
    run_id: str,
    command_id: int,
) -> CommandExecution:
    return _run_llm_task_sync(
        _task_tackle_prompt_freeform(payload=payload, agent_id=agent_id, run_id=run_id, command_id=command_id)
    )


def _looks_like_slack_channel_id(value: str) -> bool:
    raw = (value or "").strip()
    if not raw:
        return False
    return raw[0] in {"C", "G"} and len(raw) >= 6


def _slack_bot_token() -> str:
    return (os.getenv("ISPEC_SLACK_BOT_TOKEN") or os.getenv("SLACK_BOT_TOKEN") or "").strip()


def _slack_timeout_seconds() -> float:
    raw = (os.getenv("ISPEC_SLACK_TIMEOUT_SECONDS") or "").strip()
    if not raw:
        return 10.0
    try:
        return max(1.0, float(raw))
    except ValueError:
        return 10.0


def _slack_api_call(
    *,
    token: str,
    endpoint: str,
    payload: dict[str, Any],
    timeout_seconds: float,
) -> dict[str, Any]:
    resp = requests.post(
        f"https://slack.com/api/{endpoint.lstrip('/')}",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json; charset=utf-8",
        },
        json=payload,
        timeout=max(1.0, float(timeout_seconds)),
    )
    resp.raise_for_status()
    try:
        data = resp.json()
    except Exception:
        data = {}
    return data if isinstance(data, dict) else {}


def _run_slack_post_message(payload: dict[str, Any]) -> CommandExecution:
    token = _slack_bot_token()
    if not token:
        return CommandExecution(
            ok=False,
            result={"ok": False, "error": "Missing ISPEC_SLACK_BOT_TOKEN/SLACK_BOT_TOKEN."},
            error="missing_slack_token",
        )

    channel = str(payload.get("channel") or "").strip()
    text = str(payload.get("text") or "").strip()
    thread_ts = str(payload.get("thread_ts") or "").strip() or None
    if not channel:
        return CommandExecution(
            ok=False,
            result={"ok": False, "error": "Missing channel."},
            error="missing_channel",
        )
    if not text:
        return CommandExecution(
            ok=False,
            result={"ok": False, "error": "Missing text."},
            error="missing_text",
        )

    timeout_seconds = _slack_timeout_seconds()
    message_payload: dict[str, Any] = {"channel": channel, "text": text}
    if thread_ts:
        message_payload["thread_ts"] = thread_ts

    slack_response = _slack_api_call(
        token=token,
        endpoint="chat.postMessage",
        payload=message_payload,
        timeout_seconds=timeout_seconds,
    )
    if slack_response.get("ok") is not True:
        error = str(slack_response.get("error") or "").strip() or "unknown_error"
        if error == "not_in_channel" and _looks_like_slack_channel_id(channel):
            join_response = _slack_api_call(
                token=token,
                endpoint="conversations.join",
                payload={"channel": channel},
                timeout_seconds=timeout_seconds,
            )
            if join_response.get("ok") is True:
                slack_response = _slack_api_call(
                    token=token,
                    endpoint="chat.postMessage",
                    payload=message_payload,
                    timeout_seconds=timeout_seconds,
                )
            if slack_response.get("ok") is not True:
                error = str(slack_response.get("error") or "").strip() or error

        return CommandExecution(
            ok=False,
            result={
                "ok": False,
                "error": error,
                "channel": channel,
                "thread_ts": thread_ts,
                "slack": slack_response,
            },
            error=f"slack_error:{error}",
        )

    return CommandExecution(
        ok=True,
        result={
            "ok": True,
            "channel": channel,
            "thread_ts": thread_ts,
            "slack": slack_response,
        },
    )


def _run_dev_restart_services(payload: dict[str, Any]) -> CommandExecution:
    """Best-effort tmux/make restarts for local dev services.

    This is intentionally gated because it is only valid in a tmux + top-level
    Makefile workflow.
    """

    if not isinstance(payload, dict):
        payload = {}

    if payload.get("confirm") is not True:
        return CommandExecution(
            ok=False,
            result={"ok": False, "error": "confirm=true is required."},
            error="confirm_required",
        )

    services_raw = payload.get("services")
    services: list[str] = []
    if isinstance(services_raw, list):
        services = [str(item).strip().lower() for item in services_raw if isinstance(item, str) and item.strip()]
    elif isinstance(services_raw, str):
        services = [tok.strip().lower() for tok in services_raw.replace(",", " ").split() if tok.strip()]

    if not services:
        services = ["backend", "supervisor"]

    normalized: list[str] = []
    seen: set[str] = set()
    for svc in services:
        if svc in {"api"}:
            svc = "backend"
        if svc not in seen:
            normalized.append(svc)
            seen.add(svc)

    allowed = {"backend", "supervisor", "frontend", "vllm", "slack"}
    unknown = sorted({svc for svc in normalized if svc not in allowed})
    if unknown:
        return CommandExecution(
            ok=False,
            result={
                "ok": False,
                "error": "Unknown services requested.",
                "unknown": unknown,
                "allowed": sorted(allowed),
            },
            error="unknown_services",
        )

    tmux_session = str(payload.get("tmux_session") or os.getenv("DEV_TMUX_SESSION") or "ispecfull").strip() or "ispecfull"
    make_root = str(payload.get("make_root") or "").strip() or None

    enabled, reason = _dev_restart_enabled_status(tmux_session=tmux_session, make_root=make_root)
    if not enabled:
        return CommandExecution(
            ok=False,
            result={
                "ok": False,
                "error": "Dev restart is unavailable.",
                "hint": reason,
            },
            error="dev_restart_disabled",
        )

    restart_supervisor = "supervisor" in normalized
    immediate_services = [svc for svc in normalized if svc != "supervisor"]

    result: dict[str, Any] = {
        "ok": True,
        "services": normalized,
        "tmux_session": tmux_session,
        "make_root": make_root,
        "restarted": [],
        "scheduled": [],
    }

    try:
        from types import SimpleNamespace
        from ispec.cli import dev as dev_cli
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        result["ok"] = False
        result["error"] = error
        return CommandExecution(ok=False, result=result, error="dev_restart_import_failed")

    if immediate_services:
        try:
            args = SimpleNamespace(
                subcommand="restart",
                services=immediate_services,
                tmux_session=tmux_session,
                make_root=make_root,
            )
            dev_cli.dispatch(args)
            result["restarted"] = list(immediate_services)
        except SystemExit as exc:
            code = getattr(exc, "code", None)
            error = str(code) if code is not None else str(exc)
            result["ok"] = False
            result["error"] = error or "dev_restart_failed"
            return CommandExecution(ok=False, result=result, error="dev_restart_failed")
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            result["ok"] = False
            result["error"] = error
            return CommandExecution(ok=False, result=result, error="dev_restart_failed")

    if restart_supervisor:
        try:
            # Restarting the supervisor from inside itself risks killing this
            # process before it can mark the agent_command as complete. Schedule
            # the tmux respawn with a small delay so the command finish can be
            # committed first.
            make_root_path = Path(make_root).expanduser().resolve() if make_root else dev_cli._find_make_root()  # type: ignore[attr-defined]
            if make_root_path is None:
                raise RuntimeError("Unable to locate the top-level Makefile; pass make_root.")

            window = getattr(dev_cli, "_SERVICE_TO_WINDOW", {}).get("supervisor")
            make_target = getattr(dev_cli, "_SERVICE_TO_MAKE_TARGET", {}).get("supervisor")
            if not isinstance(window, str) or not window.strip():
                raise RuntimeError("Dev mapping missing for supervisor window.")
            if not isinstance(make_target, str) or not make_target.strip():
                raise RuntimeError("Dev mapping missing for supervisor make target.")

            target_window = f"{tmux_session}:{window}"
            pane_id = dev_cli._tmux_first_pane_id(target_window)  # type: ignore[attr-defined]
            if not pane_id:
                raise RuntimeError(f"Unable to find a pane for tmux window {target_window!r}.")

            cmd = f'cd "{make_root_path.as_posix()}" && make {make_target}'
            import shlex

            delay = 0.8
            script = (
                f"sleep {delay}; tmux respawn-pane -k -t {shlex.quote(str(pane_id))} {shlex.quote(cmd)}"
            )
            subprocess.Popen(
                ["bash", "-lc", script],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            result["scheduled"].append({"service": "supervisor", "delay_seconds": delay, "pane_id": pane_id})
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            result["ok"] = False
            result["error"] = error
            return CommandExecution(ok=False, result=result, error="dev_restart_supervisor_failed")

    return CommandExecution(ok=True, result=result)


@dataclass(frozen=True)
class SlackSchedule:
    name: str
    weekday: int
    hour: int
    minute: int
    timezone: str
    channel: str
    text: str
    grace_seconds: int = 0
    priority: int = 0
    max_attempts: int = 1
    enabled: bool = True


@dataclass(frozen=True)
class _ScheduledAssistantToolUser:
    id: int = -1
    username: str = "scheduled_assistant"
    role: UserRole = UserRole.admin
    is_active: bool = True
    must_change_password: bool = False
    assistant_brief: str | None = None
    can_write_project_comments: bool = True


def _load_slack_schedule_json() -> list[dict[str, Any]]:
    path = (os.getenv("ISPEC_SLACK_SCHEDULE_PATH") or "").strip()
    raw = ""
    if path:
        try:
            raw = Path(path).expanduser().read_text(encoding="utf-8")
        except Exception as exc:
            logger.warning("Failed reading ISPEC_SLACK_SCHEDULE_PATH=%s (%s)", path, exc)
            return []
    else:
        raw = (os.getenv("ISPEC_SLACK_SCHEDULE_JSON") or "").strip()

    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except Exception as exc:
        logger.warning("Invalid ISPEC_SLACK_SCHEDULE_JSON (%s)", exc)
        return []
    if isinstance(parsed, dict):
        parsed = [parsed]
    if not isinstance(parsed, list):
        return []
    return [row for row in parsed if isinstance(row, dict)]


def _load_slack_schedules() -> list[SlackSchedule]:
    default_timezone = (os.getenv("ISPEC_SLACK_DEFAULT_TIMEZONE") or "").strip() or "UTC"
    schedules: list[SlackSchedule] = []
    seen: set[str] = set()
    for row in _load_slack_schedule_json():
        name = str(row.get("name") or "").strip()
        if not name or name in seen:
            continue
        weekday = _parse_weekday(row.get("weekday"))
        hhmm = _parse_hhmm(row.get("time"))
        channel = str(row.get("channel") or "").strip()
        text = str(row.get("text") or "").strip()
        if weekday is None or hhmm is None or not channel or not text:
            continue
        hour, minute = hhmm
        timezone = str(row.get("timezone") or "").strip() or default_timezone
        grace_seconds = _safe_int(row.get("grace_seconds")) or 0
        grace_seconds = _clamp_int(grace_seconds, min_value=0, max_value=3600)
        priority = _safe_int(row.get("priority")) or 0
        priority = _clamp_int(priority, min_value=-10, max_value=10)
        max_attempts = _safe_int(row.get("max_attempts")) or 1
        max_attempts = _clamp_int(max_attempts, min_value=1, max_value=10)
        enabled = row.get("enabled")
        if isinstance(enabled, str):
            enabled = _is_truthy(enabled)
        enabled = bool(enabled) if enabled is not None else True
        schedules.append(
            SlackSchedule(
                name=name,
                weekday=int(weekday),
                hour=int(hour),
                minute=int(minute),
                timezone=timezone,
                channel=channel,
                text=text,
                grace_seconds=int(grace_seconds),
                priority=int(priority),
                max_attempts=int(max_attempts),
                enabled=enabled,
            )
        )
        seen.add(name)
    return schedules
def _slack_schedule_next_occurrence(
    *,
    now: datetime,
    schedule: SlackSchedule,
) -> tuple[datetime, datetime, str]:
    tz_name = schedule.timezone or "UTC"
    try:
        tz = ZoneInfo(tz_name)
    except Exception:
        tz = UTC
        tz_name = "UTC"

    now_local = now.astimezone(tz)
    days_ahead = (int(schedule.weekday) - int(now_local.weekday())) % 7
    candidate_local = (now_local + timedelta(days=days_ahead)).replace(
        hour=int(schedule.hour),
        minute=int(schedule.minute),
        second=0,
        microsecond=0,
    )

    available_at = candidate_local
    if days_ahead == 0 and candidate_local <= now_local:
        if schedule.grace_seconds > 0 and now_local <= candidate_local + timedelta(seconds=int(schedule.grace_seconds)):
            available_at = now
        else:
            candidate_local = candidate_local + timedelta(days=7)
            available_at = candidate_local

    occurrence_utc = candidate_local.astimezone(UTC)
    available_at_utc = available_at.astimezone(UTC)
    key = f"{schedule.name}:{occurrence_utc.isoformat()}"
    return occurrence_utc, available_at_utc, key


def _ensure_slack_scheduled_commands(*, agent_id: str, run_id: str) -> dict[str, Any]:
    schedules = [s for s in _load_slack_schedules() if s.enabled]
    if not schedules:
        return {"ok": True, "scheduled": 0, "skipped": 0}

    now = utcnow()
    scheduled_count = 0
    skipped = 0
    with get_agent_session() as db:
        run = db.query(AgentRun).filter(AgentRun.run_id == run_id).one()
        scheduler_state = _load_scheduler_state(run)
        slack_state = scheduler_state.get("slack")
        if not isinstance(slack_state, dict):
            slack_state = {}

        existing_keys: set[str] = set()
        existing_rows = (
            db.query(AgentCommand)
            .filter(AgentCommand.command_type == COMMAND_SLACK_POST_MESSAGE)
            .filter(AgentCommand.status.in_(["queued", "running"]))
            .all()
        )
        for row in existing_rows:
            payload = row.payload_json if isinstance(row.payload_json, dict) else {}
            schedule_payload = payload.get("schedule") if isinstance(payload, dict) else None
            if not isinstance(schedule_payload, dict):
                continue
            key = str(schedule_payload.get("key") or "").strip()
            if key:
                existing_keys.add(key)

        dirty = False
        for schedule in schedules:
            occurrence_utc, available_at_utc, key = _slack_schedule_next_occurrence(now=now, schedule=schedule)
            existing = slack_state.get(schedule.name)
            schedule_state = existing if isinstance(existing, dict) else {}
            if str(schedule_state.get("last_attempted_key") or "").strip() == key:
                skipped += 1
                continue
            if key in existing_keys:
                skipped += 1
                continue

            cmd = AgentCommand(
                command_type=COMMAND_SLACK_POST_MESSAGE,
                status="queued",
                priority=int(schedule.priority),
                created_at=now,
                updated_at=now,
                available_at=available_at_utc,
                attempts=0,
                max_attempts=int(schedule.max_attempts),
                payload_json={
                    "channel": schedule.channel,
                    "text": schedule.text,
                    "schedule": {
                        "name": schedule.name,
                        "key": key,
                        "occurrence_utc": occurrence_utc.isoformat(),
                        "timezone": schedule.timezone,
                        "weekday": schedule.weekday,
                        "time": f"{schedule.hour:02d}:{schedule.minute:02d}",
                    },
                    "meta": {"enqueued_by": "supervisor", "agent_id": agent_id, "run_id": run_id},
                },
                result_json={},
            )
            db.add(cmd)
            db.flush()
            existing_keys.add(key)
            scheduled_count += 1
            dirty = True

            slack_state[schedule.name] = {
                **schedule_state,
                "next_enqueued_key": key,
                "next_enqueued_at": now.isoformat(),
                "next_command_id": int(cmd.id),
                "next_occurrence_utc": occurrence_utc.isoformat(),
                "next_available_at_utc": available_at_utc.isoformat(),
            }

        if dirty:
            scheduler_state["slack"] = slack_state
            _save_scheduler_state(run=run, state=scheduler_state)
            db.commit()

    return {"ok": True, "scheduled": scheduled_count, "skipped": skipped}


def _record_slack_schedule_attempt(
    *,
    run: AgentRun,
    payload: dict[str, Any],
    execution: CommandExecution,
    ended_at: datetime,
) -> None:
    schedule = payload.get("schedule")
    if not isinstance(schedule, dict):
        return
    name = str(schedule.get("name") or "").strip()
    key = str(schedule.get("key") or "").strip()
    if not name or not key:
        return

    scheduler_state = _load_scheduler_state(run)
    slack_state = scheduler_state.get("slack")
    if not isinstance(slack_state, dict):
        slack_state = {}

    existing = slack_state.get(name)
    schedule_state = existing if isinstance(existing, dict) else {}
    schedule_state = dict(schedule_state)
    schedule_state["last_attempted_key"] = key
    schedule_state["last_attempted_at"] = ended_at.isoformat()
    schedule_state["last_attempted_ok"] = bool(execution.ok)
    if execution.ok:
        schedule_state["last_sent_key"] = key
        schedule_state["last_sent_at"] = ended_at.isoformat()
    else:
        schedule_state["last_attempted_error"] = execution.error

    next_key = str(schedule_state.get("next_enqueued_key") or "").strip()
    if next_key == key:
        schedule_state.pop("next_enqueued_key", None)
        schedule_state.pop("next_enqueued_at", None)
        schedule_state.pop("next_command_id", None)
        schedule_state.pop("next_occurrence_utc", None)
        schedule_state.pop("next_available_at_utc", None)

    slack_state[name] = schedule_state
    scheduler_state["slack"] = slack_state
    _save_scheduler_state(run=run, state=scheduler_state)


def _ensure_assistant_scheduled_commands(*, agent_id: str, run_id: str) -> dict[str, Any]:
    schedules = [s for s in _load_assistant_schedules() if s.enabled]
    if not schedules:
        return {"ok": True, "scheduled": 0, "skipped": 0}

    now = utcnow()
    scheduled_count = 0
    skipped = 0
    with get_agent_session() as db:
        run = db.query(AgentRun).filter(AgentRun.run_id == run_id).one()
        scheduler_state = _load_scheduler_state(run)
        assistant_state = scheduler_state.get("assistant_jobs")
        if not isinstance(assistant_state, dict):
            assistant_state = {}

        existing_keys: set[str] = set()
        existing_rows = (
            db.query(AgentCommand)
            .filter(AgentCommand.command_type == COMMAND_RUN_SCHEDULED_ASSISTANT_PROMPT)
            .filter(AgentCommand.status.in_(["queued", "running"]))
            .all()
        )
        for row in existing_rows:
            payload = row.payload_json if isinstance(row.payload_json, dict) else {}
            schedule_payload = payload.get("schedule") if isinstance(payload, dict) else None
            if not isinstance(schedule_payload, dict):
                continue
            key = str(schedule_payload.get("key") or "").strip()
            if key:
                existing_keys.add(key)

        dirty = False
        for schedule in schedules:
            occurrence_utc, available_at_utc, key = _slack_schedule_next_occurrence(now=now, schedule=schedule)
            existing = assistant_state.get(schedule.name)
            schedule_state = existing if isinstance(existing, dict) else {}
            if str(schedule_state.get("last_attempted_key") or "").strip() == key:
                skipped += 1
                continue
            if key in existing_keys:
                skipped += 1
                continue

            cmd = AgentCommand(
                command_type=COMMAND_RUN_SCHEDULED_ASSISTANT_PROMPT,
                status="queued",
                priority=int(schedule.priority),
                created_at=now,
                updated_at=now,
                available_at=available_at_utc,
                attempts=0,
                max_attempts=int(schedule.max_attempts),
                payload_json={
                    "job": {
                        "name": schedule.name,
                        "prompt": schedule.prompt,
                        "allowed_tools": list(schedule.allowed_tools),
                        "required_tool": schedule.required_tool,
                        "max_tool_calls": int(schedule.max_tool_calls),
                    },
                    "schedule": {
                        "name": schedule.name,
                        "key": key,
                        "occurrence_utc": occurrence_utc.isoformat(),
                        "timezone": schedule.timezone,
                        "weekday": schedule.weekday,
                        "time": f"{schedule.hour:02d}:{schedule.minute:02d}",
                    },
                    "meta": {"enqueued_by": "supervisor", "agent_id": agent_id, "run_id": run_id},
                },
                result_json={},
            )
            db.add(cmd)
            db.flush()
            existing_keys.add(key)
            scheduled_count += 1
            dirty = True

            assistant_state[schedule.name] = {
                **schedule_state,
                "next_enqueued_key": key,
                "next_enqueued_at": now.isoformat(),
                "next_command_id": int(cmd.id),
                "next_occurrence_utc": occurrence_utc.isoformat(),
                "next_available_at_utc": available_at_utc.isoformat(),
            }

        if dirty:
            scheduler_state["assistant_jobs"] = assistant_state
            _save_scheduler_state(run=run, state=scheduler_state)
            db.commit()

    return {"ok": True, "scheduled": scheduled_count, "skipped": skipped}


def _record_assistant_schedule_attempt(
    *,
    run: AgentRun,
    payload: dict[str, Any],
    execution: CommandExecution,
    ended_at: datetime,
) -> None:
    schedule = payload.get("schedule")
    if not isinstance(schedule, dict):
        return
    name = str(schedule.get("name") or "").strip()
    key = str(schedule.get("key") or "").strip()
    if not name or not key:
        return

    scheduler_state = _load_scheduler_state(run)
    assistant_state = scheduler_state.get("assistant_jobs")
    if not isinstance(assistant_state, dict):
        assistant_state = {}

    existing = assistant_state.get(name)
    schedule_state = existing if isinstance(existing, dict) else {}
    schedule_state = dict(schedule_state)
    schedule_state["last_attempted_key"] = key
    schedule_state["last_attempted_at"] = ended_at.isoformat()
    schedule_state["last_attempted_ok"] = bool(execution.ok)
    if execution.ok:
        schedule_state["last_completed_key"] = key
        schedule_state["last_completed_at"] = ended_at.isoformat()
        result_obj = execution.result if isinstance(execution.result, dict) else {}
        queued_command_ids = result_obj.get("queued_command_ids")
        if isinstance(queued_command_ids, list):
            schedule_state["last_queued_command_ids"] = [
                int(item) for item in queued_command_ids if isinstance(item, int) and int(item) > 0
            ]
    else:
        schedule_state["last_attempted_error"] = execution.error

    next_key = str(schedule_state.get("next_enqueued_key") or "").strip()
    if next_key == key:
        schedule_state.pop("next_enqueued_key", None)
        schedule_state.pop("next_enqueued_at", None)
        schedule_state.pop("next_command_id", None)
        schedule_state.pop("next_occurrence_utc", None)
        schedule_state.pop("next_available_at_utc", None)

    assistant_state[name] = schedule_state
    scheduler_state["assistant_jobs"] = assistant_state
    _save_scheduler_state(run=run, state=scheduler_state)


def _parse_iso_datetime(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    try:
        normalized = text.replace("Z", "+00:00")
        dt = datetime.fromisoformat(normalized)
    except Exception:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def _legacy_sync_enabled() -> bool:
    return _is_truthy(os.getenv("ISPEC_LEGACY_SYNC_ENABLED"))


def _legacy_sync_interval_seconds() -> int:
    raw = (os.getenv("ISPEC_LEGACY_SYNC_INTERVAL_SECONDS") or "").strip()
    if not raw:
        return 3600
    parsed = _safe_int(raw)
    if parsed is None:
        return 3600
    return _clamp_int(int(parsed), min_value=60, max_value=7 * 24 * 3600)


def _legacy_sync_payload_from_env(*, agent_id: str, run_id: str) -> tuple[dict[str, Any], int]:
    interval_seconds = _legacy_sync_interval_seconds()
    limit = _safe_int(os.getenv("ISPEC_LEGACY_SYNC_LIMIT")) or 1000
    max_pages = _safe_int(os.getenv("ISPEC_LEGACY_SYNC_MAX_PAGES"))
    if max_pages is not None and max_pages <= 0:
        max_pages = None
    max_project_comments = _safe_int(os.getenv("ISPEC_LEGACY_SYNC_MAX_PROJECT_COMMENTS")) or 25
    max_experiment_runs = _safe_int(os.getenv("ISPEC_LEGACY_SYNC_MAX_EXPERIMENT_RUNS")) or 25
    recent_project_comment_days = _safe_int(os.getenv("ISPEC_LEGACY_SYNC_RECENT_PROJECT_COMMENT_DAYS"))
    if recent_project_comment_days is None:
        recent_project_comment_days = 30
    elif recent_project_comment_days <= 0:
        recent_project_comment_days = None
    recent_project_comment_scan_limit = _safe_int(
        os.getenv("ISPEC_LEGACY_SYNC_RECENT_PROJECT_COMMENT_SCAN_LIMIT")
    )
    if recent_project_comment_scan_limit is None:
        recent_project_comment_scan_limit = max(1000, int(limit))
    elif recent_project_comment_scan_limit <= 0:
        recent_project_comment_scan_limit = None
    backfill_missing_env = os.getenv("ISPEC_LEGACY_SYNC_BACKFILL_MISSING")
    backfill_missing = True if backfill_missing_env is None else _is_truthy(backfill_missing_env)

    payload: dict[str, Any] = {
        "limit": int(limit),
        "max_pages": int(max_pages) if isinstance(max_pages, int) else None,
        "backfill_missing": bool(backfill_missing),
        "max_project_comments": int(max_project_comments),
        "max_experiment_runs": int(max_experiment_runs),
        "recent_project_comment_days": (
            int(recent_project_comment_days)
            if isinstance(recent_project_comment_days, int)
            else None
        ),
        "recent_project_comment_scan_limit": (
            int(recent_project_comment_scan_limit)
            if isinstance(recent_project_comment_scan_limit, int)
            else None
        ),
        "meta": {"enqueued_by": "supervisor", "agent_id": agent_id, "run_id": run_id},
    }
    return payload, interval_seconds


def _ensure_legacy_sync_scheduled_commands(*, agent_id: str, run_id: str) -> dict[str, Any]:
    if not _legacy_sync_enabled():
        return {"ok": True, "disabled": True}

    now = utcnow()
    payload, interval_seconds = _legacy_sync_payload_from_env(agent_id=agent_id, run_id=run_id)
    max_attempts = _safe_int(os.getenv("ISPEC_LEGACY_SYNC_MAX_ATTEMPTS")) or 1
    max_attempts = _clamp_int(int(max_attempts), min_value=1, max_value=10)

    with get_agent_session() as db:
        run = db.query(AgentRun).filter(AgentRun.run_id == run_id).one()
        scheduler_state = _load_scheduler_state(run)
        legacy_state = scheduler_state.get("legacy_sync")
        if not isinstance(legacy_state, dict):
            legacy_state = {}
        legacy_state = dict(legacy_state)

        existing = (
            db.query(AgentCommand.id)
            .filter(AgentCommand.command_type == COMMAND_LEGACY_SYNC_ALL)
            .filter(AgentCommand.status.in_(["queued", "running"]))
            .first()
        )
        if existing is not None:
            return {"ok": True, "scheduled": 0, "skipped": 1, "reason": "already_enqueued"}

        last_attempted_at = _parse_iso_datetime(legacy_state.get("last_attempted_at"))
        if last_attempted_at is not None:
            elapsed = (now - last_attempted_at).total_seconds()
            if elapsed < float(interval_seconds):
                return {
                    "ok": True,
                    "scheduled": 0,
                    "skipped": 1,
                    "reason": "interval_not_elapsed",
                    "seconds_until_due": int(float(interval_seconds) - elapsed),
                }

        cmd = AgentCommand(
            command_type=COMMAND_LEGACY_SYNC_ALL,
            status="queued",
            priority=0,
            created_at=now,
            updated_at=now,
            available_at=now,
            attempts=0,
            max_attempts=int(max_attempts),
            payload_json=payload,
            result_json={},
        )
        db.add(cmd)
        db.flush()

        legacy_state["next_enqueued_at"] = now.isoformat()
        legacy_state["next_command_id"] = int(cmd.id)
        legacy_state["interval_seconds"] = int(interval_seconds)

        scheduler_state["legacy_sync"] = legacy_state
        _save_scheduler_state(run=run, state=scheduler_state)
        db.commit()

        return {"ok": True, "scheduled": 1, "command_id": int(cmd.id), "interval_seconds": interval_seconds}


def _record_legacy_sync_attempt(
    *,
    run: AgentRun,
    command_id: int,
    execution: CommandExecution,
    ended_at: datetime,
) -> None:
    scheduler_state = _load_scheduler_state(run)
    legacy_state = scheduler_state.get("legacy_sync")
    if not isinstance(legacy_state, dict):
        legacy_state = {}

    legacy_state = dict(legacy_state)
    legacy_state["last_attempted_at"] = ended_at.isoformat()
    legacy_state["last_attempted_ok"] = bool(execution.ok)
    legacy_state["last_attempted_error"] = execution.error
    legacy_state["last_command_id"] = int(command_id)
    if execution.ok:
        legacy_state["last_succeeded_at"] = ended_at.isoformat()

    next_id = _safe_int(legacy_state.get("next_command_id"))
    if next_id is not None and int(next_id) == int(command_id):
        legacy_state.pop("next_enqueued_at", None)
        legacy_state.pop("next_command_id", None)

    scheduler_state["legacy_sync"] = legacy_state
    _save_scheduler_state(run=run, state=scheduler_state)


def _run_legacy_sync_all(payload: dict[str, Any]) -> CommandExecution:
    try:
        from ispec.db.legacy_sync_all import sync_legacy_all

        limit = _safe_int(payload.get("limit")) or 1000
        max_pages = _safe_int(payload.get("max_pages"))
        if max_pages is not None and max_pages <= 0:
            max_pages = None
        backfill_missing = payload.get("backfill_missing")
        if isinstance(backfill_missing, str):
            backfill_missing_bool = _is_truthy(backfill_missing)
        elif backfill_missing is None:
            backfill_missing_bool = True
        else:
            backfill_missing_bool = bool(backfill_missing)

        max_project_comments = _safe_int(payload.get("max_project_comments")) or 25
        max_experiment_runs = _safe_int(payload.get("max_experiment_runs")) or 25
        recent_project_comment_days = _safe_int(payload.get("recent_project_comment_days"))
        if recent_project_comment_days is not None and recent_project_comment_days <= 0:
            recent_project_comment_days = None
        recent_project_comment_scan_limit = _safe_int(payload.get("recent_project_comment_scan_limit"))
        if recent_project_comment_scan_limit is not None and recent_project_comment_scan_limit <= 0:
            recent_project_comment_scan_limit = None

        summary = sync_legacy_all(
            db_file_path=payload.get("db_file_path"),
            legacy_url=payload.get("legacy_url"),
            mapping_path=payload.get("mapping_path"),
            schema_path=payload.get("schema_path"),
            limit=int(limit),
            max_pages=max_pages,
            reset_cursor=bool(payload.get("reset_cursor") or False),
            dry_run=bool(payload.get("dry_run") or False),
            backfill_missing=bool(backfill_missing_bool),
            max_project_comments=int(max_project_comments),
            max_experiment_runs=int(max_experiment_runs),
            recent_project_comment_days=recent_project_comment_days,
            recent_project_comment_scan_limit=recent_project_comment_scan_limit,
            dump_json=payload.get("dump_json"),
        )
        return CommandExecution(ok=True, result=dict(summary))
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        return CommandExecution(ok=False, result={"ok": False, "error": error}, error=error)


def _legacy_push_project_comments_enabled() -> bool:
    return _is_truthy(os.getenv("ISPEC_LEGACY_PUSH_PROJECT_COMMENTS_ENABLED"))


def _legacy_push_project_comments_interval_seconds() -> int:
    raw = (os.getenv("ISPEC_LEGACY_PUSH_PROJECT_COMMENTS_INTERVAL_SECONDS") or "").strip()
    if not raw:
        return 1800
    parsed = _safe_int(raw)
    if parsed is None:
        return 1800
    return _clamp_int(int(parsed), min_value=60, max_value=7 * 24 * 3600)


def _legacy_push_project_comments_payload_from_env(*, agent_id: str, run_id: str) -> tuple[dict[str, Any], int]:
    interval_seconds = _legacy_push_project_comments_interval_seconds()
    limit = _safe_int(os.getenv("ISPEC_LEGACY_PUSH_PROJECT_COMMENTS_LIMIT")) or 5000
    limit = _clamp_int(int(limit), min_value=1, max_value=50000)
    project_id = _safe_int(os.getenv("ISPEC_LEGACY_PUSH_PROJECT_COMMENTS_PROJECT_ID"))
    if project_id is not None and int(project_id) <= 0:
        project_id = None
    recent_days = _safe_int(os.getenv("ISPEC_LEGACY_PUSH_PROJECT_COMMENTS_RECENT_DAYS"))
    if recent_days is None:
        recent_days = 30
    elif recent_days <= 0:
        recent_days = None

    dry_run_env = os.getenv("ISPEC_LEGACY_PUSH_PROJECT_COMMENTS_DRY_RUN")
    dry_run = _is_truthy(dry_run_env) if dry_run_env is not None else False

    payload: dict[str, Any] = {
        "project_id": int(project_id) if isinstance(project_id, int) else None,
        "limit": int(limit),
        "dry_run": bool(dry_run),
        "recent_days": int(recent_days) if isinstance(recent_days, int) else None,
        "meta": {"enqueued_by": "supervisor", "agent_id": agent_id, "run_id": run_id},
    }

    legacy_url = str(os.getenv("ISPEC_LEGACY_API_URL") or "").strip()
    if legacy_url:
        payload["legacy_url"] = legacy_url

    schema_path = str(os.getenv("ISPEC_LEGACY_SCHEMA_PATH") or "").strip()
    if schema_path:
        payload["schema_path"] = schema_path

    db_file_path = str(os.getenv("ISPEC_DB_PATH") or "").strip()
    if db_file_path:
        payload["db_file_path"] = db_file_path

    return payload, interval_seconds


def _legacy_push_project_comments_summary(summary: Any) -> dict[str, Any]:
    if not isinstance(summary, dict):
        summary = {}

    legacy_table = summary.get("legacy_table")
    return {
        "selected": int(_safe_int(summary.get("selected")) or 0),
        "candidate_comments": int(_safe_int(summary.get("candidate_comments")) or 0),
        "projects": int(_safe_int(summary.get("projects")) or 0),
        "legacy_table": str(legacy_table) if legacy_table else None,
        "legacy_existing_items": int(_safe_int(summary.get("legacy_existing_items")) or 0),
        "already_present": int(_safe_int(summary.get("already_present")) or 0),
        "would_insert": int(_safe_int(summary.get("would_insert")) or 0),
        "inserted": int(_safe_int(summary.get("inserted")) or 0),
        "skipped_blank": int(_safe_int(summary.get("skipped_blank")) or 0),
        "skipped_system": int(_safe_int(summary.get("skipped_system")) or 0),
        "duplicates_skipped": int(_safe_int(summary.get("duplicates_skipped")) or 0),
        "dry_run": bool(summary.get("dry_run")),
    }


def _ensure_legacy_push_project_comments_scheduled_commands(*, agent_id: str, run_id: str) -> dict[str, Any]:
    if not _legacy_push_project_comments_enabled():
        return {"ok": True, "disabled": True}

    now = utcnow()
    payload, interval_seconds = _legacy_push_project_comments_payload_from_env(agent_id=agent_id, run_id=run_id)
    max_attempts = _safe_int(os.getenv("ISPEC_LEGACY_PUSH_PROJECT_COMMENTS_MAX_ATTEMPTS")) or 1
    max_attempts = _clamp_int(int(max_attempts), min_value=1, max_value=10)

    with get_agent_session() as db:
        run = db.query(AgentRun).filter(AgentRun.run_id == run_id).one()
        scheduler_state = _load_scheduler_state(run)
        push_state = scheduler_state.get("legacy_push_project_comments")
        if not isinstance(push_state, dict):
            push_state = {}
        push_state = dict(push_state)

        existing = (
            db.query(AgentCommand.id)
            .filter(AgentCommand.command_type == COMMAND_LEGACY_PUSH_PROJECT_COMMENTS)
            .filter(AgentCommand.status.in_(["queued", "running"]))
            .first()
        )
        if existing is not None:
            return {"ok": True, "scheduled": 0, "skipped": 1, "reason": "already_enqueued"}

        last_attempted_at = _parse_iso_datetime(push_state.get("last_attempted_at"))
        if last_attempted_at is not None:
            elapsed = (now - last_attempted_at).total_seconds()
            if elapsed < float(interval_seconds):
                return {
                    "ok": True,
                    "scheduled": 0,
                    "skipped": 1,
                    "reason": "interval_not_elapsed",
                    "seconds_until_due": int(float(interval_seconds) - elapsed),
                }

        cmd = AgentCommand(
            command_type=COMMAND_LEGACY_PUSH_PROJECT_COMMENTS,
            status="queued",
            priority=0,
            created_at=now,
            updated_at=now,
            available_at=now,
            attempts=0,
            max_attempts=int(max_attempts),
            payload_json=payload,
            result_json={},
        )
        db.add(cmd)
        db.flush()

        push_state["next_enqueued_at"] = now.isoformat()
        push_state["next_command_id"] = int(cmd.id)
        push_state["interval_seconds"] = int(interval_seconds)
        push_state["project_id"] = _safe_int(payload.get("project_id"))
        push_state["limit"] = int(_safe_int(payload.get("limit")) or 0)
        push_state["dry_run"] = bool(payload.get("dry_run"))
        push_state["recent_days"] = _safe_int(payload.get("recent_days"))

        scheduler_state["legacy_push_project_comments"] = push_state
        _save_scheduler_state(run=run, state=scheduler_state)
        db.commit()

        return {"ok": True, "scheduled": 1, "command_id": int(cmd.id), "interval_seconds": interval_seconds}


def _record_legacy_push_project_comments_attempt(
    *,
    run: AgentRun,
    command_id: int,
    execution: CommandExecution,
    ended_at: datetime,
) -> None:
    scheduler_state = _load_scheduler_state(run)
    push_state = scheduler_state.get("legacy_push_project_comments")
    if not isinstance(push_state, dict):
        push_state = {}

    push_state = dict(push_state)
    push_state["last_attempted_at"] = ended_at.isoformat()
    push_state["last_attempted_ok"] = bool(execution.ok)
    push_state["last_attempted_error"] = execution.error
    push_state["last_command_id"] = int(command_id)
    push_state["last_summary"] = _legacy_push_project_comments_summary(execution.result)
    if execution.ok:
        push_state["last_succeeded_at"] = ended_at.isoformat()

    next_id = _safe_int(push_state.get("next_command_id"))
    if next_id is not None and int(next_id) == int(command_id):
        push_state.pop("next_enqueued_at", None)
        push_state.pop("next_command_id", None)

    scheduler_state["legacy_push_project_comments"] = push_state
    _save_scheduler_state(run=run, state=scheduler_state)


def _run_legacy_push_project_comments(payload: dict[str, Any]) -> CommandExecution:
    try:
        from ispec.db.legacy_sync import sync_project_comments_to_legacy

        project_id = _safe_int(payload.get("project_id"))
        if project_id is not None and int(project_id) <= 0:
            project_id = None

        dry_run_raw = payload.get("dry_run")
        if isinstance(dry_run_raw, str):
            dry_run = _is_truthy(dry_run_raw)
        else:
            dry_run = bool(dry_run_raw)

        recent_days = _safe_int(payload.get("recent_days"))
        if recent_days is not None and recent_days <= 0:
            recent_days = None

        summary = sync_project_comments_to_legacy(
            legacy_url=payload.get("legacy_url"),
            schema_path=payload.get("schema_path"),
            db_file_path=payload.get("db_file_path"),
            project_id=project_id,
            limit=int(_safe_int(payload.get("limit")) or 5000),
            dry_run=bool(dry_run),
            recent_days=recent_days,
        )
        return CommandExecution(ok=True, result=dict(summary))
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        return CommandExecution(ok=False, result={"ok": False, "error": error}, error=error)


def _agent_log_archive_enabled() -> bool:
    return _is_truthy(os.getenv("ISPEC_AGENT_LOG_ARCHIVE_ENABLED"))


def _agent_log_archive_interval_seconds() -> int:
    raw = (os.getenv("ISPEC_AGENT_LOG_ARCHIVE_INTERVAL_SECONDS") or "").strip()
    if not raw:
        return 6 * 3600
    parsed = _safe_int(raw)
    if parsed is None:
        return 6 * 3600
    return _clamp_int(int(parsed), min_value=300, max_value=7 * 24 * 3600)


def _agent_log_archive_payload_from_env(*, agent_id: str, run_id: str) -> tuple[dict[str, Any] | None, int]:
    interval_seconds = _agent_log_archive_interval_seconds()
    archive_db_file_path = str(os.getenv("ISPEC_AGENT_ARCHIVE_DB_PATH") or "").strip()
    if not archive_db_file_path:
        return None, interval_seconds

    older_than_days = _safe_int(os.getenv("ISPEC_AGENT_LOG_ARCHIVE_OLDER_THAN_DAYS")) or 14
    older_than_days = _clamp_int(int(older_than_days), min_value=1, max_value=3650)
    batch_size = _safe_int(os.getenv("ISPEC_AGENT_LOG_ARCHIVE_BATCH_SIZE")) or 500
    batch_size = _clamp_int(int(batch_size), min_value=1, max_value=10_000)
    max_batches_raw = _safe_int(os.getenv("ISPEC_AGENT_LOG_ARCHIVE_MAX_BATCHES"))
    max_batches = (
        None
        if max_batches_raw is None or int(max_batches_raw) <= 0
        else _clamp_int(int(max_batches_raw), min_value=1, max_value=10_000)
    )

    prune_live_env = os.getenv("ISPEC_AGENT_LOG_ARCHIVE_PRUNE_LIVE")
    prune_live = True if prune_live_env is None else _is_truthy(prune_live_env)

    archive_steps_env = os.getenv("ISPEC_AGENT_LOG_ARCHIVE_INCLUDE_STEPS")
    archive_steps = True if archive_steps_env is None else _is_truthy(archive_steps_env)
    archive_events_env = os.getenv("ISPEC_AGENT_LOG_ARCHIVE_INCLUDE_EVENTS")
    archive_events = True if archive_events_env is None else _is_truthy(archive_events_env)
    archive_commands_env = os.getenv("ISPEC_AGENT_LOG_ARCHIVE_INCLUDE_COMMANDS")
    archive_commands = True if archive_commands_env is None else _is_truthy(archive_commands_env)

    payload: dict[str, Any] = {
        "archive_db_file_path": archive_db_file_path,
        "older_than_days": int(older_than_days),
        "batch_size": int(batch_size),
        "max_batches": max_batches,
        "prune_live": bool(prune_live),
        "archive_steps": bool(archive_steps),
        "archive_events": bool(archive_events),
        "archive_commands": bool(archive_commands),
        "meta": {"enqueued_by": "supervisor", "agent_id": agent_id, "run_id": run_id},
    }
    return payload, interval_seconds


def _agent_log_archive_summary(result: dict[str, Any] | None) -> dict[str, Any]:
    summary = result if isinstance(result, dict) else {}

    def _counts(key: str) -> dict[str, int]:
        item = summary.get(key)
        if not isinstance(item, dict):
            return {"matched": 0, "archived": 0, "pruned": 0}
        return {
            "matched": int(_safe_int(item.get("matched")) or 0),
            "archived": int(_safe_int(item.get("archived")) or 0),
            "pruned": int(_safe_int(item.get("pruned")) or 0),
        }

    return {
        "older_than_days": int(_safe_int(summary.get("older_than_days")) or 0),
        "dry_run": bool(summary.get("dry_run")),
        "prune_live": bool(summary.get("prune_live")),
        "runs_archived": int(_safe_int(summary.get("runs_archived")) or 0),
        "steps": _counts("steps"),
        "events": _counts("events"),
        "commands": _counts("commands"),
    }


def _ensure_agent_log_archive_scheduled_commands(*, agent_id: str, run_id: str) -> dict[str, Any]:
    if not _agent_log_archive_enabled():
        return {"ok": True, "disabled": True}

    now = utcnow()
    payload, interval_seconds = _agent_log_archive_payload_from_env(agent_id=agent_id, run_id=run_id)
    if payload is None:
        return {"ok": False, "scheduled": 0, "error": "missing_archive_database"}

    max_attempts = _safe_int(os.getenv("ISPEC_AGENT_LOG_ARCHIVE_MAX_ATTEMPTS")) or 1
    max_attempts = _clamp_int(int(max_attempts), min_value=1, max_value=10)

    with get_agent_session() as db:
        run = db.query(AgentRun).filter(AgentRun.run_id == run_id).one()
        scheduler_state = _load_scheduler_state(run)
        archive_state = scheduler_state.get("agent_log_archive")
        if not isinstance(archive_state, dict):
            archive_state = {}
        archive_state = dict(archive_state)

        existing = (
            db.query(AgentCommand.id)
            .filter(AgentCommand.command_type == COMMAND_ARCHIVE_AGENT_LOGS)
            .filter(AgentCommand.status.in_(["queued", "running"]))
            .first()
        )
        if existing is not None:
            return {"ok": True, "scheduled": 0, "skipped": 1, "reason": "already_enqueued"}

        last_attempted_at = _parse_iso_datetime(archive_state.get("last_attempted_at"))
        if last_attempted_at is not None:
            elapsed = (now - last_attempted_at).total_seconds()
            if elapsed < float(interval_seconds):
                return {
                    "ok": True,
                    "scheduled": 0,
                    "skipped": 1,
                    "reason": "interval_not_elapsed",
                    "seconds_until_due": int(float(interval_seconds) - elapsed),
                }

        cmd = AgentCommand(
            command_type=COMMAND_ARCHIVE_AGENT_LOGS,
            status="queued",
            priority=-1,
            created_at=now,
            updated_at=now,
            available_at=now,
            attempts=0,
            max_attempts=int(max_attempts),
            payload_json=payload,
            result_json={},
        )
        db.add(cmd)
        db.flush()

        archive_state["next_enqueued_at"] = now.isoformat()
        archive_state["next_command_id"] = int(cmd.id)
        archive_state["interval_seconds"] = int(interval_seconds)
        archive_state["older_than_days"] = int(_safe_int(payload.get("older_than_days")) or 0)
        archive_state["batch_size"] = int(_safe_int(payload.get("batch_size")) or 0)
        archive_state["max_batches"] = _safe_int(payload.get("max_batches"))

        scheduler_state["agent_log_archive"] = archive_state
        _save_scheduler_state(run=run, state=scheduler_state)
        db.commit()

        return {"ok": True, "scheduled": 1, "command_id": int(cmd.id), "interval_seconds": interval_seconds}


def _record_agent_log_archive_attempt(
    *,
    run: AgentRun,
    command_id: int,
    execution: CommandExecution,
    ended_at: datetime,
) -> None:
    scheduler_state = _load_scheduler_state(run)
    archive_state = scheduler_state.get("agent_log_archive")
    if not isinstance(archive_state, dict):
        archive_state = {}

    archive_state = dict(archive_state)
    archive_state["last_attempted_at"] = ended_at.isoformat()
    archive_state["last_attempted_ok"] = bool(execution.ok)
    archive_state["last_attempted_error"] = execution.error
    archive_state["last_command_id"] = int(command_id)
    archive_state["last_summary"] = _agent_log_archive_summary(execution.result)
    if execution.ok:
        archive_state["last_succeeded_at"] = ended_at.isoformat()

    next_id = _safe_int(archive_state.get("next_command_id"))
    if next_id is not None and int(next_id) == int(command_id):
        archive_state.pop("next_enqueued_at", None)
        archive_state.pop("next_command_id", None)

    scheduler_state["agent_log_archive"] = archive_state
    _save_scheduler_state(run=run, state=scheduler_state)


def _run_agent_log_archive(payload: dict[str, Any]) -> CommandExecution:
    try:
        from ispec.agent.archive import archive_agent_logs

        summary = archive_agent_logs(
            archive_db_file_path=payload.get("archive_db_file_path"),
            older_than_days=int(_safe_int(payload.get("older_than_days")) or 14),
            batch_size=int(_safe_int(payload.get("batch_size")) or 500),
            max_batches=_safe_int(payload.get("max_batches")),
            dry_run=bool(payload.get("dry_run") or False),
            prune_live=bool(payload.get("prune_live") if payload.get("prune_live") is not None else True),
            archive_steps=bool(payload.get("archive_steps") if payload.get("archive_steps") is not None else True),
            archive_events=bool(payload.get("archive_events") if payload.get("archive_events") is not None else True),
            archive_commands=bool(
                payload.get("archive_commands") if payload.get("archive_commands") is not None else True
            ),
            archive_journal_mode=payload.get("archive_journal_mode"),
        )
        return CommandExecution(ok=True, result=dict(summary))
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        return CommandExecution(ok=False, result={"ok": False, "error": error}, error=error)


def _seed_orchestrator_tick(
    *,
    delay_seconds: int = 5,
    agent_id: str | None = None,
    run_id: str | None = None,
) -> int | None:
    if not _orchestrator_enabled():
        return None
    if isinstance(agent_id, str) and agent_id.strip():
        _recover_stale_running_commands(
            agent_id=agent_id.strip(),
            run_id=(run_id or "").strip(),
        )
    with get_agent_session() as db:
        existing = (
            db.query(AgentCommand.id)
            .filter(AgentCommand.command_type == COMMAND_ORCHESTRATOR_TICK)
            .filter(AgentCommand.status.in_(["queued", "running"]))
            .first()
        )
        if existing is not None:
            return _scalar_row_int(existing)
    return _schedule_orchestrator_tick(delay_seconds=max(10, int(delay_seconds)))


def _log_command_done(*, cmd: ClaimedCommand, execution: CommandExecution, duration_ms: int) -> None:
    status_label = "deferred" if execution.defer_seconds is not None else ("succeeded" if execution.ok else "failed")
    extra_parts: list[str] = []
    if cmd.command_type == COMMAND_ORCHESTRATOR_TICK:
        decision = execution.result.get("decision") if isinstance(execution.result, dict) else None
        scheduled = execution.result.get("scheduled") if isinstance(execution.result, dict) else None
        tick_seconds = None
        if isinstance(decision, dict):
            tick_seconds = decision.get("next_tick_seconds")
        if isinstance(tick_seconds, (int, float)):
            extra_parts.append(f"next_tick={int(tick_seconds)}s")
        if isinstance(scheduled, list):
            extra_parts.append(f"scheduled={len(scheduled)}")
    elif cmd.command_type == COMMAND_SUPPORT_CHAT_TURN:
        session_id = execution.result.get("session_id") if isinstance(execution.result, dict) else None
        if not isinstance(session_id, str) or not session_id.strip():
            chat_request = cmd.payload.get("chat_request") if isinstance(cmd.payload, dict) else None
            session_id = chat_request.get("sessionId") if isinstance(chat_request, dict) else None
        if isinstance(session_id, str) and session_id.strip():
            extra_parts.append(f"session_id={session_id.strip()}")
    elif cmd.command_type == COMMAND_SLACK_POST_MESSAGE:
        channel = cmd.payload.get("channel")
        if isinstance(channel, str) and channel.strip():
            extra_parts.append(f"channel={channel.strip()}")
        schedule = cmd.payload.get("schedule")
        if isinstance(schedule, dict):
            schedule_name = schedule.get("name")
            if isinstance(schedule_name, str) and schedule_name.strip():
                extra_parts.append(f"schedule={schedule_name.strip()}")
    elif cmd.command_type == COMMAND_RUN_SCHEDULED_ASSISTANT_PROMPT:
        job = cmd.payload.get("job")
        if isinstance(job, dict):
            job_name = job.get("name")
            if isinstance(job_name, str) and job_name.strip():
                extra_parts.append(f"job={job_name.strip()}")
        schedule = cmd.payload.get("schedule")
        if isinstance(schedule, dict):
            schedule_name = schedule.get("name")
            if isinstance(schedule_name, str) and schedule_name.strip():
                extra_parts.append(f"schedule={schedule_name.strip()}")

    defer_seconds = execution.defer_seconds
    if isinstance(defer_seconds, int):
        extra_parts.append(f"defer_seconds={defer_seconds}")
    if isinstance(execution.error, str) and execution.error.strip():
        extra_parts.append(f"error={_truncate_text(execution.error, limit=200)}")

    extra_text = ""
    if extra_parts:
        extra_text = " " + " ".join(extra_parts[:8])
    logger.info(
        "Command done id=%s type=%s status=%s ok=%s duration_ms=%s%s",
        cmd.id,
        cmd.command_type,
        status_label,
        bool(execution.ok),
        int(duration_ms),
        extra_text,
    )


def _persist_command_execution(
    *,
    cmd: ClaimedCommand,
    execution: CommandExecution,
    run_id: str,
    step_started: datetime,
    step_ended: datetime,
    duration_ms: int,
) -> None:
    if execution.defer_seconds is not None:
        _defer_command(
            command_id=cmd.id,
            delay_seconds=int(execution.defer_seconds),
            result=dict(execution.result or {}),
            error=execution.error,
        )
    else:
        _finish_command(
            command_id=cmd.id,
            ok=bool(execution.ok),
            result=dict(execution.result or {}),
            error=execution.error,
        )

    try:
        with get_agent_session() as db:
            run = db.query(AgentRun).filter(AgentRun.run_id == run_id).one()
            step_index = int(run.step_index or 0)

            state_before = dict(run.state_json or {})
            state_after = dict(state_before)
            state_after["last_command"] = {
                "id": int(cmd.id),
                "command_type": cmd.command_type,
                "ok": bool(execution.ok),
                "ended_at": step_ended.isoformat(),
            }
            run.state_json = state_after
            run.step_index = step_index + 1
            run.updated_at = step_ended
            run.last_error = execution.error
            run.status_bar = _format_status_bar(state_after)

            if cmd.command_type == COMMAND_SLACK_POST_MESSAGE:
                try:
                    _record_slack_schedule_attempt(
                        run=run,
                        payload=dict(cmd.payload or {}),
                        execution=execution,
                        ended_at=step_ended,
                    )
                except Exception:
                    logger.exception("Failed recording Slack schedule attempt (command_id=%s)", cmd.id)
            elif cmd.command_type == COMMAND_RUN_SCHEDULED_ASSISTANT_PROMPT:
                try:
                    _record_assistant_schedule_attempt(
                        run=run,
                        payload=dict(cmd.payload or {}),
                        execution=execution,
                        ended_at=step_ended,
                    )
                except Exception:
                    logger.exception("Failed recording assistant schedule attempt (command_id=%s)", cmd.id)
            elif cmd.command_type == COMMAND_LEGACY_SYNC_ALL:
                try:
                    _record_legacy_sync_attempt(
                        run=run,
                        command_id=int(cmd.id),
                        execution=execution,
                        ended_at=step_ended,
                    )
                except Exception:
                    logger.exception("Failed recording legacy sync attempt (command_id=%s)", cmd.id)
            elif cmd.command_type == COMMAND_LEGACY_PUSH_PROJECT_COMMENTS:
                try:
                    _record_legacy_push_project_comments_attempt(
                        run=run,
                        command_id=int(cmd.id),
                        execution=execution,
                        ended_at=step_ended,
                    )
                except Exception:
                    logger.exception(
                        "Failed recording legacy project comment writeback attempt (command_id=%s)",
                        cmd.id,
                    )
            elif cmd.command_type == COMMAND_ARCHIVE_AGENT_LOGS:
                try:
                    _record_agent_log_archive_attempt(
                        run=run,
                        command_id=int(cmd.id),
                        execution=execution,
                        ended_at=step_ended,
                    )
                except Exception:
                    logger.exception("Failed recording agent log archive attempt (command_id=%s)", cmd.id)

            result_payload = dict(execution.result or {})
            step = AgentStep(
                run_pk=run.id,
                step_index=step_index,
                kind=cmd.command_type,
                started_at=step_started,
                ended_at=step_ended,
                duration_ms=int(duration_ms),
                ok=bool(execution.ok),
                severity=_severity_from_result(result_payload),
                error=execution.error,
                candidates_json=None,
                chosen_index=None,
                chosen_json={
                    "command_id": int(cmd.id),
                    "command_type": cmd.command_type,
                    "attempts": int(cmd.attempts),
                    "max_attempts": int(cmd.max_attempts),
                    "payload": dict(cmd.payload or {}),
                },
                prompt_json=execution.prompt,
                response_json=execution.response,
                tool_calls_json=None,
                tool_results_json=[result_payload],
                state_before_json=state_before,
                state_after_json=state_after,
            )
            db.add(step)
            db.commit()
    except Exception:
        logger.exception("Failed to persist command step (command_id=%s)", cmd.id)


def _process_one_command(*, agent_id: str, run_id: str) -> bool:
    assert_main_thread("supervisor._process_one_command")
    cmd = _claim_next_command(agent_id=agent_id, run_id=run_id)
    if cmd is None:
        return False

    payload_summary = _summarize_command_payload(cmd.command_type, cmd.payload)
    if payload_summary:
        payload_summary = " " + payload_summary
    logger.info(
        "Command start id=%s type=%s attempts=%s/%s%s",
        cmd.id,
        cmd.command_type,
        cmd.attempts,
        cmd.max_attempts,
        payload_summary,
    )

    step_started = utcnow()
    step_monotonic = time.monotonic()

    execution: CommandExecution
    try:
        if cmd.command_type == COMMAND_COMPACT_SESSION_MEMORY:
            ok, result, error = _compact_session_memory(cmd.payload)
            execution = CommandExecution(ok=bool(ok), result=dict(result or {}), error=error)
            if (
                error == "messages_not_ready"
                and int(cmd.max_attempts) > 0
                and int(cmd.attempts) < int(cmd.max_attempts)
            ):
                execution = CommandExecution(
                    ok=True,
                    result={
                        "ok": True,
                        "deferred": True,
                        "reason": "messages_not_ready",
                        "command_id": int(cmd.id),
                        "attempts": int(cmd.attempts),
                        "max_attempts": int(cmd.max_attempts),
                        "details": dict(result or {}),
                    },
                    error=None,
                    defer_seconds=2,
                )
            elif error == "messages_not_ready":
                try:
                    payload = dict(cmd.payload or {})
                    session_id = str(payload.get("session_id") or "").strip()
                    session_pk = _safe_int(payload.get("session_pk"))
                    with get_assistant_session() as db:
                        session = None
                        if session_pk is not None:
                            session = db.query(SupportSession).filter(SupportSession.id == int(session_pk)).first()
                        if session is None and session_id:
                            session = db.query(SupportSession).filter(SupportSession.session_id == session_id).first()
                        if session is not None:
                            state = _load_json_dict(getattr(session, "state_json", None))
                            state["conversation_memory_last_error"] = "messages_not_ready"
                            state["conversation_memory_requested_up_to_id"] = int(
                                _safe_int(state.get("conversation_memory_up_to_id")) or 0
                            )
                            state["conversation_memory_updated_at"] = utcnow().isoformat()
                            session.state_json = _dump_json(state)
                            session.updated_at = utcnow()
                except Exception:
                    logger.exception("Failed resetting compaction request after messages_not_ready (command_id=%s)", cmd.id)
        elif cmd.command_type == COMMAND_BUILD_SUPPORT_DIGEST:
            execution = _run_support_digest(
                payload=cmd.payload,
                agent_id=agent_id,
                run_id=run_id,
                command_id=int(cmd.id),
            )
        elif cmd.command_type == COMMAND_ASSESS_TACKLE_RESULTS:
            execution = _run_tackle_results_assess(
                payload=cmd.payload,
                agent_id=agent_id,
                run_id=run_id,
                command_id=int(cmd.id),
            )
        elif cmd.command_type == COMMAND_RUN_TACKLE_PROMPT:
            execution = _run_tackle_prompt_freeform(
                payload=cmd.payload,
                agent_id=agent_id,
                run_id=run_id,
                command_id=int(cmd.id),
            )
        elif cmd.command_type == COMMAND_RUN_SCHEDULED_ASSISTANT_PROMPT:
            execution = _run_scheduled_assistant_prompt(
                payload=cmd.payload,
                agent_id=agent_id,
                run_id=run_id,
                command_id=int(cmd.id),
            )
        elif cmd.command_type == COMMAND_ORCHESTRATOR_TICK:
            execution = _run_orchestrator_tick(
                payload=cmd.payload,
                agent_id=agent_id,
                run_id=run_id,
                command_id=cmd.id,
            )
        elif cmd.command_type == COMMAND_SUPPORT_CHAT_TURN:
            execution = _run_support_chat_turn(
                payload=cmd.payload,
                agent_id=agent_id,
                run_id=run_id,
                command_id=int(cmd.id),
            )
        elif cmd.command_type == COMMAND_POST_SEND_PREPARE:
            execution = _run_post_send_prepare(
                payload=cmd.payload,
                agent_id=agent_id,
                run_id=run_id,
                command_id=int(cmd.id),
            )
        elif cmd.command_type == COMMAND_REVIEW_SUPPORT_SESSION:
            execution = _run_support_session_review(
                payload=cmd.payload,
                agent_id=agent_id,
                run_id=run_id,
                command_id=int(cmd.id),
            )
        elif cmd.command_type == COMMAND_REVIEW_REPO:
            execution = _run_repo_review(payload=cmd.payload, agent_id=agent_id, run_id=run_id)
        elif cmd.command_type == COMMAND_LEGACY_SYNC_ALL:
            execution = _run_legacy_sync_all(cmd.payload)
        elif cmd.command_type == COMMAND_LEGACY_PUSH_PROJECT_COMMENTS:
            execution = _run_legacy_push_project_comments(cmd.payload)
        elif cmd.command_type == COMMAND_ARCHIVE_AGENT_LOGS:
            execution = _run_agent_log_archive(cmd.payload)
        elif cmd.command_type == COMMAND_SLACK_POST_MESSAGE:
            execution = _run_slack_post_message(cmd.payload)
        elif cmd.command_type == COMMAND_DEV_RESTART_SERVICES:
            execution = _run_dev_restart_services(cmd.payload)
        else:
            error = f"Unknown command_type: {cmd.command_type}"
            execution = CommandExecution(
                ok=False,
                result={"ok": False, "error": error, "command_type": cmd.command_type},
                error=error,
            )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        execution = CommandExecution(
            ok=False,
            result={"ok": False, "error": error, "command_type": cmd.command_type},
            error=error,
        )

    duration_ms = int((time.monotonic() - step_monotonic) * 1000)
    step_ended = utcnow()

    _log_command_done(cmd=cmd, execution=execution, duration_ms=duration_ms)
    _persist_command_execution(
        cmd=cmd,
        execution=execution,
        run_id=run_id,
        step_started=step_started,
        step_ended=step_ended,
        duration_ms=duration_ms,
    )

    return True


@dataclass(frozen=True)
class SupervisorConfig:
    agent_id: str
    backend_base_url: str
    frontend_url: str
    interval_seconds: int
    timeout_seconds: float


_LLM_COMMAND_TYPES = {
    COMMAND_ORCHESTRATOR_TICK,
    COMMAND_REVIEW_SUPPORT_SESSION,
    COMMAND_BUILD_SUPPORT_DIGEST,
    COMMAND_REVIEW_REPO,
    COMMAND_RUN_SCHEDULED_ASSISTANT_PROMPT,
    COMMAND_ASSESS_TACKLE_RESULTS,
    COMMAND_RUN_TACKLE_PROMPT,
}


def _llm_task_for_command(*, cmd: ClaimedCommand, agent_id: str, run_id: str) -> LLMTask | None:
    if cmd.command_type == COMMAND_ORCHESTRATOR_TICK:
        return _task_orchestrator_tick(payload=cmd.payload, agent_id=agent_id, run_id=run_id, command_id=int(cmd.id))
    if cmd.command_type == COMMAND_REVIEW_SUPPORT_SESSION:
        return _task_support_session_review(
            payload=cmd.payload,
            agent_id=agent_id,
            run_id=run_id,
            command_id=int(cmd.id),
        )
    if cmd.command_type == COMMAND_BUILD_SUPPORT_DIGEST:
        return _task_support_digest(
            payload=cmd.payload,
            agent_id=agent_id,
            run_id=run_id,
            command_id=int(cmd.id),
        )
    if cmd.command_type == COMMAND_REVIEW_REPO:
        return _task_repo_review(payload=cmd.payload, agent_id=agent_id, run_id=run_id)
    if cmd.command_type == COMMAND_RUN_SCHEDULED_ASSISTANT_PROMPT:
        return _task_scheduled_assistant_prompt(
            payload=cmd.payload,
            agent_id=agent_id,
            run_id=run_id,
            command_id=int(cmd.id),
        )
    if cmd.command_type == COMMAND_ASSESS_TACKLE_RESULTS:
        return _task_tackle_results_assess(
            payload=cmd.payload,
            agent_id=agent_id,
            run_id=run_id,
            command_id=int(cmd.id),
        )
    if cmd.command_type == COMMAND_RUN_TACKLE_PROMPT:
        return _task_tackle_prompt_freeform(
            payload=cmd.payload,
            agent_id=agent_id,
            run_id=run_id,
            command_id=int(cmd.id),
        )
    return None


@dataclass
class _InflightInference:
    cmd: ClaimedCommand
    task: LLMTask
    step_started: datetime
    step_monotonic: float
    job_id: str


class _SupervisorCommandProcessor:
    """Supervisor command processor with optional inference broker.

    When `broker` is enabled, LLM calls are executed on a dedicated thread and
    the main supervisor thread stays responsive (heartbeats, non-LLM commands,
    and inference result finalization).
    """

    def __init__(
        self,
        *,
        agent_id: str,
        run_id: str,
        broker: InferenceBroker | None,
    ) -> None:
        self._agent_id = agent_id
        self._run_id = run_id
        self._broker = broker
        self._inflight: _InflightInference | None = None

        heartbeat_seconds = float(_supervisor_heartbeat_seconds())
        now_mono = time.monotonic()
        self._heartbeat_seconds = heartbeat_seconds
        self._heartbeat_due_at = now_mono + heartbeat_seconds
        self._command_touch_due_at = now_mono + min(heartbeat_seconds, 10.0)

    @property
    def inflight(self) -> _InflightInference | None:
        return self._inflight

    def start(self) -> None:
        if self._broker is not None:
            self._broker.start()

    def stop(self) -> None:
        if self._broker is not None:
            self._broker.stop()

    def _maybe_heartbeat(self) -> None:
        now_mono = time.monotonic()
        if now_mono >= self._heartbeat_due_at:
            _touch_supervisor_run(run_id=self._run_id)
            self._heartbeat_due_at = now_mono + float(self._heartbeat_seconds)
        if self._inflight is not None and now_mono >= self._command_touch_due_at:
            _touch_command_updated_at(command_id=int(self._inflight.cmd.id))
            self._command_touch_due_at = now_mono + min(float(self._heartbeat_seconds), 10.0)

    def _finalize_command(
        self,
        *,
        cmd: ClaimedCommand,
        execution: CommandExecution,
        step_started: datetime,
        step_monotonic: float,
    ) -> None:
        duration_ms = int((time.monotonic() - step_monotonic) * 1000)
        step_ended = utcnow()
        _log_command_done(cmd=cmd, execution=execution, duration_ms=duration_ms)
        _persist_command_execution(
            cmd=cmd,
            execution=execution,
            run_id=self._run_id,
            step_started=step_started,
            step_ended=step_ended,
            duration_ms=duration_ms,
        )

    def _handle_claimed_non_llm(self, cmd: ClaimedCommand) -> None:
        step_started = utcnow()
        step_monotonic = time.monotonic()

        execution: CommandExecution
        try:
            if cmd.command_type == COMMAND_COMPACT_SESSION_MEMORY:
                ok, result, error = _compact_session_memory(cmd.payload)
                execution = CommandExecution(ok=bool(ok), result=dict(result or {}), error=error)
                if (
                    error == "messages_not_ready"
                    and int(cmd.max_attempts) > 0
                    and int(cmd.attempts) < int(cmd.max_attempts)
                ):
                    execution = CommandExecution(
                        ok=True,
                        result={
                            "ok": True,
                            "deferred": True,
                            "reason": "messages_not_ready",
                            "command_id": int(cmd.id),
                            "attempts": int(cmd.attempts),
                            "max_attempts": int(cmd.max_attempts),
                            "details": dict(result or {}),
                        },
                        error=None,
                        defer_seconds=2,
                    )
                elif error == "messages_not_ready":
                    try:
                        payload = dict(cmd.payload or {})
                        session_id = str(payload.get("session_id") or "").strip()
                        session_pk = _safe_int(payload.get("session_pk"))
                        with get_assistant_session() as db:
                            session = None
                            if session_pk is not None:
                                session = db.query(SupportSession).filter(SupportSession.id == int(session_pk)).first()
                            if session is None and session_id:
                                session = db.query(SupportSession).filter(SupportSession.session_id == session_id).first()
                            if session is not None:
                                state = _load_json_dict(getattr(session, "state_json", None))
                                state["conversation_memory_last_error"] = "messages_not_ready"
                                state["conversation_memory_requested_up_to_id"] = int(
                                    _safe_int(state.get("conversation_memory_up_to_id")) or 0
                                )
                                state["conversation_memory_updated_at"] = utcnow().isoformat()
                                session.state_json = _dump_json(state)
                                session.updated_at = utcnow()
                    except Exception:
                        logger.exception(
                            "Failed resetting compaction request after messages_not_ready (command_id=%s)",
                            cmd.id,
                        )
            elif cmd.command_type == COMMAND_SUPPORT_CHAT_TURN:
                execution = _run_support_chat_turn(
                    payload=cmd.payload,
                    agent_id=self._agent_id,
                    run_id=self._run_id,
                    command_id=int(cmd.id),
                )
            elif cmd.command_type == COMMAND_POST_SEND_PREPARE:
                execution = _run_post_send_prepare(
                    payload=cmd.payload,
                    agent_id=self._agent_id,
                    run_id=self._run_id,
                    command_id=int(cmd.id),
                )
            elif cmd.command_type == COMMAND_LEGACY_SYNC_ALL:
                execution = _run_legacy_sync_all(cmd.payload)
            elif cmd.command_type == COMMAND_LEGACY_PUSH_PROJECT_COMMENTS:
                execution = _run_legacy_push_project_comments(cmd.payload)
            elif cmd.command_type == COMMAND_ARCHIVE_AGENT_LOGS:
                execution = _run_agent_log_archive(cmd.payload)
            elif cmd.command_type == COMMAND_SLACK_POST_MESSAGE:
                execution = _run_slack_post_message(cmd.payload)
            elif cmd.command_type == COMMAND_DEV_RESTART_SERVICES:
                execution = _run_dev_restart_services(cmd.payload)
            else:
                error = f"Unknown command_type: {cmd.command_type}"
                execution = CommandExecution(
                    ok=False,
                    result={"ok": False, "error": error, "command_type": cmd.command_type},
                    error=error,
                )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            execution = CommandExecution(
                ok=False,
                result={"ok": False, "error": error, "command_type": cmd.command_type},
                error=error,
            )

        self._finalize_command(cmd=cmd, execution=execution, step_started=step_started, step_monotonic=step_monotonic)

    def _handle_claimed_llm_sync(self, cmd: ClaimedCommand) -> None:
        step_started = utcnow()
        step_monotonic = time.monotonic()

        execution: CommandExecution
        try:
            if cmd.command_type == COMMAND_BUILD_SUPPORT_DIGEST:
                execution = _run_support_digest(
                    payload=cmd.payload,
                    agent_id=self._agent_id,
                    run_id=self._run_id,
                    command_id=int(cmd.id),
                )
            elif cmd.command_type == COMMAND_ASSESS_TACKLE_RESULTS:
                execution = _run_tackle_results_assess(
                    payload=cmd.payload,
                    agent_id=self._agent_id,
                    run_id=self._run_id,
                    command_id=int(cmd.id),
                )
            elif cmd.command_type == COMMAND_RUN_TACKLE_PROMPT:
                execution = _run_tackle_prompt_freeform(
                    payload=cmd.payload,
                    agent_id=self._agent_id,
                    run_id=self._run_id,
                    command_id=int(cmd.id),
                )
            elif cmd.command_type == COMMAND_ORCHESTRATOR_TICK:
                execution = _run_orchestrator_tick(
                    payload=cmd.payload,
                    agent_id=self._agent_id,
                    run_id=self._run_id,
                    command_id=int(cmd.id),
                )
            elif cmd.command_type == COMMAND_REVIEW_SUPPORT_SESSION:
                execution = _run_support_session_review(
                    payload=cmd.payload,
                    agent_id=self._agent_id,
                    run_id=self._run_id,
                    command_id=int(cmd.id),
                )
            elif cmd.command_type == COMMAND_REVIEW_REPO:
                execution = _run_repo_review(payload=cmd.payload, agent_id=self._agent_id, run_id=self._run_id)
            elif cmd.command_type == COMMAND_RUN_SCHEDULED_ASSISTANT_PROMPT:
                execution = _run_scheduled_assistant_prompt(
                    payload=cmd.payload,
                    agent_id=self._agent_id,
                    run_id=self._run_id,
                    command_id=int(cmd.id),
                )
            else:
                error = f"Unknown command_type: {cmd.command_type}"
                execution = CommandExecution(
                    ok=False,
                    result={"ok": False, "error": error, "command_type": cmd.command_type},
                    error=error,
                )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            execution = CommandExecution(
                ok=False,
                result={"ok": False, "error": error, "command_type": cmd.command_type},
                error=error,
            )

        self._finalize_command(cmd=cmd, execution=execution, step_started=step_started, step_monotonic=step_monotonic)

    def _start_claimed_llm(self, cmd: ClaimedCommand) -> None:
        step_started = utcnow()
        step_monotonic = time.monotonic()

        task = _llm_task_for_command(cmd=cmd, agent_id=self._agent_id, run_id=self._run_id)
        if task is None or self._broker is None:
            error = "LLM task not available or broker disabled."
            execution = CommandExecution(ok=False, result={"ok": False, "error": error}, error="llm_task_missing")
            self._finalize_command(cmd=cmd, execution=execution, step_started=step_started, step_monotonic=step_monotonic)
            return

        try:
            request = next(task)
        except StopIteration as stop:
            value = stop.value
            execution = value if isinstance(value, CommandExecution) else CommandExecution(
                ok=False,
                result={"ok": False, "error": "LLM task returned invalid result type."},
                error="invalid_llm_task_result",
            )
            self._finalize_command(cmd=cmd, execution=execution, step_started=step_started, step_monotonic=step_monotonic)
            return
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            execution = CommandExecution(ok=False, result={"ok": False, "error": error}, error=error)
            self._finalize_command(cmd=cmd, execution=execution, step_started=step_started, step_monotonic=step_monotonic)
            return

        if not isinstance(request, InferenceRequest):
            execution = CommandExecution(
                ok=False,
                result={"ok": False, "error": "LLM task yielded invalid request type."},
                error="invalid_llm_task_request",
            )
            self._finalize_command(cmd=cmd, execution=execution, step_started=step_started, step_monotonic=step_monotonic)
            return

        job_id = self._broker.submit(command_id=int(cmd.id), request=request)
        self._inflight = _InflightInference(
            cmd=cmd,
            task=task,
            step_started=step_started,
            step_monotonic=step_monotonic,
            job_id=str(job_id),
        )

    def _poll_inflight_result(self) -> bool:
        if self._inflight is None or self._broker is None:
            return False
        result = self._broker.poll_result()
        if result is None:
            return False
        if str(result.job_id) != str(self._inflight.job_id) or int(result.command_id) != int(self._inflight.cmd.id):
            logger.warning(
                "Dropping unexpected inference result job_id=%s command_id=%s inflight_job_id=%s inflight_command_id=%s",
                result.job_id,
                result.command_id,
                self._inflight.job_id,
                self._inflight.cmd.id,
            )
            return True

        cmd = self._inflight.cmd
        task = self._inflight.task
        step_started = self._inflight.step_started
        step_monotonic = self._inflight.step_monotonic
        reply = result.reply

        try:
            next_item = task.send(reply)
        except StopIteration as stop:
            value = stop.value
            execution = value if isinstance(value, CommandExecution) else CommandExecution(
                ok=False,
                result={"ok": False, "error": "LLM task returned invalid result type."},
                error="invalid_llm_task_result",
            )
            self._inflight = None
            self._finalize_command(cmd=cmd, execution=execution, step_started=step_started, step_monotonic=step_monotonic)
            return True
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            execution = CommandExecution(ok=False, result={"ok": False, "error": error}, error=error)
            self._inflight = None
            self._finalize_command(cmd=cmd, execution=execution, step_started=step_started, step_monotonic=step_monotonic)
            return True

        if not isinstance(next_item, InferenceRequest):
            execution = CommandExecution(
                ok=False,
                result={"ok": False, "error": "LLM task yielded invalid request type."},
                error="invalid_llm_task_request",
            )
            self._inflight = None
            self._finalize_command(cmd=cmd, execution=execution, step_started=step_started, step_monotonic=step_monotonic)
            return True

        job_id = self._broker.submit(command_id=int(cmd.id), request=next_item)
        self._inflight = _InflightInference(
            cmd=cmd,
            task=task,
            step_started=step_started,
            step_monotonic=step_monotonic,
            job_id=str(job_id),
        )
        return True

    def tick(self) -> bool:
        """Return True if we did any work (command claimed, result finalized, etc)."""

        assert_main_thread("supervisor.processor.tick")
        self._maybe_heartbeat()

        if self._inflight is not None and self._broker is not None:
            if self._poll_inflight_result():
                return True

        exclude = _LLM_COMMAND_TYPES if (self._inflight is not None and self._broker is not None) else None
        cmd = _claim_next_command(agent_id=self._agent_id, run_id=self._run_id, exclude_command_types=exclude)
        if cmd is None:
            return False

        payload_summary = _summarize_command_payload(cmd.command_type, cmd.payload)
        payload_summary = f" {payload_summary}" if payload_summary else ""
        logger.info(
            "Command start id=%s type=%s attempts=%s/%s%s",
            cmd.id,
            cmd.command_type,
            cmd.attempts,
            cmd.max_attempts,
            payload_summary,
        )

        if cmd.command_type in _LLM_COMMAND_TYPES:
            if self._broker is not None:
                self._start_claimed_llm(cmd)
            else:
                self._handle_claimed_llm_sync(cmd)
            return True

        self._handle_claimed_non_llm(cmd)
        return True


def _default_agent_id() -> str:
    return socket.gethostname() or "ispec-supervisor"


def _severity_from_result(result: dict[str, Any]) -> str:
    explicit = result.get("severity")
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip().lower()
    ok = result.get("ok")
    if ok is False:
        return "error"
    return "info"


def _format_status_bar(state: dict[str, Any]) -> str:
    checks = state.get("checks") if isinstance(state, dict) else None
    if not isinstance(checks, dict):
        checks = {}

    def flag(name: str) -> str:
        payload = checks.get(name)
        ok = payload.get("ok") if isinstance(payload, dict) else None
        if ok is True:
            return f"{name}=ok"
        if ok is False:
            return f"{name}=bad"
        return f"{name}=?"

    return " | ".join(
        [
            flag("backend"),
            flag("frontend"),
            flag("db"),
            flag("diskwatcher"),
            flag("system"),
            flag("gpu"),
        ]
    )


def _supervisor_idle_max_seconds(base_interval_seconds: int) -> int:
    raw = (os.getenv("ISPEC_SUPERVISOR_IDLE_MAX_SECONDS") or "").strip()
    default_max = max(300, int(base_interval_seconds) * 12)
    if not raw:
        return int(default_max)
    try:
        parsed = int(raw)
    except ValueError:
        return int(default_max)
    return _clamp_int(parsed, min_value=max(1, int(base_interval_seconds)), max_value=86400)


def _supervisor_has_check_failures(state: dict[str, Any]) -> bool:
    checks = state.get("checks") if isinstance(state, dict) else None
    if not isinstance(checks, dict):
        return False
    for payload in checks.values():
        if isinstance(payload, dict) and payload.get("ok") is False:
            return True
    return False


def _seconds_until_next_queued_command(*, db, now: datetime) -> int | None:
    row = (
        db.query(AgentCommand)
        .filter(AgentCommand.status == "queued")
        .order_by(AgentCommand.available_at.asc(), AgentCommand.id.asc())
        .first()
    )
    if row is None:
        return None
    available_at = _as_utc_datetime(getattr(row, "available_at", None))
    if not isinstance(available_at, datetime):
        return None
    delta = int((available_at - now).total_seconds())
    if delta <= 0:
        return 0
    return int(delta)


def _supervisor_command_poll_seconds(*, base_interval_seconds: int) -> float:
    """Polling interval for checking queued commands during long sleeps."""

    raw = (os.getenv("ISPEC_SUPERVISOR_COMMAND_POLL_SECONDS") or "").strip()
    default_value = 0.5
    if not raw:
        return _clamp_float(float(default_value), min_value=0.1, max_value=max(0.1, float(base_interval_seconds)))
    try:
        return _clamp_float(float(raw), min_value=0.1, max_value=60.0)
    except ValueError:
        return float(default_value)


def _supervisor_heartbeat_seconds() -> float:
    """How often to touch supervisor.updated_at while idle.

    The API uses the supervisor heartbeat to decide whether it can safely route
    work to the command queue. Without a heartbeat, long idle sleeps make the
    supervisor look "dead" even though it's running.
    """

    raw = (os.getenv("ISPEC_SUPERVISOR_HEARTBEAT_SECONDS") or "").strip()
    if not raw:
        return 15.0
    try:
        return _clamp_float(float(raw), min_value=1.0, max_value=300.0)
    except ValueError:
        return 15.0


def _touch_supervisor_run(*, run_id: str) -> None:
    if not isinstance(run_id, str) or not run_id.strip():
        return
    assert_main_thread("supervisor._touch_supervisor_run")
    try:
        now = utcnow()
        with get_agent_session() as db:
            run = db.query(AgentRun).filter(AgentRun.run_id == run_id).first()
            if run is None:
                return
            status = str(getattr(run, "status", "") or "").strip().lower()
            if status != "running":
                return
            run.updated_at = now
            db.commit()
    except Exception:
        logger.exception("Failed touching supervisor heartbeat (run_id=%s)", run_id)


def _supervisor_sleep_with_command_polling(
    *,
    run_id: str,
    sleep_seconds: int,
    base_interval_seconds: int,
    process_one_command: Callable[[], bool],
) -> None:
    """Sleep for up to ``sleep_seconds``, but keep polling for queued commands.

    This bounds end-to-end latency for queue-backed chat (and other commands)
    even when the supervisor is configured to back off health checks aggressively.
    """

    total = max(0, int(sleep_seconds))
    if total <= 0:
        return

    # Fast path: if work showed up right before we were about to sleep, handle
    # it immediately without waiting for the first poll tick.
    if process_one_command():
        return

    poll_seconds = float(_supervisor_command_poll_seconds(base_interval_seconds=int(base_interval_seconds)))
    heartbeat_seconds = float(_supervisor_heartbeat_seconds())
    heartbeat_due_at = time.monotonic() + heartbeat_seconds

    deadline = time.monotonic() + float(total)
    while True:
        remaining = float(deadline - time.monotonic())
        if remaining <= 0:
            return

        step = min(float(remaining), float(poll_seconds))
        # Avoid a tight loop if poll_seconds is misconfigured.
        time.sleep(max(0.05, float(step)))

        if process_one_command():
            return

        now_mono = time.monotonic()
        if now_mono >= heartbeat_due_at:
            _touch_supervisor_run(run_id=run_id)
            heartbeat_due_at = now_mono + heartbeat_seconds


def _supervisor_dynamic_idle_sleep_seconds(
    *,
    run: AgentRun,
    state_after: dict[str, Any],
    base_interval_seconds: int,
    now: datetime,
    db,
) -> int:
    base = max(1, int(base_interval_seconds))
    max_idle = _supervisor_idle_max_seconds(base)

    sleep_seconds = base
    if _supervisor_has_check_failures(state_after):
        # When something is down, we still want to re-check quickly at first,
        # but we shouldn't spam the DB forever at the base interval.
        failure_streak = _safe_int(state_after.get("check_failure_streak")) or 1
        raw_failure_max = (os.getenv("ISPEC_SUPERVISOR_FAILURE_MAX_SECONDS") or "").strip()
        failure_max_default = min(int(max_idle), max(60, int(base) * 12))
        failure_max = failure_max_default
        if raw_failure_max:
            try:
                failure_max = _clamp_int(int(raw_failure_max), min_value=base, max_value=max_idle)
            except ValueError:
                failure_max = failure_max_default

        sleep_seconds = int(
            backoff_exponential_current(
                int(failure_streak),
                base_seconds=float(base),
                start_step=1,
                max_exp=4,
                cap_seconds=float(failure_max),
            )
        )
    else:
        summary = run.summary_json if isinstance(run.summary_json, dict) else {}
        orchestrator = summary.get("orchestrator")
        if isinstance(orchestrator, dict):
            reason = str(orchestrator.get("next_tick_reason") or "").strip().lower()
            idle_streak = _safe_int(orchestrator.get("idle_streak")) or 0
            next_tick_seconds = _safe_int(orchestrator.get("next_tick_seconds")) or 0

            if reason == "idle_backoff" and idle_streak > 0:
                sleep_seconds = int(
                    backoff_exponential_current(
                        int(idle_streak),
                        base_seconds=float(base),
                        start_step=1,
                        max_exp=6,
                    )
                )
                if next_tick_seconds > base:
                    sleep_seconds = max(sleep_seconds, max(base, int(next_tick_seconds // 4)))
            elif reason in {"work", "review_backlog", "work_inflight"}:
                sleep_seconds = base
            elif idle_streak > 0:
                sleep_seconds = int(
                    backoff_exponential_current(
                        int(idle_streak),
                        base_seconds=float(base),
                        start_step=1,
                        max_exp=4,
                    )
                    )

    sleep_seconds = _clamp_int(int(sleep_seconds), min_value=base, max_value=max_idle)

    due_in = _seconds_until_next_queued_command(db=db, now=now)
    if due_in is not None:
        sleep_seconds = min(int(sleep_seconds), max(1, int(due_in)))

    return max(1, int(sleep_seconds))


def _build_actions(config: SupervisorConfig) -> list[dict[str, Any]]:
    return [
        {"id": "backend", "description": "Check iSPEC /status"},
        {"id": "frontend", "description": "Check UI root page"},
        {"id": "db", "description": "Stat core/assistant/schedule/agent DB files"},
        {"id": "diskwatcher", "description": "Check diskwatcher config and DB file"},
        {"id": "system", "description": "Collect CPU/memory metrics (psutil)"},
        {"id": "gpu", "description": "Collect GPU metrics (nvidia-smi)"},
    ]


def _action_funcs(config: SupervisorConfig) -> dict[str, Callable[[], dict[str, Any]]]:
    return {
        "backend": lambda: _check_ispec_backend(
            base_url=config.backend_base_url,
            timeout_seconds=config.timeout_seconds,
        ),
        "frontend": lambda: _check_frontend(url=config.frontend_url, timeout_seconds=config.timeout_seconds),
        "db": _check_db_files,
        "diskwatcher": _check_diskwatcher,
        "system": _check_system_metrics,
        "gpu": _check_nvidia_smi,
    }


def run_supervisor(config: SupervisorConfig, *, once: bool = False) -> str:
    """Run the supervisor loop, returning the new run_id."""

    prev_thread_name = threading.current_thread().name
    try:
        threading.current_thread().name = "supervisor-main"
    except Exception:
        pass

    # Declare the main thread before any DB writes so invariant checks can
    # reliably fail fast if we ever run orchestration code on a worker thread.
    set_main_thread(owner="supervisor")
    thread_main = main_thread_info()

    run_id = uuid.uuid4().hex
    started_at = utcnow()
    pid_path = _write_supervisor_pid()
    broker_enabled, broker_error = _inference_broker_enabled_status()
    if broker_error:
        logger.warning("Inference broker disabled: %s", broker_error)
    broker: InferenceBroker | None = None
    if broker_enabled:
        broker = InferenceBroker()
        broker.start()
    inference_broker_info: dict[str, Any] = {
        "enabled": bool(broker_enabled),
        "error": broker_error,
        "thread": (
            {
                "name": str(broker.thread.name),
                "ident": int(broker.thread.ident) if broker.thread.ident is not None else None,
            }
            if broker is not None
            else None
        ),
    }
    _write_supervisor_state(
        {
            "schema_version": 1,
            "kind": "supervisor",
            "status": "running",
            "run_id": run_id,
            "agent_id": config.agent_id,
            "pid": os.getpid(),
            "thread_main": thread_main,
            "inference_broker": inference_broker_info,
            "started_at": started_at.isoformat(),
            "backend_base_url": config.backend_base_url,
            "frontend_url": config.frontend_url,
            "interval_seconds": int(config.interval_seconds),
            "timeout_seconds": float(config.timeout_seconds),
        }
    )
    with get_agent_session() as db:
        previous_orchestrator: dict[str, Any] | None = None
        previous_scheduler: dict[str, Any] | None = None
        try:
            previous = (
                db.query(AgentRun)
                .filter(AgentRun.agent_id == config.agent_id)
                .filter(AgentRun.kind == "supervisor")
                .order_by(AgentRun.created_at.desc(), AgentRun.id.desc())
                .first()
            )
            if previous is not None and isinstance(previous.summary_json, dict):
                previous_state = previous.summary_json.get("orchestrator")
                if isinstance(previous_state, dict) and previous_state:
                    previous_orchestrator = dict(previous_state)
                previous_sched = previous.summary_json.get("scheduler")
                if isinstance(previous_sched, dict) and previous_sched:
                    previous_scheduler = dict(previous_sched)
        except Exception:
            previous_orchestrator = None
            previous_scheduler = None

        summary_json: dict[str, Any] = {}
        if previous_orchestrator:
            summary_json["orchestrator"] = previous_orchestrator
        if previous_scheduler:
            summary_json["scheduler"] = previous_scheduler
        run = AgentRun(
            run_id=run_id,
            agent_id=config.agent_id,
            kind="supervisor",
            status="running",
            created_at=started_at,
            updated_at=started_at,
            config_json={
                "pid": os.getpid(),
                "thread_main": thread_main,
                "inference_broker": inference_broker_info,
                "backend_base_url": config.backend_base_url,
                "frontend_url": config.frontend_url,
                "interval_seconds": config.interval_seconds,
                "timeout_seconds": config.timeout_seconds,
            },
            state_json={"checks": {}},
            summary_json=summary_json,
        )
        db.add(run)
        db.commit()
        db.refresh(run)

    actions = _build_actions(config)
    funcs = _action_funcs(config)
    processor = _SupervisorCommandProcessor(agent_id=config.agent_id, run_id=run_id, broker=broker)

    logger.info("Supervisor run started (run_id=%s, agent_id=%s)", run_id, config.agent_id)
    stale_recovery = _recover_stale_running_commands(agent_id=config.agent_id, run_id=run_id)
    if int(stale_recovery.get("recovered") or 0) > 0:
        logger.warning(
            "Recovered stale running commands recovered=%s running_total=%s stale_seconds=%s",
            stale_recovery.get("recovered"),
            stale_recovery.get("running_total"),
            stale_recovery.get("stale_seconds"),
        )
    seeded_id = _seed_orchestrator_tick(
        delay_seconds=_orchestrator_tick_min_seconds(),
        agent_id=config.agent_id,
        run_id=run_id,
    )
    if seeded_id is not None:
        logger.info("Seeded orchestrator tick (command_id=%s)", seeded_id)

    schedule_poll_seconds = _clamp_int(
        _safe_int(os.getenv("ISPEC_SLACK_SCHEDULE_POLL_SECONDS")) or 30,
        min_value=10,
        max_value=3600,
    )
    last_schedule_poll_at: datetime | None = None
    legacy_sync_poll_seconds = _clamp_int(
        _safe_int(os.getenv("ISPEC_LEGACY_SYNC_POLL_SECONDS")) or 60,
        min_value=10,
        max_value=3600,
    )
    last_legacy_sync_poll_at: datetime | None = None
    legacy_push_project_comments_poll_seconds = _clamp_int(
        _safe_int(os.getenv("ISPEC_LEGACY_PUSH_PROJECT_COMMENTS_POLL_SECONDS")) or 60,
        min_value=10,
        max_value=3600,
    )
    last_legacy_push_project_comments_poll_at: datetime | None = None
    agent_log_archive_poll_seconds = _clamp_int(
        _safe_int(os.getenv("ISPEC_AGENT_LOG_ARCHIVE_POLL_SECONDS")) or 300,
        min_value=30,
        max_value=3600,
    )
    last_agent_log_archive_poll_at: datetime | None = None
    once_command_started = False

    final_status = "stopped"
    final_error: str | None = None
    try:
        while True:
            assert_main_thread("supervisor.loop")
            now = utcnow()
            if last_schedule_poll_at is None or (now - last_schedule_poll_at).total_seconds() >= schedule_poll_seconds:
                try:
                    seeded = _ensure_slack_scheduled_commands(agent_id=config.agent_id, run_id=run_id)
                    if int(seeded.get("scheduled") or 0) > 0:
                        logger.info(
                            "Enqueued Slack scheduled messages scheduled=%s skipped=%s",
                            seeded.get("scheduled"),
                            seeded.get("skipped"),
                        )
                    seeded_assistant = _ensure_assistant_scheduled_commands(agent_id=config.agent_id, run_id=run_id)
                    if int(seeded_assistant.get("scheduled") or 0) > 0:
                        logger.info(
                            "Enqueued assistant scheduled jobs scheduled=%s skipped=%s",
                            seeded_assistant.get("scheduled"),
                            seeded_assistant.get("skipped"),
                        )
                except Exception:
                    logger.exception("Failed ensuring scheduled commands")
                last_schedule_poll_at = now

            if (
                last_legacy_sync_poll_at is None
                or (now - last_legacy_sync_poll_at).total_seconds() >= legacy_sync_poll_seconds
            ):
                try:
                    seeded = _ensure_legacy_sync_scheduled_commands(agent_id=config.agent_id, run_id=run_id)
                    if int(seeded.get("scheduled") or 0) > 0:
                        logger.info(
                            "Enqueued legacy sync command scheduled=%s",
                            seeded.get("scheduled"),
                        )
                except Exception:
                    logger.exception("Failed ensuring legacy sync commands")
                last_legacy_sync_poll_at = now

            if (
                last_legacy_push_project_comments_poll_at is None
                or (now - last_legacy_push_project_comments_poll_at).total_seconds()
                >= legacy_push_project_comments_poll_seconds
            ):
                try:
                    seeded = _ensure_legacy_push_project_comments_scheduled_commands(
                        agent_id=config.agent_id,
                        run_id=run_id,
                    )
                    if int(seeded.get("scheduled") or 0) > 0:
                        logger.info(
                            "Enqueued legacy project comment writeback command scheduled=%s",
                            seeded.get("scheduled"),
                        )
                except Exception:
                    logger.exception("Failed ensuring legacy project comment writeback commands")
                last_legacy_push_project_comments_poll_at = now

            if (
                last_agent_log_archive_poll_at is None
                or (now - last_agent_log_archive_poll_at).total_seconds() >= agent_log_archive_poll_seconds
            ):
                try:
                    seeded = _ensure_agent_log_archive_scheduled_commands(agent_id=config.agent_id, run_id=run_id)
                    if int(seeded.get("scheduled") or 0) > 0:
                        logger.info(
                            "Enqueued agent log archive command scheduled=%s",
                            seeded.get("scheduled"),
                        )
                    elif seeded.get("error"):
                        logger.warning(
                            "Skipped agent log archive scheduling error=%s",
                            seeded.get("error"),
                        )
                except Exception:
                    logger.exception("Failed ensuring agent log archive commands")
                last_agent_log_archive_poll_at = now

            did_command_work = processor.tick()
            if did_command_work:
                if once:
                    once_command_started = True
                    if processor.inflight is None:
                        break
                continue
            if processor.inflight is not None and broker is not None:
                # Keep the main loop responsive while inference is running.
                time.sleep(0.05)
                continue

            step_started = utcnow()
            step_monotonic = time.monotonic()
            sleep_seconds = max(1, int(config.interval_seconds))
            with get_agent_session() as db:
                run = db.query(AgentRun).filter(AgentRun.run_id == run_id).one()
                step_index = int(run.step_index or 0)

                if not actions:
                    raise RuntimeError("No supervisor actions configured.")

                candidates = list(actions)
                chosen_index = step_index % len(candidates)
                chosen = candidates[chosen_index]
                action_id = str(chosen.get("id"))

                result: dict[str, Any]
                ok = True
                error: str | None = None
                try:
                    func = funcs.get(action_id)
                    if func is None:
                        raise KeyError(f"Unknown action: {action_id}")
                    result = func()
                    if result.get("ok") is False:
                        ok = False
                except Exception as exc:
                    ok = False
                    error = f"{type(exc).__name__}: {exc}"
                    result = {"ok": False, "error": error}

                duration_ms = int((time.monotonic() - step_monotonic) * 1000)
                step_ended = utcnow()
                severity = _severity_from_result(result)

                state_before = dict(run.state_json or {})
                checks = state_before.get("checks")
                if not isinstance(checks, dict):
                    checks = {}
                checks[action_id] = {"checked_at": step_ended.isoformat(), **result}
                state_after = {**state_before, "checks": checks, "last_action": action_id}
                prev_failure_streak = _safe_int(state_before.get("check_failure_streak")) or 0
                if _supervisor_has_check_failures(state_after):
                    state_after["check_failure_streak"] = int(prev_failure_streak) + 1
                else:
                    state_after["check_failure_streak"] = 0

                run.state_json = state_after
                run.step_index = step_index + 1
                run.updated_at = step_ended
                run.last_error = error
                run.status_bar = _format_status_bar(state_after)

                step = AgentStep(
                    run_pk=run.id,
                    step_index=step_index,
                    kind=action_id,
                    started_at=step_started,
                    ended_at=step_ended,
                    duration_ms=duration_ms,
                    ok=ok,
                    severity=severity,
                    error=error,
                    candidates_json=candidates,
                    chosen_index=chosen_index,
                    chosen_json=chosen,
                    tool_results_json=[result],
                    state_before_json=state_before,
                    state_after_json=state_after,
                )
                db.add(step)

                event = AgentEvent(
                    agent_id=config.agent_id,
                    event_type="supervisor",
                    ts=step_ended,
                    received_at=step_ended,
                    name=action_id,
                    severity=severity,
                    trace_id=f"{run_id}:{step_index}",
                    correlation_id=run_id,
                    payload_json=json.dumps(
                        {
                            "run_id": run_id,
                            "step_index": step_index,
                            "action": action_id,
                            "ok": ok,
                            "result": result,
                            "status_bar": run.status_bar,
                        },
                        separators=(",", ":"),
                        sort_keys=True,
                    ),
                )
                db.add(event)
                sleep_seconds = _supervisor_dynamic_idle_sleep_seconds(
                    run=run,
                    state_after=state_after,
                    base_interval_seconds=int(config.interval_seconds),
                    now=step_ended,
                    db=db,
                )
                db.commit()

            if once:
                break
            _supervisor_sleep_with_command_polling(
                run_id=run_id,
                sleep_seconds=max(1, int(sleep_seconds)),
                base_interval_seconds=int(config.interval_seconds),
                process_one_command=processor.tick,
            )
    except KeyboardInterrupt:
        logger.info("Supervisor interrupted (run_id=%s)", run_id)
    except Exception as exc:
        final_status = "failed"
        final_error = f"{type(exc).__name__}: {exc}"
        logger.exception("Supervisor crashed (run_id=%s)", run_id)
        raise
    finally:
        try:
            processor.stop()
        except Exception:
            logger.exception("Failed stopping inference broker (run_id=%s)", run_id)

        try:
            assert_main_thread("supervisor.shutdown")
            ended_at = utcnow()
            with get_agent_session() as db:
                run = db.query(AgentRun).filter(AgentRun.run_id == run_id).one()
                run.status = final_status
                run.ended_at = ended_at
                run.updated_at = ended_at
                if final_status != "stopped" and final_error:
                    run.last_error = final_error
                db.commit()
        except Exception:
            logger.exception("Failed to mark supervisor run stopped (run_id=%s)", run_id)

        _write_supervisor_state(
            {
                "schema_version": 1,
                "kind": "supervisor",
                "status": final_status,
                "run_id": run_id,
                "agent_id": config.agent_id,
                "pid": os.getpid(),
                "thread_main": thread_main,
                "inference_broker": inference_broker_info,
                "started_at": started_at.isoformat(),
                "ended_at": utcnow().isoformat(),
                "error": final_error,
                "backend_base_url": config.backend_base_url,
                "frontend_url": config.frontend_url,
                "interval_seconds": int(config.interval_seconds),
                "timeout_seconds": float(config.timeout_seconds),
            }
        )
        _remove_supervisor_pid(pid_path)
        try:
            threading.current_thread().name = prev_thread_name
        except Exception:
            pass

    logger.info("Supervisor run stopped (run_id=%s)", run_id)
    return run_id
