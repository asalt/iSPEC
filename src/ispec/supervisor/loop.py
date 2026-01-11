from __future__ import annotations

import json
import os
import socket
import subprocess
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

import psutil
import requests

from ispec.agent.connect import get_agent_db_uri, get_agent_session
from ispec.agent.models import AgentEvent, AgentRun, AgentStep
from ispec.assistant.connect import get_assistant_db_uri
from ispec.db.connect import get_db_path
from ispec.logging import get_logger
from ispec.schedule.connect import get_schedule_db_uri

logger = get_logger(__file__)


def utcnow() -> datetime:
    return datetime.now(UTC)


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
    core_uri = (os.getenv("ISPEC_DB_PATH") or "").strip() or get_db_path()
    assistant_uri = get_assistant_db_uri()
    schedule_uri = get_schedule_db_uri()
    agent_uri = get_agent_db_uri()
    return {
        "ok": True,
        "core_db": _stat_path(_sqlite_path_from_uri(core_uri)),
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


@dataclass(frozen=True)
class SupervisorConfig:
    agent_id: str
    backend_base_url: str
    frontend_url: str
    interval_seconds: int
    timeout_seconds: float


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

    run_id = uuid.uuid4().hex
    started_at = utcnow()
    with get_agent_session() as db:
        run = AgentRun(
            run_id=run_id,
            agent_id=config.agent_id,
            kind="supervisor",
            status="running",
            created_at=started_at,
            updated_at=started_at,
            config_json={
                "backend_base_url": config.backend_base_url,
                "frontend_url": config.frontend_url,
                "interval_seconds": config.interval_seconds,
                "timeout_seconds": config.timeout_seconds,
            },
            state_json={"checks": {}},
            summary_json={},
        )
        db.add(run)
        db.commit()
        db.refresh(run)

    actions = _build_actions(config)
    funcs = _action_funcs(config)

    logger.info("Supervisor run started (run_id=%s, agent_id=%s)", run_id, config.agent_id)

    while True:
        step_started = utcnow()
        step_monotonic = time.monotonic()
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
            db.commit()

        if once:
            break
        time.sleep(max(1, int(config.interval_seconds)))

    with get_agent_session() as db:
        run = db.query(AgentRun).filter(AgentRun.run_id == run_id).one()
        run.status = "stopped"
        run.ended_at = utcnow()
        db.commit()

    logger.info("Supervisor run stopped (run_id=%s)", run_id)
    return run_id

