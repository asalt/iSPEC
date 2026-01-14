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
from ispec.agent.commands import COMMAND_COMPACT_SESSION_MEMORY
from ispec.agent.models import AgentCommand, AgentEvent, AgentRun, AgentStep
from ispec.assistant.compaction import distill_conversation_memory
from ispec.assistant.connect import get_assistant_db_uri, get_assistant_session
from ispec.assistant.models import SupportMemory, SupportMemoryEvidence, SupportMessage, SupportSession
from ispec.assistant.service import generate_reply
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


def _claim_next_command(*, agent_id: str, run_id: str) -> ClaimedCommand | None:
    now = utcnow()
    with get_agent_session() as db:
        cmd = (
            db.query(AgentCommand)
            .filter(AgentCommand.status == "queued")
            .filter(AgentCommand.available_at <= now)
            .order_by(AgentCommand.priority.desc(), AgentCommand.id.asc())
            .first()
        )
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


def _process_one_command(*, agent_id: str, run_id: str) -> bool:
    cmd = _claim_next_command(agent_id=agent_id, run_id=run_id)
    if cmd is None:
        return False

    ok = False
    result: dict[str, Any] = {"ok": False, "error": "Unhandled command."}
    error: str | None = None

    if cmd.command_type == COMMAND_COMPACT_SESSION_MEMORY:
        ok, result, error = _compact_session_memory(cmd.payload)
    else:
        error = f"Unknown command_type: {cmd.command_type}"
        result = {"ok": False, "error": error, "command_type": cmd.command_type}

    _finish_command(command_id=cmd.id, ok=ok, result=result, error=error)
    return True


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
        if _process_one_command(agent_id=config.agent_id, run_id=run_id):
            if once:
                break
            continue

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
