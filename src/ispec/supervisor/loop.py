from __future__ import annotations

import json
import os
import socket
import subprocess
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Callable

import psutil
import requests

from ispec.agent.connect import get_agent_db_uri, get_agent_session
from ispec.agent.commands import (
    COMMAND_COMPACT_SESSION_MEMORY,
    COMMAND_ORCHESTRATOR_TICK,
    COMMAND_REVIEW_REPO,
    COMMAND_REVIEW_SUPPORT_SESSION,
)
from ispec.agent.models import AgentCommand, AgentEvent, AgentRun, AgentStep
from ispec.assistant.compaction import distill_conversation_memory
from ispec.assistant.connect import get_assistant_db_uri, get_assistant_session
from ispec.assistant.models import (
    SupportMemory,
    SupportMemoryEvidence,
    SupportMessage,
    SupportSession,
    SupportSessionReview,
)
from ispec.assistant.service import generate_reply
from ispec.db.connect import get_db_path
from ispec.logging import get_logger
from ispec.schedule.connect import get_schedule_db_uri

logger = get_logger(__file__)


def utcnow() -> datetime:
    return datetime.now(UTC)


_TRUTHY = {"1", "true", "yes", "y", "on"}


def _is_truthy(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in _TRUTHY


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


def _parse_json_object(text: str | None) -> dict[str, Any] | None:
    if not text:
        return None
    raw = str(text).strip()
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        pass

    start = raw.find("{")
    end = raw.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        parsed = json.loads(raw[start : end + 1])
        return parsed if isinstance(parsed, dict) else None
    except Exception:
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


@dataclass(frozen=True)
class CommandExecution:
    ok: bool
    result: dict[str, Any]
    error: str | None = None
    prompt: dict[str, Any] | None = None
    response: dict[str, Any] | None = None


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


def _enqueue_command(
    *,
    command_type: str,
    payload: dict[str, Any] | None = None,
    priority: int = 0,
    available_at: datetime | None = None,
    max_attempts: int = 3,
) -> int | None:
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


_ORCHESTRATOR_STATE_VERSION = 1
_ORCHESTRATOR_DECISION_VERSION = 1
_SESSION_REVIEW_VERSION = 1
_REPO_REVIEW_VERSION = 1


def _orchestrator_enabled() -> bool:
    raw = (os.getenv("ISPEC_ORCHESTRATOR_ENABLED") or "").strip()
    if raw:
        return _is_truthy(raw)
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
        return 600
    try:
        return max(30, int(raw))
    except ValueError:
        return 600


def _orchestrator_max_commands_per_tick() -> int:
    raw = (os.getenv("ISPEC_ORCHESTRATOR_MAX_COMMANDS_PER_TICK") or "").strip()
    if not raw:
        return 2
    try:
        return _clamp_int(int(raw), min_value=0, max_value=6)
    except ValueError:
        return 2


def _load_orchestrator_state(run: AgentRun) -> dict[str, Any]:
    summary = run.summary_json if isinstance(run.summary_json, dict) else {}
    state = summary.get("orchestrator")
    if not isinstance(state, dict):
        state = {}
    if int(state.get("schema_version") or 0) != _ORCHESTRATOR_STATE_VERSION:
        state = {"schema_version": _ORCHESTRATOR_STATE_VERSION}
    if not isinstance(state.get("recent_thoughts"), list):
        state["recent_thoughts"] = []
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


def _assistant_snapshot(*, assistant_db) -> dict[str, Any]:
    try:
        total_sessions = int(assistant_db.query(SupportSession.id).count())
    except Exception:
        total_sessions = 0
    try:
        total_messages = int(assistant_db.query(SupportMessage.id).count())
    except Exception:
        total_messages = 0

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
        "recent_sessions": items,
        "sessions_needing_review": needs_review,
    }


def _orchestrator_decision_schema(*, max_commands: int) -> dict[str, Any]:
    allowed_commands = [COMMAND_REVIEW_SUPPORT_SESSION, COMMAND_REVIEW_REPO]
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


def _orchestrator_system_prompt(*, max_commands: int) -> str:
    lines = [
        "You are the iSPEC internal orchestrator.",
        "You run periodically to decide what self-work to do next.",
        "",
        "Goals:",
        "- Review new/updated user support sessions to spot issues and follow-ups.",
        "- Optionally review the codebase (backend + frontend) based on what users are running into.",
        "- Keep internal notes concise and actionable.",
        "",
        "Rules:",
        "- Enqueue at most one command per tick unless there is a strong reason for two.",
        "- If any sessions are marked as needing review, enqueue a support-session review command for the most recently updated session.",
        "- If unsure, enqueue nothing and schedule the next tick later.",
        "- Always include a short 'thoughts' string explaining your choice.",
        "- Return ONLY JSON that matches the schema.",
        "",
        f"Max commands this tick: {max_commands}.",
    ]
    return "\n".join(lines).strip()


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
        if cmd_type not in {COMMAND_REVIEW_SUPPORT_SESSION, COMMAND_REVIEW_REPO}:
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


def _schedule_orchestrator_tick(*, delay_seconds: int, current_command_id: int | None = None) -> int | None:
    if not _orchestrator_enabled():
        return None
    delay_seconds = max(10, int(delay_seconds))
    available_at = utcnow() + timedelta(seconds=delay_seconds)
    with get_agent_session() as db:
        existing = (
            db.query(AgentCommand)
            .filter(AgentCommand.command_type == COMMAND_ORCHESTRATOR_TICK)
            .filter(AgentCommand.status == "queued")
            .filter(AgentCommand.available_at > utcnow())
            .order_by(AgentCommand.available_at.asc())
            .first()
        )
        if existing is not None:
            return int(existing.id)
    return _enqueue_command(
        command_type=COMMAND_ORCHESTRATOR_TICK,
        payload={"source": "self_schedule", "previous_command_id": int(current_command_id) if current_command_id else None},
        priority=-5,
        available_at=available_at,
    )


def _run_orchestrator_tick(*, payload: dict[str, Any], agent_id: str, run_id: str, command_id: int) -> CommandExecution:
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

    state_for_prompt: dict[str, Any] = {}
    with get_agent_session() as agent_db:
        run = agent_db.query(AgentRun).filter(AgentRun.run_id == run_id).one()
        state = _load_orchestrator_state(run)
        state["ticks"] = int(state.get("ticks") or 0) + 1
        state["last_tick_at"] = utcnow().isoformat()
        recent_thoughts = [str(x) for x in state.get("recent_thoughts") if isinstance(x, str)]
        state["recent_thoughts"] = recent_thoughts[-10:]
        state_for_prompt = {
            "schema_version": int(state.get("schema_version") or 0),
            "ticks": int(state.get("ticks") or 0),
            "idle_streak": int(state.get("idle_streak") or 0),
            "error_streak": int(state.get("error_streak") or 0),
            "last_thought": _truncate_text(state.get("last_thought") if isinstance(state.get("last_thought"), str) else None, limit=600),
            "recent_thoughts": [_truncate_text(text, limit=300) for text in recent_thoughts[-5:]],
            "last_tick_at": state.get("last_tick_at"),
            "next_tick_seconds": state.get("next_tick_seconds"),
            "next_tick_command_id": state.get("next_tick_command_id"),
        }
        _save_orchestrator_state(run=run, state=state)
        agent_db.commit()

    with get_assistant_session() as assistant_db:
        assistant_snapshot = _assistant_snapshot(assistant_db=assistant_db)

    context = {
        "schema_version": 1,
        "agent": {"agent_id": agent_id, "run_id": run_id, "command_id": int(command_id)},
        "orchestrator": state_for_prompt,
        "assistant": assistant_snapshot,
        "requested": dict(payload or {}),
    }

    schema = _orchestrator_decision_schema(max_commands=max_commands)
    messages = [
        {"role": "system", "content": _orchestrator_system_prompt(max_commands=max_commands)},
        {"role": "user", "content": json.dumps(context, ensure_ascii=False)},
    ]
    reply = generate_reply(
        messages=messages,
        tools=None,
        vllm_extra_body={"guided_json": schema, "temperature": 0, "max_tokens": 800},
    )

    parsed = _parse_json_object(reply.content)
    validated = _validate_orchestrator_decision(
        parsed if isinstance(parsed, dict) else None,
        min_tick_seconds=min_tick,
        max_tick_seconds=max_tick,
        max_commands=max_commands,
    )

    sessions_needing_review = assistant_snapshot.get("sessions_needing_review") if isinstance(assistant_snapshot, dict) else None
    if not isinstance(sessions_needing_review, list):
        sessions_needing_review = []

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

    scheduled: list[dict[str, Any]] = []
    if validated is not None:
        now = utcnow()
        for cmd in validated["commands"]:
            delay_seconds = int(cmd.get("delay_seconds") or 0)
            scheduled_at = now + timedelta(seconds=delay_seconds)
            enqueued_id = _enqueue_command(
                command_type=str(cmd["command_type"]),
                payload=dict(cmd.get("payload") or {}),
                priority=int(cmd.get("priority") or 0),
                available_at=scheduled_at,
            )
            scheduled.append(
                {
                    "id": enqueued_id,
                    "command_type": cmd["command_type"],
                    "available_at": scheduled_at.isoformat(),
                    "priority": cmd.get("priority"),
                }
            )

        tick_seconds = int(validated.get("next_tick_seconds") or min_tick)
        scheduled_tick_id = _schedule_orchestrator_tick(
            delay_seconds=tick_seconds,
            current_command_id=command_id,
        )

        with get_agent_session() as agent_db:
            run = agent_db.query(AgentRun).filter(AgentRun.run_id == run_id).one()
            state = _load_orchestrator_state(run)
            state["last_thought"] = validated.get("thoughts") or ""
            state["recent_thoughts"] = (
                [*([str(x) for x in state.get("recent_thoughts") if isinstance(x, str)]), validated.get("thoughts") or ""]
            )[-10:]
            if scheduled:
                state["idle_streak"] = 0
                state["error_streak"] = 0
            else:
                state["idle_streak"] = int(state.get("idle_streak") or 0) + 1
            state["next_tick_seconds"] = tick_seconds
            state["next_tick_command_id"] = scheduled_tick_id
            _save_orchestrator_state(run=run, state=state)
            agent_db.commit()

        return CommandExecution(
            ok=True,
            result={
                "ok": True,
                "decision": validated,
                "scheduled": scheduled,
                "scheduled_tick_command_id": scheduled_tick_id,
                "assistant_snapshot": assistant_snapshot,
                "llm": {
                    "provider": reply.provider,
                    "model": reply.model,
                    "ok": bool(reply.ok),
                    "meta": reply.meta,
                },
            },
            prompt={"messages": messages, "guided_json": schema},
            response={"raw": reply.content, "parsed": parsed, "validated": validated},
        )

    # Fallback: exponential backoff when router output is invalid.
    with get_agent_session() as agent_db:
        run = agent_db.query(AgentRun).filter(AgentRun.run_id == run_id).one()
        state = _load_orchestrator_state(run)
        state["error_streak"] = int(state.get("error_streak") or 0) + 1
        backoff = min_tick * (2 ** min(6, int(state["error_streak"])))
        tick_seconds = _clamp_int(int(backoff), min_value=min_tick, max_value=max_tick)
        scheduled_tick_id = _schedule_orchestrator_tick(delay_seconds=tick_seconds, current_command_id=command_id)
        state["next_tick_seconds"] = tick_seconds
        state["next_tick_command_id"] = scheduled_tick_id
        _save_orchestrator_state(run=run, state=state)
        agent_db.commit()

    return CommandExecution(
        ok=False,
        result={
            "ok": False,
            "error": "Invalid orchestrator decision output.",
            "assistant_snapshot": assistant_snapshot,
            "scheduled_tick_seconds": tick_seconds,
            "scheduled_tick_command_id": scheduled_tick_id,
            "llm": {
                "provider": reply.provider,
                "model": reply.model,
                "ok": bool(reply.ok),
                "meta": reply.meta,
                "raw": _truncate_text(reply.content, limit=800),
            },
        },
        error="invalid_orchestrator_output",
        prompt={"messages": messages, "guided_json": schema},
        response={"raw": reply.content, "parsed": parsed},
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


def _session_review_system_prompt() -> str:
    return "\n".join(
        [
            "You are the iSPEC internal QA reviewer.",
            "You review a single support session transcript and write internal notes.",
            "Focus on: missed tool opportunities, incorrect claims, confusing UX guidance, bugs, and follow-ups.",
            "Do NOT call tools. Do NOT write anything user-facing.",
            "Return ONLY JSON that matches the schema.",
        ]
    ).strip()


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

    review = parsed.get("review")
    if not isinstance(review, dict):
        return None

    issues: list[dict[str, Any]] = []

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
        for item in raw_followups:
            if not isinstance(item, str):
                continue
            text = item.strip()
            if not text:
                continue
            followups.append(_truncate_text(text, limit=240))

    summary = ""
    if isinstance(review.get("summary"), str):
        summary = _truncate_text(review.get("summary"), limit=2000)
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
        "repo_search_queries": [],
        "followups": followups[:20],
    }


def _run_support_session_review(
    *,
    payload: dict[str, Any],
    agent_id: str,
    run_id: str,
    command_id: int | None = None,
) -> CommandExecution:
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
        messages = [
            {"role": "system", "content": _session_review_system_prompt()},
            {"role": "user", "content": json.dumps(context, ensure_ascii=False)},
        ]
        reply = generate_reply(
            messages=messages,
            tools=None,
            vllm_extra_body={"guided_json": schema, "temperature": 0, "max_tokens": 1200},
        )
        parsed = _parse_json_object(reply.content)
        coerced = _coerce_session_review_output(
            parsed if isinstance(parsed, dict) else None,
            session_id=session.session_id,
            target_message_id=int(target_id),
        )
        if not isinstance(coerced, dict) or int(coerced.get("schema_version") or 0) != _SESSION_REVIEW_VERSION:
            return CommandExecution(
                ok=False,
                result={
                    "ok": False,
                    "error": "Invalid review output.",
                    "raw": _truncate_text(reply.content, limit=1200),
                    "llm": {"provider": reply.provider, "model": reply.model, "meta": reply.meta},
                },
                error="invalid_review_output",
                prompt={"messages": messages, "guided_json": schema},
                response={"raw": reply.content, "parsed": parsed, "coerced": coerced},
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
                "llm": {"provider": reply.provider, "model": reply.model, "meta": reply.meta},
            },
            prompt={"messages": messages, "guided_json": schema},
            response={"raw": reply.content, "parsed": parsed, "coerced": coerced},
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


def _repo_review_system_prompt() -> str:
    return "\n".join(
        [
            "You are the iSPEC internal code reviewer.",
            "You are given a small set of code snippets and grep matches from the repo.",
            "Produce an internal review report with actionable recommendations.",
            "Do NOT invent files or line numbers; reference only what is provided.",
            "Return ONLY JSON that matches the schema.",
        ]
    ).strip()


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


def _run_repo_review(*, payload: dict[str, Any], agent_id: str, run_id: str) -> CommandExecution:
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
    messages = [
        {"role": "system", "content": _repo_review_system_prompt()},
        {"role": "user", "content": json.dumps(context, ensure_ascii=False)},
    ]
    reply = generate_reply(
        messages=messages,
        tools=None,
        vllm_extra_body={"guided_json": schema, "temperature": 0, "max_tokens": 1200},
    )
    parsed = _parse_json_object(reply.content)
    coerced = _coerce_repo_review_output(parsed if isinstance(parsed, dict) else None)
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
                "llm": {"provider": reply.provider, "model": reply.model, "meta": reply.meta},
            },
            error="invalid_repo_review_output",
            prompt={"messages": messages, "guided_json": schema},
            response={"raw": reply.content, "parsed": parsed, "coerced": coerced},
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
            "llm": {"provider": reply.provider, "model": reply.model, "meta": reply.meta},
        },
        prompt={"messages": messages, "guided_json": schema},
        response={"raw": reply.content, "parsed": parsed, "coerced": coerced},
    )


def _seed_orchestrator_tick(*, delay_seconds: int = 5) -> int | None:
    if not _orchestrator_enabled():
        return None
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


def _process_one_command(*, agent_id: str, run_id: str) -> bool:
    cmd = _claim_next_command(agent_id=agent_id, run_id=run_id)
    if cmd is None:
        return False

    step_started = utcnow()
    step_monotonic = time.monotonic()

    execution: CommandExecution
    try:
        if cmd.command_type == COMMAND_COMPACT_SESSION_MEMORY:
            ok, result, error = _compact_session_memory(cmd.payload)
            execution = CommandExecution(ok=bool(ok), result=dict(result or {}), error=error)
        elif cmd.command_type == COMMAND_ORCHESTRATOR_TICK:
            execution = _run_orchestrator_tick(
                payload=cmd.payload,
                agent_id=agent_id,
                run_id=run_id,
                command_id=cmd.id,
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

            result_payload = dict(execution.result or {})
            step = AgentStep(
                run_pk=run.id,
                step_index=step_index,
                kind=cmd.command_type,
                started_at=step_started,
                ended_at=step_ended,
                duration_ms=duration_ms,
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
    seeded_id = _seed_orchestrator_tick(delay_seconds=5)
    if seeded_id is not None:
        logger.info("Seeded orchestrator tick (command_id=%s)", seeded_id)

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
