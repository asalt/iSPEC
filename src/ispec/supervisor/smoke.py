from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from ispec.agent.commands import COMMAND_ORCHESTRATOR_TICK
from ispec.agent.connect import get_agent_session
from ispec.agent.models import AgentCommand
from ispec.assistant.connect import get_assistant_session
from ispec.assistant.models import SupportMessage, SupportSession, SupportSessionReview


def utcnow() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True)
class SeededSupportSession:
    session_id: str
    session_pk: int
    user_message_id: int
    assistant_message_id: int


@dataclass(frozen=True)
class SmokeReviewResult:
    ok: bool
    session_id: str
    target_message_id: int
    review_id: int | None = None
    elapsed_seconds: float | None = None
    error: str | None = None


def generate_smoke_session_id(*, prefix: str = "smoke") -> str:
    safe_prefix = (prefix or "smoke").strip() or "smoke"
    token = uuid.uuid4().hex[:12]
    return f"{safe_prefix}-{token}"


def seed_support_session_for_review(
    *,
    session_id: str,
    user_message: str,
    assistant_message: str,
    user_id: int | None = None,
    assistant_db_path: str | Path | None = None,
) -> SeededSupportSession:
    """Insert a support session with a user+assistant turn needing review."""

    now = utcnow()
    with get_assistant_session(assistant_db_path) as db:
        session = (
            db.query(SupportSession)
            .filter(SupportSession.session_id == str(session_id))
            .first()
        )
        if session is None:
            session = SupportSession(session_id=str(session_id), user_id=user_id)
            db.add(session)
            db.flush()

        user_row = SupportMessage(
            session_pk=int(session.id),
            role="user",
            content=str(user_message),
            provider="smoke",
        )
        db.add(user_row)
        db.flush()

        assistant_row = SupportMessage(
            session_pk=int(session.id),
            role="assistant",
            content=str(assistant_message),
            provider="smoke",
        )
        db.add(assistant_row)

        session.updated_at = now
        db.flush()
        db.commit()

        return SeededSupportSession(
            session_id=str(session.session_id),
            session_pk=int(session.id),
            user_message_id=int(user_row.id),
            assistant_message_id=int(assistant_row.id),
        )


def latest_assistant_message_id(
    *,
    session_id: str,
    assistant_db_path: str | Path | None = None,
) -> int | None:
    with get_assistant_session(assistant_db_path) as db:
        session = (
            db.query(SupportSession)
            .filter(SupportSession.session_id == str(session_id))
            .first()
        )
        if session is None:
            return None
        last_assistant = (
            db.query(SupportMessage)
            .filter(SupportMessage.session_pk == session.id)
            .filter(SupportMessage.role == "assistant")
            .order_by(SupportMessage.id.desc())
            .first()
        )
        if last_assistant is None:
            return None
        return int(last_assistant.id)


def enqueue_orchestrator_tick(
    *,
    payload: dict | None = None,
    priority: int = 10,
    allow_existing: bool = True,
    agent_db_path: str | Path | None = None,
) -> int | None:
    """Enqueue an orchestrator tick for the supervisor loop to process."""

    now = utcnow()
    with get_agent_session(agent_db_path) as db:
        if allow_existing:
            existing = (
                db.query(AgentCommand.id)
                .filter(AgentCommand.command_type == COMMAND_ORCHESTRATOR_TICK)
                .filter(AgentCommand.status.in_(["queued", "running"]))
                .order_by(AgentCommand.id.asc())
                .first()
            )
            if existing is not None:
                return int(existing[0])

        row = AgentCommand(
            command_type=COMMAND_ORCHESTRATOR_TICK,
            status="queued",
            priority=int(priority),
            created_at=now,
            updated_at=now,
            available_at=now,
            attempts=0,
            max_attempts=1,
            payload_json=dict(payload or {}),
            result_json={},
        )
        db.add(row)
        db.commit()
        db.refresh(row)
        return int(row.id)


def wait_for_support_session_review(
    *,
    session_id: str,
    target_message_id: int,
    timeout_seconds: float = 60.0,
    poll_seconds: float = 1.0,
    assistant_db_path: str | Path | None = None,
) -> SmokeReviewResult:
    """Block until `support_session_review` exists for `(session_id, target_message_id)`."""

    session_id = str(session_id or "").strip()
    if not session_id:
        return SmokeReviewResult(
            ok=False,
            session_id=session_id,
            target_message_id=int(target_message_id or 0),
            error="missing_session_id",
        )

    target = int(target_message_id or 0)
    if target <= 0:
        return SmokeReviewResult(
            ok=False,
            session_id=session_id,
            target_message_id=target,
            error="missing_target_message_id",
        )

    started = time.monotonic()
    deadline = started + max(0.0, float(timeout_seconds))
    poll = max(0.1, float(poll_seconds))

    while True:
        with get_assistant_session(assistant_db_path) as db:
            session = (
                db.query(SupportSession)
                .filter(SupportSession.session_id == session_id)
                .first()
            )
            if session is None:
                return SmokeReviewResult(
                    ok=False,
                    session_id=session_id,
                    target_message_id=target,
                    elapsed_seconds=time.monotonic() - started,
                    error="session_not_found",
                )

            review = (
                db.query(SupportSessionReview)
                .filter(SupportSessionReview.session_pk == int(session.id))
                .filter(SupportSessionReview.target_message_id == int(target))
                .first()
            )
            if review is not None:
                return SmokeReviewResult(
                    ok=True,
                    session_id=session_id,
                    target_message_id=target,
                    review_id=int(review.id),
                    elapsed_seconds=time.monotonic() - started,
                )

        if time.monotonic() >= deadline:
            return SmokeReviewResult(
                ok=False,
                session_id=session_id,
                target_message_id=target,
                elapsed_seconds=time.monotonic() - started,
                error="timeout_waiting_for_review",
            )

        time.sleep(poll)

