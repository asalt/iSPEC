"""Personal browser chat history. Reading a thread never runs the assistant."""

from datetime import UTC, datetime
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ispec.api.security import require_user
from ispec.assistant.connect import get_assistant_session_dep
from ispec.assistant.models import SupportMessage, SupportSession
from ispec.db.models import AuthUser

router = APIRouter(prefix="/support/threads", tags=["support"])


class ThreadInfo(BaseModel):
    sessionId: str
    title: str
    updatedAt: datetime


class ThreadList(BaseModel):
    threads: list[ThreadInfo]
    nextOffset: int | None


class ChatTurn(BaseModel):
    id: int
    role: Literal["user", "assistant"]
    content: str
    createdAt: datetime
    feedback: Literal["up", "down"] | None


class Transcript(BaseModel):
    thread: ThreadInfo
    messages: list[ChatTurn]
    beforeId: int | None


def _threads_for(user: AuthUser):
    if not user.id or user.id <= 0:
        raise HTTPException(403, "Browser account required")
    user_turn = SupportMessage.role == "user"
    first_line = (
        select(func.substr(SupportMessage.content, 1, 100))
        .where(SupportMessage.session_pk == SupportSession.id, user_turn)
        .order_by(SupportMessage.id)
        .limit(1)
        .scalar_subquery()
    )
    return select(
        SupportSession.id,
        SupportSession.session_id,
        SupportSession.updated_at,
        first_line.label("first_line"),
    ).where(
        SupportSession.user_id == user.id,
        SupportSession.messages.any(user_turn),
        ~SupportSession.messages.any(SupportMessage.provider == "slack"),
    )


def _utc(value: datetime) -> datetime:
    # SQLite returns the assistant DB's UTC columns without their timezone.
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def _info(row) -> ThreadInfo:
    title = " ".join(row.first_line.split()) or "Untitled chat"
    return ThreadInfo(
        sessionId=row.session_id, title=title, updatedAt=_utc(row.updated_at)
    )


@router.get("", response_model=ThreadList)
def list_threads(
    response: Response,
    limit: int = Query(default=25, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    user: AuthUser = Depends(require_user),
    db: Session = Depends(get_assistant_session_dep),
):
    response.headers["Cache-Control"] = "private, no-store"
    query = _threads_for(user).order_by(
        SupportSession.updated_at.desc(), SupportSession.id.desc()
    )
    rows = db.execute(query.offset(offset).limit(limit + 1)).all()
    return ThreadList(
        threads=[_info(row) for row in rows[:limit]],
        nextOffset=offset + limit if len(rows) > limit else None,
    )


@router.get("/{session_id}/messages", response_model=Transcript)
def read_thread(
    session_id: str,
    response: Response,
    before: int | None = Query(default=None, ge=1),
    limit: int = Query(default=40, ge=1, le=100),
    user: AuthUser = Depends(require_user),
    db: Session = Depends(get_assistant_session_dep),
):
    response.headers["Cache-Control"] = "private, no-store"
    row = db.execute(
        _threads_for(user).where(SupportSession.session_id == session_id)
    ).one_or_none()
    if row is None:
        raise HTTPException(404, "Chat not found")
    query = select(
        SupportMessage.id,
        SupportMessage.role,
        SupportMessage.content,
        SupportMessage.created_at,
        SupportMessage.feedback,
    ).where(
        SupportMessage.session_pk == row.id,
        SupportMessage.role.in_(("user", "assistant")),
    )
    if before is not None:
        query = query.where(SupportMessage.id < before)
    rows = db.execute(query.order_by(SupportMessage.id.desc()).limit(limit + 1)).all()
    messages = [
        ChatTurn(
            id=m.id,
            role=m.role,
            content=m.content,
            createdAt=_utc(m.created_at),
            feedback={1: "up", -1: "down"}.get(m.feedback),
        )
        for m in reversed(rows[:limit])
    ]
    return Transcript(
        thread=_info(row),
        messages=messages,
        beforeId=messages[0].id if len(rows) > limit else None,
    )
