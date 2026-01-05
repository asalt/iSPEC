from __future__ import annotations

import os
import re
import json
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from ispec.api.security import require_access, require_assistant_access
from ispec.assistant.context import build_ispec_context, extract_project_ids
from ispec.assistant.connect import get_assistant_session_dep
from ispec.assistant.formatting import split_plan_final
from ispec.assistant.memory import update_state_from_message
from ispec.assistant.models import SupportMessage, SupportSession
from ispec.assistant.prompting import estimate_tokens_for_messages, summarize_messages
from ispec.assistant.service import _system_prompt, generate_reply
from ispec.assistant.tools import (
    TOOL_CALL_PREFIX,
    format_tool_result_message,
    parse_tool_call,
    run_tool,
)
from ispec.db.connect import get_session_dep
from ispec.db.models import AuthUser, UserRole
from ispec.schedule.connect import get_schedule_session_dep


router = APIRouter(prefix="/support", tags=["Support"])

_PROJECT_ROUTE_RE = re.compile(r"/project/(\d+)", re.IGNORECASE)
_EXPERIMENT_ROUTE_RE = re.compile(r"/experiment/(\d+)", re.IGNORECASE)
_EXPERIMENT_RUN_ROUTE_RE = re.compile(r"/experiment-run/(\d+)", re.IGNORECASE)
_CONTEXT_SCHEMA_VERSION = 1


def utcnow() -> datetime:
    return datetime.now(UTC)


class ChatHistoryItem(BaseModel):
    role: str
    content: str


class UIContext(BaseModel):
    name: str | None = None
    path: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    query: dict[str, Any] = Field(default_factory=dict)


class ChatRequest(BaseModel):
    sessionId: str = Field(min_length=1, max_length=256)
    message: str = Field(min_length=1, max_length=20_000)
    history: list[ChatHistoryItem] = Field(default_factory=list)
    ui: UIContext | None = None


class ChatResponse(BaseModel):
    sessionId: str
    messageId: int
    message: str


class FeedbackRequest(BaseModel):
    sessionId: str = Field(min_length=1, max_length=256)
    messageId: int = Field(ge=1)
    rating: str = Field(min_length=1, max_length=16)
    comment: str | None = Field(default=None, max_length=4000)
    ui: UIContext | None = None


class FeedbackItem(BaseModel):
    sessionId: str
    messageId: int
    rating: int
    note: str | None = None
    message: str
    createdAt: datetime
    feedbackAt: datetime | None = None
    assistant: dict[str, Any] | None = None
    feedbackMeta: dict[str, Any] | None = None


def _rating_value(rating: str) -> int:
    normalized = rating.strip().lower()
    if normalized in {"up", "thumbs_up", "1", "+1", "true", "yes", "y"}:
        return 1
    if normalized in {"down", "thumbs_down", "-1", "0", "false", "no", "n"}:
        return -1
    raise ValueError("rating must be 'up' or 'down'")


def _history_limit() -> int:
    raw = (os.getenv("ISPEC_ASSISTANT_HISTORY_LIMIT") or "").strip()
    if not raw:
        return 20
    try:
        return max(0, int(raw))
    except ValueError:
        return 20


def _max_prompt_tokens() -> int:
    raw = (os.getenv("ISPEC_ASSISTANT_MAX_PROMPT_TOKENS") or "").strip()
    if not raw:
        return 6000
    try:
        return max(256, int(raw))
    except ValueError:
        return 6000


def _summary_max_chars() -> int:
    raw = (os.getenv("ISPEC_ASSISTANT_SUMMARY_MAX_CHARS") or "").strip()
    if not raw:
        return 2000
    try:
        return max(0, int(raw))
    except ValueError:
        return 2000


def _max_tool_calls() -> int:
    raw = (os.getenv("ISPEC_ASSISTANT_MAX_TOOL_CALLS") or "").strip()
    if not raw:
        return 2
    try:
        return max(0, int(raw))
    except ValueError:
        return 2


def _load_state(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _dump_state(state: dict[str, Any]) -> str:
    return json.dumps(state, ensure_ascii=False, separators=(",", ":"))


def _context_message(*, payload: dict[str, Any]) -> str:
    version = payload.get("schema_version") or _CONTEXT_SCHEMA_VERSION
    return f"CONTEXT v{version} (read-only JSON):\n" + json.dumps(payload, ensure_ascii=False)


def _safe_int_from_state(value: Any) -> int | None:
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.isdigit():
            parsed = int(stripped)
            return parsed if parsed >= 0 else None
    return None


def _update_conversation_summary(
    *,
    assistant_db: Session,
    session_pk: int,
    state: dict[str, Any],
    summarize_through_id: int,
) -> tuple[dict[str, Any], bool]:
    """Update `state['conversation_summary']` up to `summarize_through_id` (inclusive)."""

    max_chars = _summary_max_chars()
    if max_chars <= 0 or summarize_through_id <= 0:
        return state, False

    last_done = _safe_int_from_state(state.get("conversation_summary_up_to_id")) or 0
    if summarize_through_id <= last_done:
        return state, False

    rows = (
        assistant_db.query(SupportMessage)
        .filter(SupportMessage.session_pk == session_pk)
        .filter(SupportMessage.id > last_done)
        .filter(SupportMessage.id <= summarize_through_id)
        .order_by(SupportMessage.id.asc())
        .all()
    )
    if not rows:
        state["conversation_summary_up_to_id"] = summarize_through_id
        return state, True

    chunk = summarize_messages(
        (
            {"role": row.role, "content": row.content}
            for row in rows
            if row.role in {"user", "assistant", "system"} and row.content
        ),
        max_chars=max_chars,
    )
    if not chunk:
        state["conversation_summary_up_to_id"] = summarize_through_id
        return state, True

    existing = state.get("conversation_summary")
    summary = existing.strip() if isinstance(existing, str) else ""
    summary = (summary + "\n" + chunk).strip() if summary else chunk
    if len(summary) > max_chars:
        summary = "…" + summary[-(max_chars - 1) :]

    state["conversation_summary"] = summary
    state["conversation_summary_up_to_id"] = summarize_through_id
    state["conversation_summary_updated_at"] = utcnow().isoformat()
    state["conversation_summary_version"] = 1
    return state, True


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.isdigit():
            parsed = int(stripped)
            return parsed if parsed >= 0 else None
    return None


def _extract_id_from_path(regex: re.Pattern[str], path: str | None) -> int | None:
    if not path:
        return None
    match = regex.search(path)
    if not match:
        return None
    return _safe_int(match.group(1))


@router.post("/chat", response_model=ChatResponse)
def chat(
    payload: ChatRequest,
    assistant_db: Session = Depends(get_assistant_session_dep),
    core_db: Session = Depends(get_session_dep),
    schedule_db: Session = Depends(get_schedule_session_dep),
    user: AuthUser | None = Depends(require_assistant_access),
):
    session = (
        assistant_db.query(SupportSession)
        .filter(SupportSession.session_id == payload.sessionId)
        .first()
    )
    if session is None:
        session = SupportSession(
            session_id=payload.sessionId,
            user_id=user.id if user is not None else None,
        )
        assistant_db.add(session)
        assistant_db.flush()
    elif user is not None and session.user_id is None:
        session.user_id = user.id

    state = _load_state(getattr(session, "state_json", None))
    state, state_changed = update_state_from_message(state, payload.message)

    ui_payload: dict[str, Any] | None = None
    ui_project_id: int | None = None
    ui_experiment_id: int | None = None
    ui_experiment_run_id: int | None = None
    if payload.ui is not None:
        ui_payload = payload.ui.model_dump()
        ui_route = {"name": payload.ui.name, "path": payload.ui.path}
        if state.get("ui_route") != ui_route:
            state["ui_route"] = ui_route
            state_changed = True

        ui_project_id = _safe_int((payload.ui.params or {}).get("id"))
        if ui_project_id is None:
            ui_project_id = _extract_id_from_path(_PROJECT_ROUTE_RE, payload.ui.path)

        ui_experiment_id = _safe_int((payload.ui.params or {}).get("id"))
        if ui_experiment_id is None:
            ui_experiment_id = _extract_id_from_path(_EXPERIMENT_ROUTE_RE, payload.ui.path)

        ui_experiment_run_id = _safe_int((payload.ui.params or {}).get("id"))
        if ui_experiment_run_id is None:
            ui_experiment_run_id = _extract_id_from_path(_EXPERIMENT_RUN_ROUTE_RE, payload.ui.path)

        if payload.ui.name == "ProjectDetail" and ui_project_id is not None:
            if state.get("ui_project_id") != ui_project_id:
                state["ui_project_id"] = ui_project_id
                state_changed = True
            if state.get("current_project_id") != ui_project_id:
                state["current_project_id"] = ui_project_id
                state_changed = True
        if payload.ui.name == "ExperimentDetail" and ui_experiment_id is not None:
            if state.get("ui_experiment_id") != ui_experiment_id:
                state["ui_experiment_id"] = ui_experiment_id
                state_changed = True
        if payload.ui.name == "ExperimentRunDetail" and ui_experiment_run_id is not None:
            if state.get("ui_experiment_run_id") != ui_experiment_run_id:
                state["ui_experiment_run_id"] = ui_experiment_run_id
                state_changed = True

    referenced_projects = extract_project_ids(payload.message)
    focused_project_id: int | None = None
    if referenced_projects:
        focused_project_id = referenced_projects[0]
        if state.get("current_project_id") != focused_project_id:
            state["current_project_id"] = focused_project_id
            state_changed = True
    else:
        candidate = state.get("current_project_id")
        if isinstance(candidate, int) and candidate >= 0:
            focused_project_id = candidate

    user_message = SupportMessage(
        session_pk=session.id,
        role="user",
        content=payload.message,
        provider="frontend",
    )
    assistant_db.add(user_message)
    session.updated_at = utcnow()
    assistant_db.flush()

    history_limit = _history_limit()
    max_tokens = _max_prompt_tokens()

    selected_history: list[SupportMessage] = []
    context_message = ""

    # Iterate twice to account for summary growth impacting the budget.
    for _ in range(2):
        summary_up_to_id = _safe_int_from_state(state.get("conversation_summary_up_to_id")) or 0

        candidate_rows = (
            assistant_db.query(SupportMessage)
            .filter(SupportMessage.session_pk == session.id)
            .order_by(SupportMessage.id.desc())
            .limit((history_limit + 1) if history_limit else 0)
            .all()
        )
        candidate_rows.reverse()
        history_rows = [
            row
            for row in candidate_rows
            if row.id != user_message.id
            and row.id > summary_up_to_id
            and row.role in {"user", "assistant", "system"}
            and row.content
        ]

        ispec_context = build_ispec_context(core_db, message=payload.message, state=state)
        context_payload: dict[str, Any] = {
            "schema_version": _CONTEXT_SCHEMA_VERSION,
            "session": {"id": session.session_id, "state": state},
            "user": {
                "username": user.username,
                "role": str(user.role),
            }
            if user is not None
            else None,
            "ui": ui_payload,
            "ispec": ispec_context,
        }
        context_message = _context_message(payload=context_payload)

        base_messages = [
            {"role": "system", "content": _system_prompt()},
            {"role": "system", "content": context_message},
            {"role": "user", "content": payload.message},
        ]
        tokens = estimate_tokens_for_messages(base_messages)

        selected_rev: list[SupportMessage] = []
        for row in reversed(history_rows):
            message_tokens = estimate_tokens_for_messages([{"role": row.role, "content": row.content}])
            if tokens + message_tokens > max_tokens:
                break
            tokens += message_tokens
            selected_rev.append(row)
        selected_history = list(reversed(selected_rev))

        summarize_through_id = (
            int(selected_history[0].id) - 1 if selected_history else (int(user_message.id) - 1)
        )
        if summarize_through_id < 0:
            summarize_through_id = 0

        state, summary_changed = _update_conversation_summary(
            assistant_db=assistant_db,
            session_pk=session.id,
            state=state,
            summarize_through_id=summarize_through_id,
        )
        if summary_changed:
            state_changed = True
            continue
        break

    if state_changed:
        session.state_json = _dump_state(state)

    history_payload = [
        {"role": row.role, "content": row.content}
        for row in selected_history
        if row.role in {"user", "assistant", "system"} and row.content
    ]

    tool_calls: list[dict[str, Any]] = []
    tool_messages: list[dict[str, str]] = []
    reply = None

    max_tool_calls = _max_tool_calls()
    used_tool_calls = 0
    while True:
        if tool_messages:
            base_messages = [
                {"role": "system", "content": _system_prompt()},
                {"role": "system", "content": context_message},
                *tool_messages,
                {"role": "user", "content": payload.message},
            ]
            tokens = estimate_tokens_for_messages(base_messages)
            trimmed_rev: list[dict[str, str]] = []
            for item in reversed(history_payload):
                message_tokens = estimate_tokens_for_messages([item])
                if tokens + message_tokens > max_tokens:
                    break
                tokens += message_tokens
                trimmed_rev.append(item)
            history_for_llm = list(reversed(trimmed_rev)) + tool_messages
        else:
            history_for_llm = history_payload

        reply = generate_reply(
            message=payload.message,
            history=history_for_llm,
            context=context_message,
        )

        tool_call = parse_tool_call(reply.content)
        if tool_call is None:
            break

        if used_tool_calls >= max_tool_calls:
            reply = None
            break

        used_tool_calls += 1
        tool_name, tool_args = tool_call
        tool_payload = run_tool(
            name=tool_name,
            args=tool_args,
            core_db=core_db,
            schedule_db=schedule_db,
            user=user,
        )
        tool_calls.append(
            {
                "name": tool_name,
                "arguments": tool_args,
                "ok": bool(tool_payload.get("ok")),
                "error": tool_payload.get("error"),
            }
        )
        tool_messages.extend(
            [
                {"role": "assistant", "content": reply.content.strip()},
                {"role": "system", "content": format_tool_result_message(tool_name, tool_payload)},
            ]
        )

    if reply is None:
        reply = generate_reply(
            message=payload.message,
            history=history_payload
            + [{"role": "system", "content": f"{TOOL_CALL_PREFIX} limit exceeded; answer without tools."}],
            context=context_message,
        )

    raw_reply_content = reply.content or ""
    plan_text, assistant_content = split_plan_final(raw_reply_content)
    if not assistant_content.strip():
        assistant_content = raw_reply_content.strip()

    meta: dict[str, Any] = {
        "provider": reply.provider,
        "model": reply.model,
        "references": {
            "projects": referenced_projects
            if referenced_projects
            else ([focused_project_id] if focused_project_id is not None else []),
        },
    }
    if reply.meta:
        meta["provider_meta"] = reply.meta
    if tool_calls:
        meta["tool_calls"] = tool_calls
    if plan_text:
        meta["plan"] = plan_text
    if raw_reply_content.strip() and raw_reply_content.strip() != assistant_content.strip():
        meta["raw_content"] = raw_reply_content.strip()

    assistant_message = SupportMessage(
        session_pk=session.id,
        role="assistant",
        content=assistant_content,
        provider=reply.provider,
        model=reply.model,
        meta_json=json.dumps(meta, ensure_ascii=False),
    )
    assistant_db.add(assistant_message)
    session.updated_at = utcnow()
    assistant_db.flush()

    return ChatResponse(
        sessionId=session.session_id,
        messageId=int(assistant_message.id),
        message=assistant_message.content,
    )


@router.post("/feedback")
def feedback(
    payload: FeedbackRequest,
    assistant_db: Session = Depends(get_assistant_session_dep),
    user: AuthUser | None = Depends(require_assistant_access),
):
    session = (
        assistant_db.query(SupportSession)
        .filter(SupportSession.session_id == payload.sessionId)
        .first()
    )
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found.")

    message_id = payload.messageId

    message = (
        assistant_db.query(SupportMessage)
        .filter(SupportMessage.id == message_id)
        .filter(SupportMessage.session_pk == session.id)
        .first()
    )
    if message is None:
        raise HTTPException(status_code=404, detail="Message not found.")

    try:
        value = _rating_value(payload.rating)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    message.feedback = value
    message.feedback_at = utcnow()

    note = (payload.comment or "").strip()
    message.feedback_note = note or None

    feedback_meta: dict[str, Any] = {}
    if payload.ui is not None:
        feedback_meta["ui"] = payload.ui.model_dump()
    if user is not None:
        feedback_meta["user"] = {"id": user.id, "username": user.username, "role": str(user.role)}
    message.feedback_meta_json = (
        json.dumps(feedback_meta, ensure_ascii=False) if feedback_meta else None
    )

    session.updated_at = utcnow()
    assistant_db.flush()

    return {"ok": True}


@router.get("/feedback", response_model=list[FeedbackItem])
def list_feedback(
    rating: str | None = None,
    session_id: str | None = Query(default=None, alias="sessionId"),
    limit: int = Query(default=200, ge=1, le=2000),
    offset: int = Query(default=0, ge=0),
    assistant_db: Session = Depends(get_assistant_session_dep),
    user: AuthUser | None = Depends(require_access),
):
    if user is not None and user.role != UserRole.admin:
        raise HTTPException(status_code=403, detail="Admin access required.")

    query = (
        assistant_db.query(SupportMessage, SupportSession)
        .join(SupportSession, SupportMessage.session_pk == SupportSession.id)
        .filter(SupportMessage.feedback.isnot(None))
    )
    if rating:
        try:
            value = _rating_value(rating)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        query = query.filter(SupportMessage.feedback == value)
    if session_id:
        query = query.filter(SupportSession.session_id == session_id)

    rows = (
        query.order_by(SupportMessage.feedback_at.desc(), SupportMessage.id.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )

    payload: list[FeedbackItem] = []
    for message, session in rows:
        meta = None
        raw_meta = getattr(message, "feedback_meta_json", None)
        if raw_meta:
            try:
                parsed = json.loads(raw_meta)
                meta = parsed if isinstance(parsed, dict) else None
            except Exception:
                meta = None

        payload.append(
            FeedbackItem(
                sessionId=session.session_id,
                messageId=int(message.id),
                rating=int(message.feedback or 0),
                note=getattr(message, "feedback_note", None),
                message=message.content,
                createdAt=message.created_at,
                feedbackAt=message.feedback_at,
                assistant={"provider": message.provider, "model": message.model},
                feedbackMeta=meta,
            )
        )

    return payload
