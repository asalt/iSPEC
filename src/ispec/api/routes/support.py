from __future__ import annotations

import os
import re
import json
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi import Request
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from ispec.api.security import require_access, require_assistant_access
from ispec.assistant.context import build_ispec_context, extract_project_ids
from ispec.assistant.connect import get_assistant_session_dep
from ispec.assistant.formatting import split_compare_finals, split_plan_final
from ispec.assistant.memory import update_state_from_message
from ispec.assistant.models import SupportMessage, SupportSession
from ispec.assistant.prompting import estimate_tokens_for_messages, summarize_messages
from ispec.assistant.service import (
    _system_prompt_answer,
    _system_prompt_planner,
    _system_prompt_review,
    _system_prompt_review_decider,
    generate_reply,
)
from ispec.assistant.tools import (
    TOOL_CALL_PREFIX,
    extract_tool_call_line,
    format_tool_result_message,
    openai_tools_for_user,
    parse_tool_call,
    run_tool,
)
from ispec.db.connect import get_session_dep
from ispec.db.models import AuthUser, UserRole
from ispec.omics.connect import get_omics_session_dep
from ispec.schedule.connect import get_schedule_session_dep


router = APIRouter(prefix="/support", tags=["Support"])

_PROJECT_ROUTE_RE = re.compile(r"/project/(\d+)", re.IGNORECASE)
_EXPERIMENT_ROUTE_RE = re.compile(r"/experiment/(\d+)", re.IGNORECASE)
_EXPERIMENT_RUN_ROUTE_RE = re.compile(r"/experiment-run/(\d+)", re.IGNORECASE)
_CONTEXT_SCHEMA_VERSION = 1
_EXPLICIT_TOOL_REQUEST_RE = re.compile(r"\b(use|call|run)\s+(a\s+)?tool\b", re.IGNORECASE)
_COUNT_PROJECTS_RE = re.compile(
    r"\bhow\s+many\s+projects?\b|\bnumber\s+of\s+projects?\b|\bcount\s+projects?\b",
    re.IGNORECASE,
)


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


class ChatCompareChoice(BaseModel):
    index: int = Field(ge=0, le=1)
    message: str


class ChatComparePayload(BaseModel):
    userMessageId: int = Field(ge=1)
    choices: list[ChatCompareChoice] = Field(min_length=2, max_length=2)


class ChatResponse(BaseModel):
    sessionId: str
    messageId: int | None = None
    message: str | None = None
    compare: ChatComparePayload | None = None


class ChooseResponse(ChatResponse):
    pass


class ChooseRequest(BaseModel):
    sessionId: str = Field(min_length=1, max_length=256)
    userMessageId: int = Field(ge=1)
    choiceIndex: int = Field(ge=0, le=1)
    ui: UIContext | None = None


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


def _is_truthy(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _self_review_enabled() -> bool:
    return _is_truthy(os.getenv("ISPEC_ASSISTANT_SELF_REVIEW"))


def _self_review_decider_enabled() -> bool:
    return _is_truthy(os.getenv("ISPEC_ASSISTANT_SELF_REVIEW_DECIDER"))


def _compare_mode_enabled() -> bool:
    return _is_truthy(os.getenv("ISPEC_ASSISTANT_COMPARE_MODE"))


def _decide_if_dualchoice(*, payload: ChatRequest) -> bool:
    # TODO: Implement heuristics for when compare mode is actually helpful.
    return True


def _assistant_provider() -> str:
    return (os.getenv("ISPEC_ASSISTANT_PROVIDER") or "stub").strip().lower()


def _tool_protocol() -> str:
    raw = (os.getenv("ISPEC_ASSISTANT_TOOL_PROTOCOL") or "").strip().lower()
    return raw if raw in {"line", "openai"} else "line"


def _openai_tool_choice_for_message(message: str) -> dict[str, Any] | None:
    if not _EXPLICIT_TOOL_REQUEST_RE.search(message or ""):
        return None

    project_ids = extract_project_ids(message)
    if project_ids:
        return {"type": "function", "function": {"name": "get_project"}}

    if _COUNT_PROJECTS_RE.search(message or ""):
        return {"type": "function", "function": {"name": "count_projects"}}

    return None


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


def _enforce_session_access(session: SupportSession, user: AuthUser | None) -> None:
    """Ensure the authenticated user is allowed to access this assistant session."""

    if user is None:
        if session.user_id is not None:
            raise HTTPException(status_code=403, detail="Session requires authentication.")
        return

    if session.user_id is None:
        session.user_id = user.id
        return

    if session.user_id != user.id:
        raise HTTPException(status_code=403, detail="Session belongs to another user.")


@router.post("/chat", response_model=ChatResponse)
def chat(
    payload: ChatRequest,
    request: Request = None,
    assistant_db: Session = Depends(get_assistant_session_dep),
    core_db: Session = Depends(get_session_dep),
    omics_db: Session = Depends(get_omics_session_dep),
    schedule_db: Session = Depends(get_schedule_session_dep),
    user: AuthUser | None = Depends(require_assistant_access),
):
    api_schema: dict[str, Any] | None = None
    if request is not None:
        try:
            api_schema = request.app.openapi()
        except Exception:
            api_schema = None

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
    else:
        _enforce_session_access(session, user)

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

    tool_protocol = _tool_protocol()
    max_tool_calls = _max_tool_calls()
    used_tool_calls = 0
    tools_enabled = max_tool_calls > 0
    tools_enabled_initial = tools_enabled
    tool_schemas = openai_tools_for_user(user) if tool_protocol == "openai" and tools_enabled else None
    tool_schemas_count = len(tool_schemas) if isinstance(tool_schemas, list) else 0

    compare_mode_requested = _compare_mode_enabled() and _decide_if_dualchoice(payload=payload)
    response_format = "compare" if compare_mode_requested else "single"

    tools_available = tool_schemas_count > 0 and tools_enabled
    planner_prompt = _system_prompt_planner(tools_available=tools_available, response_format=response_format)
    answer_prompt = _system_prompt_answer(response_format=response_format)
    review_prompt = _system_prompt_review()
    prompt_for_budget = planner_prompt if tools_enabled else answer_prompt

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
                "id": int(user.id),
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
            {"role": "system", "content": prompt_for_budget},
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
    tool_messages: list[dict[str, Any]] = []
    tool_result_messages: list[dict[str, Any]] = []
    reply = None
    forced_tool_choice = _openai_tool_choice_for_message(payload.message) if tool_protocol == "openai" else None

    while True:
        system_prompt = planner_prompt if tools_enabled else answer_prompt
        base_messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "system", "content": context_message},
            {"role": "user", "content": payload.message},
            *tool_messages,
        ]
        tokens = estimate_tokens_for_messages(base_messages)
        trimmed_rev: list[dict[str, Any]] = []
        for item in reversed(history_payload):
            message_tokens = estimate_tokens_for_messages([item])
            if tokens + message_tokens > max_tokens:
                break
            tokens += message_tokens
            trimmed_rev.append(item)
        trimmed_history = list(reversed(trimmed_rev))

        messages_for_llm: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "system", "content": context_message},
            *trimmed_history,
            {"role": "user", "content": payload.message},
            *tool_messages,
        ]

        tool_choice = forced_tool_choice if tool_schemas is not None and not tool_calls and used_tool_calls == 0 else None
        reply = generate_reply(
            messages=messages_for_llm,
            tools=tool_schemas if tools_enabled else None,
            tool_choice=tool_choice,
        )

        if reply.tool_calls:
            tool_messages.append(
                {
                    "role": "assistant",
                    "content": reply.content or "",
                    "tool_calls": reply.tool_calls,
                }
            )
            for tool_call in reply.tool_calls:
                if not isinstance(tool_call, dict):
                    continue
                call_id = str(tool_call.get("id") or tool_call.get("tool_call_id") or "")
                if not call_id:
                    call_id = f"call_{used_tool_calls + 1}"
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

                if used_tool_calls >= max_tool_calls:
                    tool_payload = {
                        "ok": False,
                        "tool": tool_name or None,
                        "error": "Tool call limit exceeded; no further tools executed.",
                    }
                    tools_enabled = False
                else:
                    used_tool_calls += 1
                    tool_payload = run_tool(
                        name=tool_name,
                        args=parsed_args if isinstance(parsed_args, dict) else {},
                        core_db=core_db,
                        schedule_db=schedule_db,
                        omics_db=omics_db,
                        user=user,
                        api_schema=api_schema,
                        user_message=payload.message,
                    )
                    if used_tool_calls >= max_tool_calls:
                        tools_enabled = False

                tool_calls.append(
                    {
                        "name": tool_name,
                        "arguments": parsed_args if isinstance(parsed_args, dict) else {},
                        "ok": bool(tool_payload.get("ok")),
                        "error": tool_payload.get("error"),
                        "result_preview": _tool_result_preview(tool_payload),
                        "protocol": "openai",
                    }
                )
                tool_result_text = format_tool_result_message(tool_name, tool_payload)
                tool_result_messages.append({"role": "system", "content": tool_result_text})
                tool_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": json.dumps(tool_payload, ensure_ascii=False),
                    }
                )
                tool_messages.append({"role": "system", "content": tool_result_text})
            continue

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
            omics_db=omics_db,
            user=user,
            api_schema=api_schema,
            user_message=payload.message,
        )
        tool_calls.append(
            {
                "name": tool_name,
                "arguments": tool_args,
                "ok": bool(tool_payload.get("ok")),
                "error": tool_payload.get("error"),
                "result_preview": _tool_result_preview(tool_payload),
                "protocol": "line",
            }
        )
        tool_call_line = extract_tool_call_line(reply.content) or reply.content.strip()
        tool_result_messages.append(
            {"role": "system", "content": format_tool_result_message(tool_name, tool_payload)}
        )
        tool_messages.extend(
            [
                {"role": "assistant", "content": tool_call_line},
                {"role": "system", "content": format_tool_result_message(tool_name, tool_payload)},
            ]
        )

    if reply is None:
        reply = generate_reply(
            messages=[
                {"role": "system", "content": answer_prompt},
                {"role": "system", "content": context_message},
                *history_payload,
                {"role": "user", "content": payload.message},
                *tool_messages,
                {"role": "system", "content": f"{TOOL_CALL_PREFIX} limit exceeded; answer without tools."},
            ],
            tools=None,
        )

    compare_choices: tuple[str, str] | None = None

    # First-pass reply (after any tool calls).
    draft_raw_reply_content = reply.content or ""
    if compare_mode_requested:
        compare_choices = split_compare_finals(draft_raw_reply_content)

    if compare_choices is not None:
        draft_plan_text = None
        draft_assistant_content = compare_choices[0]
    else:
        draft_plan_text, draft_assistant_content = split_plan_final(draft_raw_reply_content)
        if not draft_assistant_content.strip():
            draft_assistant_content = draft_raw_reply_content.strip()

    plan_text = draft_plan_text
    assistant_content = draft_assistant_content
    final_reply = reply
    final_raw_reply_content = draft_raw_reply_content

    self_review_changed = False
    self_review_error: str | None = None
    self_review_mode: str | None = None
    self_review_decision: str | None = None
    if _self_review_enabled() and compare_choices is None and final_reply.ok and assistant_content.strip():
        self_review_mode = "rewrite"

        if _assistant_provider() == "vllm" and _self_review_decider_enabled():
            self_review_mode = "guided_choice"
            decider_instruction = (
                "Decide if the draft answer needs changes. "
                "Output only KEEP (no changes) or REWRITE (needs changes)."
            )
            decider_reply = generate_reply(
                messages=[
                    {"role": "system", "content": _system_prompt_review_decider()},
                    {"role": "system", "content": context_message},
                    {"role": "user", "content": payload.message},
                    *tool_messages,
                    {"role": "assistant", "content": assistant_content},
                    {"role": "user", "content": decider_instruction},
                ],
                tools=None,
                vllm_extra_body={"guided_choice": ["KEEP", "REWRITE"], "max_tokens": 1, "temperature": 0},
            )
            if not decider_reply.ok:
                self_review_error = "review_decider_error"
                self_review_decision = "keep"
            else:
                decider_text = (decider_reply.content or "").strip().upper()
                if decider_text.startswith("KEEP"):
                    self_review_decision = "keep"
                elif decider_text.startswith("REWRITE"):
                    self_review_decision = "rewrite"
                else:
                    self_review_error = "review_decider_invalid_output"

        if self_review_decision != "keep":
            review_instruction = (
                "Review the assistant answer above for correctness (grounded in CONTEXT/TOOL_RESULT), "
                "clarity, and iSPEC tone. If it's already good, repeat it verbatim. "
                "If it needs changes, rewrite it. Do not call tools.\n"
                "Output only:\nFINAL:\n<answer>"
            )
            review_reply = generate_reply(
                messages=[
                    {"role": "system", "content": review_prompt},
                    {"role": "system", "content": context_message},
                    {"role": "user", "content": payload.message},
                    *tool_messages,
                    {"role": "assistant", "content": assistant_content},
                    {"role": "user", "content": review_instruction},
                ],
                tools=None,
            )

            if not review_reply.ok:
                self_review_error = "review_error"
            else:
                review_tool_call = parse_tool_call(review_reply.content)
                if review_tool_call is not None or review_reply.tool_calls:
                    self_review_error = "review_requested_tool_call"
                else:
                    review_raw = (review_reply.content or "").strip()
                    review_plan_text, review_final_content = split_plan_final(review_raw)
                    reviewed_content = (review_final_content or review_raw).strip()
                    if reviewed_content:
                        final_reply = reply
                        final_raw_reply_content = draft_raw_reply_content
                        assistant_content = draft_assistant_content

                        if reviewed_content != assistant_content.strip():
                            self_review_changed = True
                            assistant_content = reviewed_content
                            final_reply = review_reply
                            final_raw_reply_content = review_raw
                    else:
                        self_review_error = "empty_review_output"

    meta: dict[str, Any] = {
        "provider": final_reply.provider,
        "model": final_reply.model,
        "references": {
            "projects": referenced_projects
            if referenced_projects
            else ([focused_project_id] if focused_project_id is not None else []),
        },
    }
    meta["tooling"] = {
        "enabled": bool(tools_enabled_initial),
        "protocol_config": tool_protocol,
        "schemas_provided": tool_schemas_count > 0,
        "schemas_count": tool_schemas_count,
        "max_tool_calls": max_tool_calls,
        "used_tool_calls": used_tool_calls,
        "forced_tool_choice": forced_tool_choice,
    }
    if final_reply.meta:
        meta["provider_meta"] = final_reply.meta
    meta["tool_calls"] = tool_calls
    if plan_text:
        meta["plan"] = plan_text
    raw_final = final_raw_reply_content.strip()
    if raw_final and raw_final != assistant_content.strip():
        meta["raw_content"] = raw_final
    if _self_review_enabled():
        meta["self_review"] = {
            "changed": self_review_changed,
            "error": self_review_error,
            "mode": self_review_mode,
            "decision": self_review_decision,
        }
        if self_review_changed:
            draft_raw = draft_raw_reply_content.strip()
            if draft_raw and draft_raw != raw_final:
                meta["draft_raw_content"] = draft_raw

    if compare_choices is not None and final_reply.ok and assistant_content.strip():
        answer_a, answer_b = compare_choices
        meta_a = {**meta, "compare_choice": {"index": 0}}
        meta_b = {**meta, "compare_choice": {"index": 1}}
        compare_record: dict[str, Any] = {
            "schema_version": 1,
            "created_at": utcnow().isoformat(),
            "choices": [
                {
                    "index": 0,
                    "message": answer_a,
                    "provider": final_reply.provider,
                    "model": final_reply.model,
                    "meta": meta_a,
                },
                {
                    "index": 1,
                    "message": answer_b,
                    "provider": final_reply.provider,
                    "model": final_reply.model,
                    "meta": meta_b,
                },
            ],
            "selected_index": None,
            "selected_message_id": None,
        }

        user_meta: dict[str, Any] = {"compare": compare_record}
        if ui_payload is not None:
            user_meta["ui"] = ui_payload
        if user is not None:
            user_meta["user"] = {"id": int(user.id), "username": user.username, "role": str(user.role)}
        user_message.meta_json = json.dumps(user_meta, ensure_ascii=False)
        session.updated_at = utcnow()
        assistant_db.flush()

        return ChatResponse(
            sessionId=session.session_id,
            compare=ChatComparePayload(
                userMessageId=int(user_message.id),
                choices=[
                    ChatCompareChoice(index=0, message=answer_a),
                    ChatCompareChoice(index=1, message=answer_b),
                ],
            ),
        )

    assistant_message = SupportMessage(
        session_pk=session.id,
        role="assistant",
        content=assistant_content,
        provider=final_reply.provider,
        model=final_reply.model,
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


@router.post("/choose", response_model=ChooseResponse)
def choose(
    payload: ChooseRequest,
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
    _enforce_session_access(session, user)

    user_message = (
        assistant_db.query(SupportMessage)
        .filter(SupportMessage.session_pk == session.id)
        .filter(SupportMessage.id == int(payload.userMessageId))
        .first()
    )
    if user_message is None:
        raise HTTPException(status_code=404, detail="Message not found.")
    if user_message.role != "user":
        raise HTTPException(status_code=400, detail="Only user messages can be chosen.")

    meta_raw = getattr(user_message, "meta_json", None)
    compare_meta = _load_state(meta_raw).get("compare")
    if not isinstance(compare_meta, dict):
        raise HTTPException(status_code=400, detail="No compare choices found for this message.")

    selected_message_id = compare_meta.get("selected_message_id")
    if isinstance(selected_message_id, int) and selected_message_id > 0:
        assistant_message = (
            assistant_db.query(SupportMessage)
            .filter(SupportMessage.session_pk == session.id)
            .filter(SupportMessage.id == selected_message_id)
            .first()
        )
        if assistant_message is not None:
            return ChooseResponse(
                sessionId=session.session_id,
                messageId=int(assistant_message.id),
                message=assistant_message.content,
            )

    raw_choices = compare_meta.get("choices")
    if not isinstance(raw_choices, list) or not raw_choices:
        raise HTTPException(status_code=400, detail="Compare choices are missing.")

    chosen: dict[str, Any] | None = None
    for item in raw_choices:
        if not isinstance(item, dict):
            continue
        if item.get("index") == payload.choiceIndex:
            chosen = item
            break
    if chosen is None:
        raise HTTPException(status_code=400, detail="Invalid choiceIndex.")

    chosen_message = str(chosen.get("message") or "").strip()
    if not chosen_message:
        raise HTTPException(status_code=400, detail="Chosen message is empty.")

    assistant_meta = chosen.get("meta") if isinstance(chosen.get("meta"), dict) else {}
    if not isinstance(assistant_meta, dict):
        assistant_meta = {}
    compare_selection: dict[str, Any] = {
        "user_message_id": int(user_message.id),
        "choice_index": int(payload.choiceIndex),
    }
    if payload.ui is not None:
        compare_selection["ui"] = payload.ui.model_dump()
    if user is not None:
        compare_selection["user"] = {"id": int(user.id), "username": user.username, "role": str(user.role)}
    assistant_meta = {**assistant_meta, "compare_selection": compare_selection}

    assistant_message = SupportMessage(
        session_pk=session.id,
        role="assistant",
        content=chosen_message,
        provider=str(chosen.get("provider") or "") or None,
        model=str(chosen.get("model") or "") or None,
        meta_json=json.dumps(assistant_meta, ensure_ascii=False) if assistant_meta else None,
    )
    assistant_db.add(assistant_message)
    session.updated_at = utcnow()
    assistant_db.flush()

    compare_meta["selected_index"] = int(payload.choiceIndex)
    compare_meta["selected_message_id"] = int(assistant_message.id)
    compare_meta["selected_at"] = utcnow().isoformat()
    user_message.meta_json = json.dumps({**_load_state(meta_raw), "compare": compare_meta}, ensure_ascii=False)
    assistant_db.flush()

    return ChooseResponse(
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
    _enforce_session_access(session, user)

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
