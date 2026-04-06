from __future__ import annotations

from functools import cache
import os
import re
import json
import time
from datetime import UTC, datetime, timedelta
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from pydantic import BaseModel, Field
from sqlalchemy import func
from sqlalchemy.orm import Session

from ispec.api.security import require_access, require_assistant_access
from ispec.agent.connect import get_agent_session_dep
from ispec.agent.commands import COMMAND_COMPACT_SESSION_MEMORY, COMMAND_ORCHESTRATOR_TICK, COMMAND_SUPPORT_CHAT_TURN
from ispec.agent.models import AgentCommand, AgentRun
from ispec.assistant.comment_intent import decide_project_comment_intent_vllm
from ispec.assistant.context import build_ispec_context, extract_project_ids
from ispec.assistant.compaction import normalize_conversation_memory
from ispec.assistant.connect import get_assistant_session_dep
from ispec.assistant.controller import (
    enqueue_post_send_prepare_command,
    run_message_pre_send_controller,
    support_post_send_thread_key,
)
from ispec.assistant.formatting import split_compare_finals, split_plan_final
from ispec.assistant.memory import update_state_from_message
from ispec.assistant.models import SupportMessage, SupportSession
from ispec.assistant.prompt_header import build_prompt_header, prompt_header_enabled
from ispec.assistant.prompting import estimate_tokens_for_messages, summarize_messages
from ispec.assistant.response_contracts import (
    response_contracts_mode,
    response_contract_names,
)
from ispec.assistant.service import (
    _system_prompt_answer,
    _system_prompt_planner,
    generate_reply,
)
from ispec.assistant.support_policies import (
    comment_intent_messages_for_write_mode,
    hinted_support_tool_groups,
    select_support_tool_policy,
)
from ispec.assistant.tool_routing import (
    route_tool_groups_vllm,
    tool_groups_for_available_tools,
    tool_names_for_groups,
)
from ispec.assistant.turn_decision import (
    parse_turn_decision_mode,
    run_turn_decision_pipeline,
    selected_tool_names_from_decision,
)
from ispec.assistant.tools import (
    TOOL_CALL_PREFIX,
    extract_tool_call_line,
    format_tool_result_message,
    openai_tools_for_user,
    openai_tool_names_all,
    parse_tool_call,
    run_tool,
    tool_writes_data,
)
from ispec.db.connect import get_session_dep
from ispec.db.models import AuthUser, UserRole
from ispec.omics.connect import get_omics_session_dep
from ispec.schedule.connect import get_schedule_session_dep


router = APIRouter(prefix="/support", tags=["Support"])

# we will be careful about these "hard coding of keywords" and try not
# to be too reliant on it for functionality
_PROJECT_ROUTE_RE = re.compile(r"/project/(\d+)", re.IGNORECASE)
_EXPERIMENT_ROUTE_RE = re.compile(r"/experiment/(\d+)", re.IGNORECASE)
_EXPERIMENT_RUN_ROUTE_RE = re.compile(r"/experiment-run/(\d+)", re.IGNORECASE)
_CONTEXT_SCHEMA_VERSION = 1
_PROJECT_COMMENT_INTENT_RE = re.compile(
    r"\b(comment|note|history|save|draft|rewrite|edit|word|wording|log|record|add|commit)\b",
    re.IGNORECASE,
)
_WRITE_SUCCESS_VERB_RE = re.compile(
    r"\b(saved|added|logged|recorded|created|written|updated|committed)\b",
    re.IGNORECASE,
)
_WRITE_SUCCESS_NOTE_RE = re.compile(
    r"\b(?:i|we)\s+(?:have|ve)\s+(?:made|added|saved|logged|recorded|created)\s+(?:a|an|the)?\s*(?:note|comment|entry|memo)\b",
    re.IGNORECASE,
)
_WRITE_SUCCESS_FIRST_PERSON_RE = re.compile(
    r"\b(?:i|we)\s+(?:(?:have|ve)\s+)?(?:saved|added|logged|recorded|created|written|updated|committed)\b"
    r".*\b(?:note|comment|history|memo|entry)\b",
    re.IGNORECASE,
)
_WRITE_SUCCESS_PASSIVE_RE = re.compile(
    r"\b(?:note|comment|history|entry|memo)\b.*\b(?:has|have|is|was)\s+(?:been\s+)?\b(saved|added|logged|recorded|created|written|updated|committed)\b"
    r"|\bsuccessfully\b.*\b(saved|added|logged|recorded|created|written|updated|committed)\b",
    re.IGNORECASE,
)
_WRITE_RESULT_ID_RE = re.compile(r"\bcomment\s+id\b", re.IGNORECASE)
_TRUTHY_ENV = {"1", "true", "yes", "y", "on"}
_FALSY_ENV = {"0", "false", "no", "n", "off"}
_REPLY_INTERPRETATION_ACTIONS: dict[str, dict[str, str]] = {
    "project_comment_save_confirmation": {
        "approve": "approve_save",
        "deny": "deny_save",
        "defer": "defer_save",
        "modify": "modify_before_save",
        "unclear": "clarify",
    }
}
_REPLY_INTERPRETATION_POLICY_TEXT: dict[str, str] = {
    "deny_save": (
        "The user declined the pending save. Do not save anything. "
        "Answer briefly that nothing was saved."
    ),
    "defer_save": (
        "The user wants to wait before saving. Do not save anything. "
        "Answer briefly that nothing was saved and they can come back later."
    ),
    "modify_before_save": (
        "The user wants to change the pending draft before saving. Do not save anything. "
        "Ask what should be revised or invite them to provide the updated wording."
    ),
    "clarify": (
        "The turn is awaiting a bounded save confirmation, but the latest reply is not safely actionable. "
        "Do not save anything. Ask a brief clarification question about whether to save, revise, or cancel."
    ),
}


@cache
def _truthy_value(raw: str) -> bool:
    return raw.strip().lower() in _TRUTHY_ENV


def _env_truthy(key: str) -> bool:
    return _truthy_value(os.getenv(key) or "")


def _env_tristate_bool(key: str) -> bool | None:
    raw = os.getenv(key)
    if raw is None:
        return None
    text = str(raw).strip().lower()
    if not text or text == "auto":
        return None
    if text in _TRUTHY_ENV:
        return True
    if text in _FALSY_ENV:
        return False
    # Invalid values are treated as "auto" for safety/forward compatibility.
    return None


def _state_dir_is_dev() -> bool:
    raw = (os.getenv("ISPEC_STATE_DIR") or "").strip()
    if not raw:
        return False
    try:
        from pathlib import Path

        path = Path(raw).expanduser().resolve()
    except Exception:
        return False
    return path.name == ".pids"


@cache
def _parse_int_setting(raw: str, default: int, minimum: int) -> int:
    if not raw:
        return default
    try:
        return max(minimum, int(raw))
    except ValueError:
        return default


def _turn_decision_mode() -> str:
    return parse_turn_decision_mode(
        os.getenv("ISPEC_ASSISTANT_TURN_DECISION_MODE"),
        auto_shadow=_state_dir_is_dev() and _assistant_provider() == "vllm",
    )


def _legacy_comment_intent_to_write_mode(decision: dict[str, Any] | None) -> str | None:
    if not isinstance(decision, dict):
        return None
    intent = str(decision.get("intent") or "").strip()
    if intent == "draft_only":
        return "draft_only"
    if intent == "save_now":
        return "save_now"
    if intent == "confirm_save":
        return "confirm_save"
    if intent == "other":
        return "none"
    return None


def _merge_response_contract_request_meta(
    *,
    request_meta: dict[str, Any] | None,
    contract_cap: str | None,
) -> dict[str, Any] | None:
    if not isinstance(request_meta, dict) and not contract_cap:
        return request_meta

    merged = dict(request_meta) if isinstance(request_meta, dict) else {}
    if not contract_cap:
        return merged

    raw_policy = merged.get("response_contract")
    policy = dict(raw_policy) if isinstance(raw_policy, dict) else {}
    if not str(policy.get("force") or "").strip() and not str(policy.get("cap") or "").strip():
        policy["cap"] = contract_cap
    if policy:
        merged["response_contract"] = policy
    return merged


def _response_contract_policy_meta(request_meta: dict[str, Any] | None) -> dict[str, str | None]:
    raw_policy = request_meta.get("response_contract") if isinstance(request_meta, dict) else None
    if not isinstance(raw_policy, dict):
        return {"force": None, "cap": None}
    force = str(raw_policy.get("force") or "").strip() or None
    cap = str(raw_policy.get("cap") or "").strip() or None
    return {"force": force, "cap": cap}


def _response_contract_live_eligibility(
    *,
    tool_calls: list[dict[str, Any]],
    comment_intent_decision: dict[str, Any] | None,
    turn_decision_result: Any,
) -> tuple[bool, str | None]:
    for call in tool_calls:
        if not isinstance(call, dict):
            continue
        tool_name = str(call.get("name") or "").strip()
        if tool_name == "create_project_comment":
            return False, "project_comment_tool_call"

    decision = getattr(turn_decision_result, "decision", None)
    primary_goal = str(getattr(decision, "primary_goal", "") or "").strip()
    if primary_goal in {"draft_project_comment", "save_project_comment"}:
        return False, f"turn_decision_primary_goal:{primary_goal}"

    write_plan = getattr(decision, "write_plan", None)
    write_mode = str(getattr(write_plan, "mode", "") or "").strip()
    if write_mode in {"draft_only", "save_now", "confirm_save"}:
        return False, f"turn_decision_write_mode:{write_mode}"

    comment_write_mode = _legacy_comment_intent_to_write_mode(comment_intent_decision)
    if comment_write_mode in {"draft_only", "save_now", "confirm_save"}:
        return False, f"comment_intent_write_mode:{comment_write_mode}"

    return True, None


def _policy_tool_for_message(message: str) -> str | None:
    selection = select_support_tool_policy(message=message)
    return selection.tool_name if selection is not None else None


def _policy_tool_args_for_message(*, tool_name: str | None, message: str) -> dict[str, Any]:
    selection = select_support_tool_policy(message=message)
    if selection is None or selection.tool_name != tool_name:
        return {}
    return dict(selection.args or {})


def _policy_messages_for_message(*, tool_name: str | None, message: str) -> list[dict[str, str]]:
    selection = select_support_tool_policy(message=message)
    if selection is None or selection.tool_name != tool_name:
        return []
    return list(selection.messages or [])


def _hinted_tool_groups_for_message(*, message: str, focused_project_id: int | None) -> set[str]:
    return hinted_support_tool_groups(message=message, focused_project_id=focused_project_id)


def _truncate(value: str | None, limit: int = 400) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if limit <= 0 or len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def _openai_tool_names(tools: list[dict[str, Any]] | None) -> set[str]:
    names: set[str] = set()
    if not tools:
        return names
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        func_obj = tool.get("function")
        if not isinstance(func_obj, dict):
            continue
        name = func_obj.get("name")
        if isinstance(name, str) and name.strip():
            names.add(name.strip())
    return names


def _openai_tool_required_keys(tool: dict[str, Any]) -> set[str]:
    func_obj = tool.get("function")
    if not isinstance(func_obj, dict):
        return set()
    params = func_obj.get("parameters")
    if not isinstance(params, dict):
        return set()
    required = params.get("required")
    if not isinstance(required, list):
        return set()
    required_keys: set[str] = set()
    for item in required:
        if isinstance(item, str) and item.strip():
            required_keys.add(item.strip())
    return required_keys


def _openai_no_arg_tool_names(tools: list[dict[str, Any]] | None) -> set[str]:
    names: set[str] = set()
    if not tools:
        return names
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        func_obj = tool.get("function")
        if not isinstance(func_obj, dict):
            continue
        name = func_obj.get("name")
        if not isinstance(name, str) or not name.strip():
            continue
        if _openai_tool_required_keys(tool):
            continue
        names.add(name.strip())
    return names


def _suggested_tool_from_text(text: str | None, *, candidates: set[str]) -> str | None:
    if not text or not candidates:
        return None
    found: list[str] = []
    for name in sorted(candidates):
        if re.search(rf"\b{re.escape(name)}\b", text):
            found.append(name)
            if len(found) > 2:
                break
    if len(found) == 1:
        return found[0]
    return None


def _requested_tool_names_from_text(
    text: str | None,
    *,
    candidates: set[str],
    max_items: int = 5,
) -> list[str]:
    """Return tool names (from candidates) that appear verbatim in text."""

    if not text or not candidates:
        return []
    found: list[str] = []
    for name in sorted(candidates):
        if not name:
            continue
        if re.search(rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])", text):
            found.append(name)
            if len(found) >= max(1, int(max_items)):
                break
    return found


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
    meta: dict[str, Any] | None = None


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


class SupportDebugMessageItem(BaseModel):
    sessionId: str
    sessionUser: dict[str, Any] | None = None
    messageId: int
    role: str
    message: str
    createdAt: datetime
    provider: str | None = None
    model: str | None = None
    uiPath: str | None = None
    uiName: str | None = None
    previousUserMessageId: int | None = None
    previousUserMessage: str | None = None
    previousUserMeta: dict[str, Any] | None = None
    messageMeta: dict[str, Any] | None = None
    toolRouter: dict[str, Any] | None = None
    tooling: dict[str, Any] | None = None
    toolCalls: list[dict[str, Any]] | None = None
    llmTrace: list[dict[str, Any]] | None = None
    promptHeader: dict[str, Any] | None = None


class SupportDebugContextItem(BaseModel):
    messageId: int
    role: str
    message: str
    createdAt: datetime
    provider: str | None = None
    model: str | None = None
    isSelected: bool = False


class SupportDebugMessageDetail(SupportDebugMessageItem):
    sessionState: dict[str, Any] | None = None
    sessionMessageCount: int | None = None
    contextMessages: list[SupportDebugContextItem] = Field(default_factory=list)


def _rating_value(rating: str) -> int:
    normalized = rating.strip().lower()
    if normalized in {"up", "thumbs_up", "1", "+1", "true", "yes", "y"}:
        return 1
    if normalized in {"down", "thumbs_down", "-1", "0", "false", "no", "n"}:
        return -1
    raise ValueError("rating must be 'up' or 'down'")


def _history_limit() -> int:
    raw = (os.getenv("ISPEC_ASSISTANT_HISTORY_LIMIT") or "").strip()
    return _parse_int_setting(raw, 10, 0)


def _max_prompt_tokens() -> int:
    raw = (os.getenv("ISPEC_ASSISTANT_MAX_PROMPT_TOKENS") or "").strip()
    return _parse_int_setting(raw, 6000, 256)


def _summary_max_chars() -> int:
    raw = (os.getenv("ISPEC_ASSISTANT_SUMMARY_MAX_CHARS") or "").strip()
    return _parse_int_setting(raw, 2000, 0)


def _max_tool_calls() -> int:
    raw = (os.getenv("ISPEC_ASSISTANT_MAX_TOOL_CALLS") or "").strip()
    return _parse_int_setting(raw, 2, 0)


# this func is redundant with above env truthy
def _is_truthy(value: str | None) -> bool:
    return _truthy_value(value or "")


def _self_review_enabled() -> bool:
    parsed = _env_tristate_bool("ISPEC_ASSISTANT_SELF_REVIEW")
    if parsed is not None:
        return bool(parsed)
    # Auto-enable self-review in the dev `.pids` tmux workflow when using vLLM.
    return _state_dir_is_dev() and _assistant_provider() == "vllm"


def _self_review_decider_enabled() -> bool:
    parsed = _env_tristate_bool("ISPEC_ASSISTANT_SELF_REVIEW_DECIDER")
    if parsed is not None:
        return bool(parsed)
    # Auto-enable the decider only when self-review is enabled and we're using vLLM.
    return _self_review_enabled() and _assistant_provider() == "vllm"


def _compare_mode_enabled() -> bool:
    parsed = _env_tristate_bool("ISPEC_ASSISTANT_COMPARE_MODE")
    if parsed is not None:
        return bool(parsed)
    # Compare mode changes the UI response shape; keep it off by default.
    return False


def _compaction_enabled() -> bool:
    parsed = _env_tristate_bool("ISPEC_ASSISTANT_COMPACTION_ENABLED")
    if parsed is not None:
        return bool(parsed)
    # Default to enabled (it's only used for vLLM anyway).
    return True


def _compaction_keep_last() -> int:
    raw = (os.getenv("ISPEC_ASSISTANT_COMPACTION_KEEP_LAST") or "").strip()
    return _parse_int_setting(raw, 6, 1)


def _compaction_min_new_messages() -> int:
    raw = (os.getenv("ISPEC_ASSISTANT_COMPACTION_MIN_NEW_MESSAGES") or "").strip()
    return _parse_int_setting(raw, 8, 1)


def _chat_queue_enabled() -> bool:
    parsed = _env_tristate_bool("ISPEC_ASSISTANT_CHAT_QUEUE_ENABLED")
    if parsed is not None:
        return bool(parsed)
    # Default to enabling queue-mode in the top-level dev Makefile (.pids) layout
    # so the supervisor can pick up work quickly, while remaining off by default
    # in production-style deployments.
    return _state_dir_is_dev()


def _project_comment_intent_hints_enabled() -> bool:
    parsed = _env_tristate_bool("ISPEC_ASSISTANT_PROJECT_COMMENT_INTENT_HINTS")
    if parsed is not None:
        return bool(parsed)
    return False


def _chat_queue_wait_seconds() -> float:
    raw = (os.getenv("ISPEC_ASSISTANT_CHAT_QUEUE_WAIT_SECONDS") or "").strip()
    if not raw:
        return 120.0
    try:
        return max(1.0, min(600.0, float(raw)))
    except ValueError:
        return 120.0


def _chat_queue_poll_seconds() -> float:
    raw = (os.getenv("ISPEC_ASSISTANT_CHAT_QUEUE_POLL_SECONDS") or "").strip()
    if not raw:
        return 0.5
    try:
        return max(0.1, min(5.0, float(raw)))
    except ValueError:
        return 0.5


def _queue_force_inline(payload: ChatRequest) -> bool:
    meta = payload.meta if isinstance(payload.meta, dict) else {}
    return bool(meta.get("_queue_force_inline"))


def _chat_queue_supervisor_max_age_seconds() -> float:
    raw = (os.getenv("ISPEC_ASSISTANT_CHAT_QUEUE_SUPERVISOR_MAX_AGE_SECONDS") or "").strip()
    if not raw:
        return 60.0
    try:
        return max(1.0, min(3600.0, float(raw)))
    except ValueError:
        return 60.0


def _as_utc_datetime(value: datetime | None) -> datetime | None:
    if not isinstance(value, datetime):
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    try:
        return value.astimezone(UTC)
    except Exception:
        return value


def _supervisor_heartbeat_ok(*, agent_db: Session) -> bool:
    row = (
        agent_db.query(AgentRun)
        .filter(AgentRun.kind == "supervisor")
        .order_by(AgentRun.updated_at.desc(), AgentRun.id.desc())
        .first()
    )
    if row is None:
        return False
    status = str(getattr(row, "status", "") or "").strip().lower()
    if status != "running":
        return False
    updated_at = _as_utc_datetime(getattr(row, "updated_at", None))
    if updated_at is None:
        return False
    max_age = _chat_queue_supervisor_max_age_seconds()
    return (utcnow() - updated_at).total_seconds() <= float(max_age)


def _decide_if_dualchoice(*, payload: ChatRequest) -> bool:
    # TODO: Implement heuristics for when compare mode is actually helpful.
    return True


def _assistant_provider() -> str:
    return _parse_assistant_provider(os.getenv("ISPEC_ASSISTANT_PROVIDER") or "")


def _tool_protocol() -> str:
    return _parse_tool_protocol(os.getenv("ISPEC_ASSISTANT_TOOL_PROTOCOL") or "")


@cache
def _parse_assistant_provider(raw: str) -> str:
    normalized = raw.strip().lower()
    return normalized or "stub"


@cache
def _parse_tool_protocol(raw: str) -> str:
    normalized = raw.strip().lower()
    return normalized if normalized in {"line", "openai"} else "line"


def _is_confirmation_reply(message: str | None) -> bool:
    if not message:
        return False
    normalized = re.sub(r"[^\w\s]", "", message.strip().lower())
    normalized = re.sub(r"\s+", " ", normalized).strip()
    if not normalized:
        return False
    if len(normalized) > 64:
        return False
    if normalized in {
        "yes",
        "y",
        "yeah",
        "yep",
        "yup",
        "ok",
        "okay",
        "sure",
        "confirm",
        "go ahead",
        "do it",
        "please do",
        "please do it",
        "no",
        "n",
        "nope",
        "nah",
    }:
        return True

    tokens = normalized.split()
    if len(tokens) > 6:
        return False
    affirmative = {"yes", "y", "yeah", "yep", "yup", "ok", "okay", "sure", "confirm"}
    negative = {"no", "n", "nope", "nah"}
    return any(token in affirmative for token in tokens) or any(token in negative for token in tokens)


def _is_affirmative_reply(message: str | None) -> bool:
    if not message:
        return False
    normalized = re.sub(r"[^\w\s]", "", message.strip().lower())
    normalized = re.sub(r"\s+", " ", normalized).strip()
    if not normalized:
        return False
    if len(normalized) > 64:
        return False
    if normalized in {
        "yes",
        "y",
        "yeah",
        "yep",
        "yup",
        "ok",
        "okay",
        "sure",
        "confirm",
        "go ahead",
        "do it",
        "please do",
        "please do it",
    }:
        return True

    tokens = normalized.split()
    if len(tokens) > 6:
        return False
    negative = {"no", "n", "nope", "nah"}
    if any(token in negative for token in tokens):
        return False
    affirmative = {"yes", "y", "yeah", "yep", "yup", "ok", "okay", "sure", "confirm"}
    return any(token in affirmative for token in tokens)


def _assistant_requested_project_history_save(message: str | None) -> bool:
    text = (message or "").strip().lower()
    if not text:
        return False
    if not any(token in text for token in ("save", "log", "record", "add")):
        return False
    if not any(token in text for token in ("history", "comment", "note", "meeting")):
        return False
    if any(phrase in text for phrase in ("would you like me to", "should i", "want me to", "confirm")):
        return True
    if "project history" in text:
        return True
    return "project" in text


def _awaiting_reply_interpretation_state(
    *,
    tool_protocol: str,
    available_tool_names: set[str],
    focused_project_id: int | None,
    last_assistant_message: str | None,
) -> dict[str, Any] | None:
    if str(tool_protocol or "").strip().lower() != "openai":
        return None
    if "create_project_comment" not in available_tool_names:
        return None
    if not isinstance(focused_project_id, int) or focused_project_id <= 0:
        return None
    if not _assistant_requested_project_history_save(last_assistant_message):
        return None
    return {
        "name": "project_comment_save_confirmation",
        "project_id": focused_project_id,
        "pending_tool": "create_project_comment",
    }


def _legacy_reply_interpretation_kind(message: str | None) -> str:
    if not _is_confirmation_reply(message):
        return "unclear"
    return "approve" if _is_affirmative_reply(message) else "deny"


def _reply_interpretation_action(awaiting_state: dict[str, Any] | None, kind: str | None) -> str | None:
    if not isinstance(awaiting_state, dict):
        return None
    state_name = str(awaiting_state.get("name") or "").strip()
    if not state_name:
        return None
    mapping = _REPLY_INTERPRETATION_ACTIONS.get(state_name, {})
    normalized_kind = str(kind or "").strip()
    return mapping.get(normalized_kind)


def _reply_interpretation_messages(
    *,
    action: str | None,
    awaiting_state: dict[str, Any] | None,
) -> list[dict[str, str]]:
    if action == "approve_save" or not isinstance(awaiting_state, dict):
        return []
    text = _REPLY_INTERPRETATION_POLICY_TEXT.get(str(action or "").strip())
    if not text:
        return []
    state_name = str(awaiting_state.get("name") or "").strip()
    if state_name == "project_comment_save_confirmation":
        project_id = awaiting_state.get("project_id")
        if isinstance(project_id, int) and project_id > 0:
            text += f" The pending draft is for project {project_id}."
    return [{"role": "system", "content": text}]


def _remove_write_tools_from_openai_schemas(
    tools: list[dict[str, Any]] | None,
) -> tuple[list[dict[str, Any]] | None, list[str]]:
    if not isinstance(tools, list):
        return tools, []
    filtered: list[dict[str, Any]] = []
    removed: list[str] = []
    for tool in tools:
        if not isinstance(tool, dict):
            filtered.append(tool)
            continue
        func_obj = tool.get("function")
        name = func_obj.get("name") if isinstance(func_obj, dict) else None
        if isinstance(name, str) and name.strip() and tool_writes_data(name.strip()):
            removed.append(name.strip())
            continue
        filtered.append(tool)
    return filtered, sorted(set(removed))


def _assistant_leaks_tool_protocol(message: str | None) -> bool:
    text = (message or "").strip()
    if not text:
        return False
    if TOOL_CALL_PREFIX not in text:
        return False
    return True


def _assistant_claims_write_success(message: str | None) -> bool:
    text = (message or "").strip()
    if not text:
        return False
    if _WRITE_RESULT_ID_RE.search(text):
        return True
    if _WRITE_SUCCESS_NOTE_RE.search(text):
        return True
    if _WRITE_SUCCESS_FIRST_PERSON_RE.search(text):
        return True
    if _WRITE_SUCCESS_PASSIVE_RE.search(text):
        return True
    return False


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
    round_value = payload.get("agent", {}).get("round") if isinstance(payload.get("agent"), dict) else None
    suffix = f" - round {round_value}" if isinstance(round_value, int) and round_value > 0 else ""
    return f"CONTEXT v{version} (read-only JSON){suffix}:\n" + json.dumps(payload, ensure_ascii=False)


def _isoformat_utc(value: datetime | None) -> str | None:
    dt = _as_utc_datetime(value)
    return dt.isoformat() if isinstance(dt, datetime) else None


def _prompt_state_from_session_state(state: dict[str, Any]) -> dict[str, Any]:
    prompt_state: dict[str, Any] = {}

    current_project_id = _safe_int_from_state(state.get("current_project_id"))
    if current_project_id is not None and current_project_id > 0:
        prompt_state["current_project_id"] = int(current_project_id)

    ui_route = state.get("ui_route") if isinstance(state.get("ui_route"), dict) else {}
    ui_payload: dict[str, Any] = {}
    route_name = str(ui_route.get("name") or "").strip()
    route_path = str(ui_route.get("path") or "").strip()
    if route_name:
        ui_payload["route_name"] = route_name
    if route_path:
        ui_payload["route_path"] = route_path

    for state_key, output_key in (
        ("ui_project_id", "project_id"),
        ("ui_experiment_id", "experiment_id"),
        ("ui_experiment_run_id", "experiment_run_id"),
    ):
        value = _safe_int_from_state(state.get(state_key))
        if value is not None and value > 0:
            ui_payload[output_key] = int(value)
    if ui_payload:
        prompt_state["ui"] = ui_payload

    memory = state.get("conversation_memory") if isinstance(state.get("conversation_memory"), dict) else None
    memory = normalize_conversation_memory(memory) if memory is not None else {}
    summary = str(state.get("conversation_summary") or "").strip()
    memory_up_to_id = _safe_int_from_state(state.get("conversation_memory_up_to_id")) or 0
    summary_up_to_id = _safe_int_from_state(state.get("conversation_summary_up_to_id")) or 0

    if memory:
        prompt_state["conversation_memory"] = memory
        if memory_up_to_id > 0:
            prompt_state["conversation_memory_up_to_id"] = int(memory_up_to_id)

    include_summary = bool(summary)
    if memory and memory_up_to_id > 0 and summary_up_to_id > 0 and memory_up_to_id >= summary_up_to_id:
        include_summary = False
    if include_summary:
        prompt_state["conversation_summary"] = summary
        if summary_up_to_id > 0:
            prompt_state["conversation_summary_up_to_id"] = int(summary_up_to_id)

    return prompt_state


def _context_time_payload(
    *,
    session: SupportSession,
    previous_message: SupportMessage | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"now_utc": utcnow().isoformat()}
    previous_message_at = _isoformat_utc(getattr(previous_message, "created_at", None))
    if previous_message_at:
        payload["previous_message_at_utc"] = previous_message_at
    session_created_at = _isoformat_utc(getattr(session, "created_at", None))
    if session_created_at:
        payload["session_created_at_utc"] = session_created_at
    return payload


def _controller_root_state(state: dict[str, Any]) -> dict[str, Any]:
    raw = state.get("controller")
    root = dict(raw) if isinstance(raw, dict) else {}
    state["controller"] = root
    return root


def _support_post_send_state(state: dict[str, Any]) -> dict[str, Any]:
    root = _controller_root_state(state)
    raw = root.get("support_post_send")
    support_state = dict(raw) if isinstance(raw, dict) else {}
    root["support_post_send"] = support_state
    return support_state


def _support_prepared_followup_state(state: dict[str, Any]) -> dict[str, Any] | None:
    root = _controller_root_state(state)
    raw = root.get("prepared_followup")
    return dict(raw) if isinstance(raw, dict) else None


def _set_support_prepared_followup_state(state: dict[str, Any], prepared: dict[str, Any] | None) -> None:
    root = _controller_root_state(state)
    if isinstance(prepared, dict) and prepared:
        root["prepared_followup"] = prepared
    else:
        root.pop("prepared_followup", None)


def _support_prepared_followup_context_message(prepared: dict[str, Any]) -> str:
    payload = {
        "schema_version": 1,
        "kind": "prepared_followup",
        "for_message_id": _safe_int_from_state(prepared.get("for_message_id")),
        "prepared_at": str(prepared.get("prepared_at") or "").strip() or None,
        "summary": str(prepared.get("summary") or "").strip(),
        "issues": list(prepared.get("issues") or [])[:3],
        "followups": list(prepared.get("followups") or [])[:3],
    }
    return "PREPARED_FOLLOWUP_CONTEXT v1 (controller-generated; use as bounded follow-up context):\n" + json.dumps(
        payload, ensure_ascii=False
    )


def _consume_support_prepared_followup_for_turn(
    *,
    state: dict[str, Any],
    user_message_id: int,
) -> tuple[dict[str, str] | None, bool]:
    prepared = _support_prepared_followup_state(state)
    if not isinstance(prepared, dict) or not prepared:
        return None, False
    if _safe_int_from_state(prepared.get("consumed_by_user_message_id")):
        return None, False

    prepared["consumed_by_user_message_id"] = int(user_message_id)
    prepared["consumed_at"] = utcnow().isoformat()
    _set_support_prepared_followup_state(state, prepared)
    return {"role": "system", "content": _support_prepared_followup_context_message(prepared)}, True


def _enqueue_support_post_send_prepare(
    *,
    assistant_db: Session,
    agent_db: Session,
    session: SupportSession,
    state: dict[str, Any],
    target_message_id: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if _assistant_provider() != "vllm":
        return state, {
            "action": "none",
            "enqueued": False,
            "thread_key": support_post_send_thread_key(session.session_id),
            "reason": "provider_not_vllm",
        }
    if not hasattr(agent_db, "query"):
        return state, {
            "action": "none",
            "enqueued": False,
            "thread_key": support_post_send_thread_key(session.session_id),
            "reason": "agent_db_unavailable",
        }

    now = utcnow()
    thread_key = support_post_send_thread_key(session.session_id)
    support_state = _support_post_send_state(state)
    support_state["thread_key"] = thread_key
    support_state["latest_target_message_id"] = int(target_message_id)
    support_state["last_requested_at"] = now.isoformat()

    enqueue_result = enqueue_post_send_prepare_command(
        agent_db=agent_db,
        source="support_chat",
        thread_key=thread_key,
        payload={
            "session_id": session.session_id,
            "session_pk": int(session.id),
            "target_message_id": int(target_message_id),
            "requested_at": now.isoformat(),
        },
        priority=1,
    )

    if enqueue_result.enqueued and enqueue_result.command_id is not None:
        support_state["queued_target_message_id"] = int(target_message_id)
        support_state["queued_command_id"] = int(enqueue_result.command_id)
        support_state["last_enqueued_at"] = now.isoformat()
        support_state.pop("pending_target_message_id", None)
    elif enqueue_result.reason == "duplicate_inflight":
        queued_target_id = _safe_int_from_state(support_state.get("queued_target_message_id")) or 0
        pending_target_id = _safe_int_from_state(support_state.get("pending_target_message_id")) or 0
        if int(target_message_id) > max(queued_target_id, pending_target_id):
            support_state["pending_target_message_id"] = int(target_message_id)
            support_state["pending_requested_at"] = now.isoformat()

    session.state_json = _dump_state(state)
    assistant_db.flush()
    return state, enqueue_result.as_meta()


def _normalize_assistant_brief(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _support_user_summary(user: Any | None) -> dict[str, Any] | None:
    if user is None:
        return None
    return {
        "id": int(getattr(user, "id", 0) or 0),
        "username": str(getattr(user, "username", "") or "") or None,
        "role": str(getattr(getattr(user, "role", None), "value", getattr(user, "role", None)) or "") or None,
        "assistant_brief": _normalize_assistant_brief(getattr(user, "assistant_brief", None)),
    }


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


def _maybe_enqueue_session_compaction(
    *,
    assistant_db: Session,
    agent_db: Session,
    session: SupportSession,
    state: dict[str, Any],
    triggered_by_message_id: int | None,
    reason: str,
) -> tuple[dict[str, Any], bool]:
    if not _compaction_enabled():
        return state, False

    provider = _assistant_provider()
    if provider != "vllm":
        return state, False

    keep_last = _compaction_keep_last()
    min_new_messages = _compaction_min_new_messages()

    memory_up_to_id = _safe_int_from_state(state.get("conversation_memory_up_to_id")) or 0
    requested_up_to_id = _safe_int_from_state(state.get("conversation_memory_requested_up_to_id")) or 0

    boundary_rows = (
        assistant_db.query(SupportMessage)
        .filter(SupportMessage.session_pk == session.id)
        .order_by(SupportMessage.id.desc())
        .limit(keep_last + 1)
        .all()
    )
    if len(boundary_rows) <= keep_last:
        return state, False

    target_id = int(boundary_rows[keep_last].id)
    if target_id <= 0:
        return state, False
    if target_id <= memory_up_to_id or target_id <= requested_up_to_id:
        return state, False

    new_count = (
        assistant_db.query(SupportMessage)
        .filter(SupportMessage.session_pk == session.id)
        .filter(SupportMessage.id > memory_up_to_id)
        .filter(SupportMessage.id <= target_id)
        .count()
    )
    if new_count < min_new_messages:
        return state, False

    agent_db.add(
        AgentCommand(
            command_type=COMMAND_COMPACT_SESSION_MEMORY,
            status="queued",
            priority=0,
            payload_json={
                "session_id": session.session_id,
                "session_pk": int(session.id),
                "target_message_id": target_id,
                "memory_up_to_id": memory_up_to_id,
                "triggered_by_message_id": int(triggered_by_message_id)
                if isinstance(triggered_by_message_id, int)
                else None,
                "requested_at": utcnow().isoformat(),
                "reason": reason,
                "keep_last": keep_last,
                "min_new_messages": min_new_messages,
            },
        )
    )
    state["conversation_memory_requested_up_to_id"] = target_id
    state["conversation_memory_requested_at"] = utcnow().isoformat()
    state["conversation_memory_requested_reason"] = reason
    return state, True


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


def _enqueue_chat_turn_command(*, agent_db: Session, payload: ChatRequest, user: AuthUser | None) -> int:
    queue_payload = {
        "schema_version": 1,
        "chat_request": payload.model_dump(mode="json"),
        "user_id": int(user.id) if user is not None else None,
        "enqueued_at": utcnow().isoformat(),
        "source": "api_support_chat",
    }
    cmd = AgentCommand(
        command_type=COMMAND_SUPPORT_CHAT_TURN,
        status="queued",
        priority=5,
        payload_json=queue_payload,
        result_json={},
    )
    agent_db.add(cmd)
    agent_db.commit()
    agent_db.refresh(cmd)
    return int(cmd.id)


def _poke_orchestrator_tick_now(
    *,
    agent_db: Session,
    source: str,
    session_id: str | None = None,
    delay_seconds: float = 0.0,
) -> int | None:
    """Bring the next orchestrator tick forward so session reviews kick off quickly.

    This is an event-driven "nudge": it does not change what the orchestrator decides,
    it just makes sure it runs promptly after new assistant work lands.
    """

    now = utcnow()
    try:
        delay_seconds = float(delay_seconds)
    except Exception:
        delay_seconds = 0.0
    delay_seconds = max(0.0, min(60.0, delay_seconds))
    target_available_at = now + timedelta(seconds=delay_seconds)

    # If a tick is already queued soon enough, leave it alone.
    existing = (
        agent_db.query(AgentCommand)
        .filter(AgentCommand.command_type == COMMAND_ORCHESTRATOR_TICK)
        .filter(AgentCommand.status == "queued")
        .order_by(AgentCommand.available_at.asc(), AgentCommand.id.asc())
        .first()
    )
    if existing is not None:
        available_at = getattr(existing, "available_at", None)
        if isinstance(available_at, datetime) and available_at <= target_available_at:
            return int(existing.id)
        existing.available_at = target_available_at
        existing.updated_at = now
        payload = dict(existing.payload_json or {})
        payload["source"] = payload.get("source") or "api_poke"
        payload["poke_source"] = str(source or "").strip() or "unknown"
        payload["poked_at"] = now.isoformat()
        if isinstance(session_id, str) and session_id.strip():
            payload["session_id"] = session_id.strip()
        existing.payload_json = payload
        agent_db.commit()
        return int(existing.id)

    cmd_payload: dict[str, Any] = {
        "source": "api_poke",
        "poke_source": str(source or "").strip() or "unknown",
        "poked_at": now.isoformat(),
    }
    if isinstance(session_id, str) and session_id.strip():
        cmd_payload["session_id"] = session_id.strip()

    cmd = AgentCommand(
        command_type=COMMAND_ORCHESTRATOR_TICK,
        status="queued",
        priority=-5,
        payload_json=cmd_payload,
        result_json={},
    )
    cmd.available_at = target_available_at
    agent_db.add(cmd)
    agent_db.commit()
    agent_db.refresh(cmd)
    return int(cmd.id)


def _wait_for_queued_chat_response(
    *,
    agent_db: Session,
    command_id: int,
    wait_seconds: float,
    poll_seconds: float,
) -> ChatResponse:
    deadline = time.monotonic() + max(1.0, float(wait_seconds))
    while True:
        # Start each poll in a fresh transaction so we can observe updates
        # from the supervisor process.
        agent_db.rollback()
        row = agent_db.query(AgentCommand).filter(AgentCommand.id == int(command_id)).first()
        if row is None:
            raise HTTPException(status_code=500, detail=f"Queued chat command {command_id} not found.")

        status = str(row.status or "").strip().lower()
        if status == "succeeded":
            result_payload = row.result_json if isinstance(row.result_json, dict) else {}
            response_payload = result_payload.get("chat_response")
            if not isinstance(response_payload, dict):
                raise HTTPException(
                    status_code=500,
                    detail=f"Queued chat command {command_id} completed without chat_response payload.",
                )
            try:
                return ChatResponse.model_validate(response_payload)
            except Exception as exc:
                raise HTTPException(
                    status_code=500,
                    detail=f"Queued chat command {command_id} returned invalid response: {type(exc).__name__}.",
                ) from exc

        if status == "failed":
            error_text = str(row.error or "").strip() or "queued_chat_failed"
            raise HTTPException(
                status_code=500,
                detail=f"Queued chat command {command_id} failed: {error_text}",
            )

        if time.monotonic() >= deadline:
            raise HTTPException(
                status_code=504,
                detail=f"Timed out waiting for queued chat command {command_id}.",
            )
        time.sleep(max(0.1, float(poll_seconds)))


def _chat_via_queue(*, payload: ChatRequest, assistant_db: Session, agent_db: Session, user: AuthUser | None) -> ChatResponse:
    existing = (
        assistant_db.query(SupportSession)
        .filter(SupportSession.session_id == payload.sessionId)
        .first()
    )
    if existing is not None:
        _enforce_session_access(existing, user)

    command_id = _enqueue_chat_turn_command(agent_db=agent_db, payload=payload, user=user)
    return _wait_for_queued_chat_response(
        agent_db=agent_db,
        command_id=command_id,
        wait_seconds=_chat_queue_wait_seconds(),
        poll_seconds=_chat_queue_poll_seconds(),
    )


@router.post("/chat", response_model=ChatResponse)
def chat(
    payload: ChatRequest,
    request: Request = None,
    assistant_db: Session = Depends(get_assistant_session_dep),
    agent_db: Session = Depends(get_agent_session_dep),
    core_db: Session = Depends(get_session_dep),
    omics_db: Session = Depends(get_omics_session_dep),
    schedule_db: Session = Depends(get_schedule_session_dep),
    user: AuthUser | None = Depends(require_assistant_access),
):
    if _chat_queue_enabled() and not _queue_force_inline(payload) and _supervisor_heartbeat_ok(agent_db=agent_db):
        response = _chat_via_queue(payload=payload, assistant_db=assistant_db, agent_db=agent_db, user=user)
        if response.messageId is not None:
            # Queue mode: assistant message is committed by the supervisor, so we can poke immediately.
            try:
                _poke_orchestrator_tick_now(
                    agent_db=agent_db,
                    source="support_chat_queue",
                    session_id=payload.sessionId,
                    delay_seconds=0,
                )
            except Exception:
                pass
        return response

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

    if isinstance(state.get("conversation_memory"), dict):
        normalized_memory = normalize_conversation_memory(state.get("conversation_memory"))
        if normalized_memory != state.get("conversation_memory"):
            state["conversation_memory"] = normalized_memory
            state_changed = True

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
        provider=(
            "slack"
            if isinstance(payload.meta, dict)
            and str(payload.meta.get("source") or "").strip().lower() == "slack"
            else "frontend"
        ),
        meta_json=(
            json.dumps({"client_meta": payload.meta}, ensure_ascii=False)
            if isinstance(payload.meta, dict) and payload.meta
            else None
        ),
    )
    assistant_db.add(user_message)
    session.updated_at = utcnow()
    assistant_db.flush()
    previous_session_message = (
        assistant_db.query(SupportMessage)
        .filter(SupportMessage.session_pk == session.id)
        .filter(SupportMessage.id < int(user_message.id))
        .order_by(SupportMessage.id.desc())
        .first()
    )
    assistant_db.commit()

    prepared_followup_message = None
    prepared_followup_consumed = False
    prepared_followup_message, prepared_followup_consumed = _consume_support_prepared_followup_for_turn(
        state=state,
        user_message_id=int(user_message.id),
    )
    if prepared_followup_consumed:
        state_changed = True

    history_limit = _history_limit()
    max_tokens = _max_prompt_tokens()
    assistant_provider = _assistant_provider()

    tool_protocol = _tool_protocol()
    max_tool_calls = _max_tool_calls()
    used_tool_calls = 0
    tools_enabled = max_tool_calls > 0
    tools_enabled_initial = tools_enabled
    tool_router: dict[str, Any] | None = None
    tool_schemas = openai_tools_for_user(user) if tool_protocol == "openai" and tools_enabled else None
    known_tool_names = openai_tool_names_all() if tool_protocol == "openai" and tools_enabled else set()
    available_openai_tool_names_full = _openai_tool_names(tool_schemas)
    available_write_tool_names_full = sorted(
        name for name in available_openai_tool_names_full if tool_writes_data(name)
    )
    requested_tool_names_known = (
        _requested_tool_names_from_text(payload.message, candidates=known_tool_names) if known_tool_names else []
    )

    confirmation_reply = _is_confirmation_reply(payload.message)
    compare_mode_enabled = _compare_mode_enabled()
    self_review_enabled = _self_review_enabled()
    self_review_decider_enabled = _self_review_decider_enabled()
    turn_decision_mode = _turn_decision_mode()
    turn_decision_result = None
    turn_decision_runtime_applied = False
    turn_decision_groups = (
        tool_groups_for_available_tools(available_openai_tool_names_full)
        if tool_protocol == "openai" and tools_enabled and isinstance(tool_schemas, list) and tool_schemas
        else []
    )
    last_assistant_turn_row = (
        assistant_db.query(SupportMessage)
        .filter(SupportMessage.session_pk == session.id)
        .filter(SupportMessage.role == "assistant")
        .order_by(SupportMessage.id.desc())
        .first()
    )
    last_assistant_text_for_turn_decision = (
        str(getattr(last_assistant_turn_row, "content", "") or "").strip()
        if last_assistant_turn_row is not None
        else ""
    )
    awaiting_reply_state = _awaiting_reply_interpretation_state(
        tool_protocol=tool_protocol,
        available_tool_names=available_openai_tool_names_full,
        focused_project_id=focused_project_id,
        last_assistant_message=last_assistant_text_for_turn_decision,
    )
    legacy_reply_interpretation_kind = (
        _legacy_reply_interpretation_kind(payload.message)
        if awaiting_reply_state is not None
        else "none"
    )
    legacy_reply_interpretation_action = _reply_interpretation_action(
        awaiting_reply_state,
        legacy_reply_interpretation_kind,
    )
    if assistant_provider == "vllm" and turn_decision_mode != "off":
        turn_decision_result = run_turn_decision_pipeline(
            generate_reply_fn=generate_reply,
            mode=turn_decision_mode,
            source="support_chat",
            user_message=payload.message,
            last_assistant_message=last_assistant_text_for_turn_decision,
            focused_project_id=focused_project_id,
            referenced_project_ids=referenced_projects,
            groups=turn_decision_groups,
            response_modes=["single", "compare"] if compare_mode_enabled else ["single"],
            contract_caps=list(response_contract_names()),
            extra_context={
                "confirmation_reply": confirmation_reply,
                "tool_protocol": tool_protocol,
                "tools_enabled": bool(tools_enabled),
                "requested_tool_names": requested_tool_names_known,
                "available_write_tools": available_write_tool_names_full,
                "awaiting_reply_state": awaiting_reply_state,
                "ui": ui_payload or {},
            },
        )
        turn_decision_runtime_applied = bool(
            turn_decision_result.ok and turn_decision_mode == "own" and turn_decision_result.decision is not None
        )
    classifier_reply_interpretation_kind = (
        turn_decision_result.decision.reply_interpretation.kind
        if turn_decision_result is not None and turn_decision_result.decision is not None
        else "none"
    )
    classifier_reply_interpretation_action = _reply_interpretation_action(
        awaiting_reply_state,
        classifier_reply_interpretation_kind,
    )
    reply_interpretation_runtime_applied = bool(
        turn_decision_runtime_applied and awaiting_reply_state is not None
    )
    reply_interpretation_runtime_kind = (
        classifier_reply_interpretation_kind
        if reply_interpretation_runtime_applied
        else legacy_reply_interpretation_kind
    )
    reply_interpretation_runtime_action = (
        classifier_reply_interpretation_action
        if reply_interpretation_runtime_applied
        else legacy_reply_interpretation_action
    )
    reply_interpretation_policy_messages = _reply_interpretation_messages(
        action=reply_interpretation_runtime_action if reply_interpretation_runtime_applied else None,
        awaiting_state=awaiting_reply_state,
    )
    skip_tool_router_for_reply_interpretation = bool(reply_interpretation_runtime_applied)

    selected_policy_rule_name: str | None = None
    tool_selector_source: str | None = None
    support_policy_selection = None
    if (
        tool_protocol == "openai"
        and tools_enabled
        and isinstance(tool_schemas, list)
        and tool_schemas
        and assistant_provider == "vllm"
        and not confirmation_reply
        and not skip_tool_router_for_reply_interpretation
    ):
        available_tool_names = _openai_tool_names(tool_schemas)
        requested_tool_names_available = _requested_tool_names_from_text(
            payload.message, candidates=available_tool_names
        )
        groups = turn_decision_groups or tool_groups_for_available_tools(available_tool_names)
        support_policy_selection = select_support_tool_policy(
            message=payload.message,
            focused_project_id=focused_project_id,
            generate_reply_fn=generate_reply,
        )
        hinted_group_names = hinted_support_tool_groups(
            message=payload.message,
            focused_project_id=focused_project_id,
        )
        if support_policy_selection is not None and support_policy_selection.force_tool_choice:
            tool_selector_source = "support_policy_rule"
            selected_policy_rule_name = support_policy_selection.rule_name
            grouped_tool_names = {group.name: set(group.tool_names) for group in groups}
            selected_tool_names = {
                "assistant_prompt_header",
                "assistant_list_tools",
                support_policy_selection.tool_name,
            }
            selected_tool_names |= grouped_tool_names.get("tmux", set())
            if requested_tool_names_available:
                selected_tool_names |= set(requested_tool_names_available)
            tool_router = {
                "ok": True,
                "source": "support_policy_rule",
                "groups": [{"name": group.name, "tools": list(group.tool_names)} for group in groups],
                "policy_rule": support_policy_selection.rule_name,
                "hinted_group_names": sorted(hinted_group_names) if hinted_group_names else [],
                "selected_tool_names": sorted(selected_tool_names),
            }
            if support_policy_selection.meta:
                tool_router["tmux_resolution"] = dict(support_policy_selection.meta)
            filtered: list[dict[str, Any]] = []
            for spec in tool_schemas:
                func_obj = spec.get("function") if isinstance(spec, dict) else None
                name = func_obj.get("name") if isinstance(func_obj, dict) else None
                if isinstance(name, str) and name.strip() in selected_tool_names:
                    filtered.append(spec)
            if filtered:
                tool_schemas = filtered
        elif turn_decision_runtime_applied and turn_decision_result and turn_decision_result.decision is not None:
            tool_selector_source = "turn_decision"
            selected_tool_names = selected_tool_names_from_decision(
                groups=groups,
                decision=turn_decision_result.decision,
                explicit_requested_tool_names=requested_tool_names_available,
                always_include=["assistant_prompt_header", "assistant_list_tools"],
            )
            tool_router = {
                "ok": True,
                "source": "turn_decision",
                "groups": [{"name": group.name, "tools": list(group.tool_names)} for group in groups],
                "decision": turn_decision_result.decision.as_dict(),
            }
            if selected_tool_names:
                filtered: list[dict[str, Any]] = []
                for spec in tool_schemas:
                    func_obj = spec.get("function") if isinstance(spec, dict) else None
                    name = func_obj.get("name") if isinstance(func_obj, dict) else None
                    if isinstance(name, str) and name.strip() in selected_tool_names:
                        filtered.append(spec)
                if filtered:
                    tool_schemas = filtered
                    tool_router["selected_tool_names"] = sorted(selected_tool_names)
        else:
            decision, router_reply = route_tool_groups_vllm(
                user_message=payload.message,
                groups=groups,
                generate_reply_fn=generate_reply,
            )
            tool_router = {
                "ok": bool(router_reply.ok),
                "provider": router_reply.provider,
                "model": router_reply.model,
                "groups": [{"name": group.name, "tools": list(group.tool_names)} for group in groups],
                "decision": decision,
            }
            if turn_decision_result and turn_decision_result.decision is not None:
                tool_router["turn_decision"] = turn_decision_result.decision.as_dict()
            if requested_tool_names_available:
                tool_router["explicit_requested_tool_names"] = requested_tool_names_available
            if router_reply.meta:
                tool_router["provider_meta"] = router_reply.meta
            if hinted_group_names:
                tool_router["hinted_group_names"] = sorted(hinted_group_names)
            if support_policy_selection is not None:
                tool_router["policy_rule"] = support_policy_selection.rule_name

            grouped_tool_names = {group.name: set(group.tool_names) for group in groups}
            if decision:
                tool_selector_source = "generic_router"
                selected_tool_names = tool_names_for_groups(
                    groups=groups,
                    primary=str(decision["primary"]),
                    secondary=list(decision["secondary"]),
                )
                for group_name in hinted_group_names:
                    selected_tool_names |= grouped_tool_names.get(group_name, set())
                selected_tool_names |= {"assistant_prompt_header", "assistant_list_tools"}
                if requested_tool_names_available:
                    selected_tool_names |= set(requested_tool_names_available)
                if selected_tool_names:
                    filtered: list[dict[str, Any]] = []
                    for spec in tool_schemas:
                        func_obj = spec.get("function") if isinstance(spec, dict) else None
                        name = func_obj.get("name") if isinstance(func_obj, dict) else None
                        if isinstance(name, str) and name.strip() in selected_tool_names:
                            filtered.append(spec)
                if filtered:
                    tool_schemas = filtered
                    tool_router["selected_tool_names"] = sorted(selected_tool_names)
            elif hinted_group_names:
                tool_selector_source = "hinted_groups"
                selected_tool_names = {"assistant_prompt_header", "assistant_list_tools"}
                for group_name in hinted_group_names:
                    selected_tool_names |= grouped_tool_names.get(group_name, set())
                filtered = []
                for spec in tool_schemas:
                    func_obj = spec.get("function") if isinstance(spec, dict) else None
                    name = func_obj.get("name") if isinstance(func_obj, dict) else None
                    if isinstance(name, str) and name.strip() in selected_tool_names:
                        filtered.append(spec)
                if filtered:
                    tool_schemas = filtered
                    tool_router["selected_tool_names"] = sorted(selected_tool_names)
    elif (
        tool_protocol == "openai"
        and tools_enabled
        and isinstance(tool_schemas, list)
        and tool_schemas
        and assistant_provider == "vllm"
        and (confirmation_reply or skip_tool_router_for_reply_interpretation)
    ):
        tool_router = {
            "ok": True,
            "skipped": True,
            "reason": "reply_interpretation" if skip_tool_router_for_reply_interpretation else "confirmation_reply",
        }

    reply_interpretation_removed_write_tools: list[str] = []
    if (
        tool_protocol == "openai"
        and reply_interpretation_runtime_applied
        and reply_interpretation_runtime_action != "approve_save"
    ):
        tool_schemas, reply_interpretation_removed_write_tools = _remove_write_tools_from_openai_schemas(tool_schemas)
        if isinstance(tool_router, dict) and reply_interpretation_removed_write_tools:
            tool_router["reply_interpretation_removed_write_tools"] = reply_interpretation_removed_write_tools

    tool_schemas_count = len(tool_schemas) if isinstance(tool_schemas, list) else 0
    openai_tool_names = _openai_tool_names(tool_schemas)
    openai_no_arg_tool_names = _openai_no_arg_tool_names(tool_schemas)
    available_write_tool_names = sorted(name for name in openai_tool_names if tool_writes_data(name))
    missing_requested_tool_names = [
        name for name in requested_tool_names_known if isinstance(name, str) and name.strip() not in openai_tool_names
    ]
    tool_availability_messages: list[dict[str, Any]] = []
    if missing_requested_tool_names:
        tool_availability_messages = [
            {
                "role": "system",
                "content": (
                    "NOTE: The user requested tool(s) by name that are not available for this turn: "
                    + ", ".join(missing_requested_tool_names)
                    + ". Do not claim you called them or invent their outputs; explain that the tool is unavailable "
                    "(permissions/config) and ask to enable it if needed."
                ),
            }
        ]
    if tool_router is not None and requested_tool_names_known:
        tool_router["requested_tool_names"] = requested_tool_names_known
        if missing_requested_tool_names:
            tool_router["missing_requested_tool_names"] = missing_requested_tool_names

    turn_decision_contract_cap = (
        turn_decision_result.decision.response_plan.contract_cap
        if turn_decision_runtime_applied and turn_decision_result and turn_decision_result.decision is not None
        else None
    )
    compare_mode_requested = (
        compare_mode_enabled and bool(turn_decision_result.decision.response_plan.mode == "compare")
        if turn_decision_runtime_applied and turn_decision_result and turn_decision_result.decision is not None
        else compare_mode_enabled and _decide_if_dualchoice(payload=payload)
    )
    response_format = "compare" if compare_mode_requested else "single"

    tools_available = tool_schemas_count > 0 and tools_enabled
    planner_prompt = _system_prompt_planner(
        tools_available=tools_available,
        response_format=response_format,
        tool_names=openai_tool_names if tool_protocol == "openai" else None,
    )
    answer_prompt = _system_prompt_answer(response_format=response_format)
    prompt_for_budget = planner_prompt if tools_enabled else answer_prompt

    selected_history: list[SupportMessage] = []
    context_message = ""

    # Iterate twice to account for summary growth impacting the budget.
    for _ in range(2):
        summary_up_to_id = _safe_int_from_state(state.get("conversation_summary_up_to_id")) or 0
        memory_up_to_id = 0
        if isinstance(state.get("conversation_memory"), dict) and state.get("conversation_memory"):
            memory_up_to_id = _safe_int_from_state(state.get("conversation_memory_up_to_id")) or 0
        history_cutoff_id = max(summary_up_to_id, memory_up_to_id)

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
            and row.id > history_cutoff_id
            and row.role in {"user", "assistant", "system"}
            and row.content
        ]

        ispec_context = build_ispec_context(core_db, message=payload.message, state=state, user=user)
        state_for_context = _prompt_state_from_session_state(state)
        context_payload: dict[str, Any] = {
            "schema_version": _CONTEXT_SCHEMA_VERSION,
            "agent": {"multi_round": True, "round": 0},
            "session": {"id": session.session_id, "state": state_for_context},
            "time": _context_time_payload(session=session, previous_message=previous_session_message),
            "user": _support_user_summary(user),
            "ispec": ispec_context,
        }
        context_message = _context_message(payload=context_payload)

        base_messages = [
            {"role": "system", "content": prompt_for_budget},
            {"role": "system", "content": context_message},
            *([prepared_followup_message] if prepared_followup_message else []),
            *tool_availability_messages,
            *reply_interpretation_policy_messages,
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
    last_assistant_text = next(
        (
            str(item.get("content") or "")
            for item in reversed(history_payload)
            if isinstance(item, dict) and item.get("role") == "assistant" and item.get("content")
        ),
        "",
    )
    comment_intent_decision: dict[str, Any] | None = None
    comment_intent_reply = None
    comment_intent_messages: list[dict[str, Any]] = []
    if turn_decision_runtime_applied and turn_decision_result and turn_decision_result.decision is not None:
        write_mode = turn_decision_result.decision.write_plan.mode
        if not (reply_interpretation_runtime_applied and reply_interpretation_runtime_action != "approve_save"):
            comment_intent_messages = comment_intent_messages_for_write_mode(
                write_mode=write_mode,
                missing_comment_text=(
                    write_mode == "save_now"
                    and turn_decision_result.decision.needs_clarification
                    and turn_decision_result.decision.clarification_reason == "missing_comment_text"
                ),
            )
    elif (
        tool_protocol == "openai"
        and tools_enabled
        and assistant_provider == "vllm"
        and _project_comment_intent_hints_enabled()
        and "create_project_comment" in openai_tool_names
        and (confirmation_reply or _PROJECT_COMMENT_INTENT_RE.search(payload.message or ""))
    ):
        comment_intent_decision, comment_intent_reply = decide_project_comment_intent_vllm(
            user_message=payload.message,
            last_assistant_message=last_assistant_text,
            focused_project_id=focused_project_id,
            generate_reply_fn=generate_reply,
        )
        if isinstance(comment_intent_decision, dict):
            intent = str(comment_intent_decision.get("intent") or "").strip()
            confidence = float(comment_intent_decision.get("confidence") or 0.0)
            reason = _truncate(str(comment_intent_decision.get("reason") or ""), 240)
            comment_intent_messages = comment_intent_messages_for_write_mode(
                write_mode=_legacy_comment_intent_to_write_mode(comment_intent_decision),
                confidence=confidence,
                reason=reason,
            )

    tool_calls: list[dict[str, Any]] = []
    tool_messages: list[dict[str, Any]] = []
    tool_result_messages: list[dict[str, Any]] = []
    llm_trace: list[dict[str, Any]] = []
    reply = None
    history_for_llm: list[dict[str, Any]] = []
    write_claim_retry_used = False
    write_claim_guard_triggered = False
    tool_protocol_retry_used = False
    tool_protocol_guard_triggered = False
    forced_tool_choice: str | dict[str, Any] | None = None
    if (
        tool_protocol == "openai"
        and reply_interpretation_runtime_applied
        and reply_interpretation_runtime_action == "approve_save"
        and isinstance(focused_project_id, int)
        and focused_project_id > 0
        and "create_project_comment" in openai_tool_names
    ):
        forced_tool_choice = {"type": "function", "function": {"name": "create_project_comment"}}
    if (
        tool_protocol == "openai"
        and confirmation_reply
        and not _project_comment_intent_hints_enabled()
        and _is_affirmative_reply(payload.message)
        and isinstance(focused_project_id, int)
        and focused_project_id > 0
        and "create_project_comment" in openai_tool_names
    ):
        if _assistant_requested_project_history_save(last_assistant_text):
            forced_tool_choice = {"type": "function", "function": {"name": "create_project_comment"}}
    if (
        tool_protocol == "openai"
        and forced_tool_choice is None
        and turn_decision_runtime_applied
        and turn_decision_result
        and turn_decision_result.decision is not None
        and turn_decision_result.decision.tool_plan.preferred_first_tool in openai_tool_names
    ):
        tool_selector_source = tool_selector_source or "turn_decision"
        forced_tool_choice = {
            "type": "function",
            "function": {"name": str(turn_decision_result.decision.tool_plan.preferred_first_tool)},
        }
    policy_tool_name = support_policy_selection.tool_name if support_policy_selection is not None else None
    policy_tool_args = dict(support_policy_selection.args or {}) if support_policy_selection is not None else {}
    policy_messages = list(support_policy_selection.messages or []) if support_policy_selection is not None else []
    policy_force_tool_choice = bool(support_policy_selection.force_tool_choice) if support_policy_selection is not None else False
    policy_override_tool_args = bool(support_policy_selection.override_tool_args) if support_policy_selection is not None else False
    if policy_tool_name and (
        policy_tool_name not in openai_tool_names
        or (policy_tool_name not in openai_no_arg_tool_names and not policy_tool_args)
    ):
        policy_tool_name = None
        policy_tool_args = {}
        policy_messages = []
        support_policy_selection = None

    if (
        tool_protocol == "openai"
        and (forced_tool_choice is None or policy_force_tool_choice)
        and policy_tool_name in {"assistant_list_tmux_panes", "assistant_capture_tmux_pane", "get_project"}
        and policy_tool_name in openai_tool_names
    ):
        tool_selector_source = tool_selector_source or "support_policy_rule"
        selected_policy_rule_name = (
            support_policy_selection.rule_name if support_policy_selection is not None else selected_policy_rule_name
        )
        forced_tool_choice = {"type": "function", "function": {"name": policy_tool_name}}

    prompt_header = None
    prompt_header_included_round1 = False
    prompt_header_error: str | None = None
    if prompt_header_enabled():
        try:
            prompt_header = build_prompt_header(
                session_id=session.session_id,
                user_role=user.role if user is not None else None,
                user_id=int(user.id) if user is not None else None,
                session_state=state_for_context,
                user_message_id=int(user_message.id),
                tools_available=bool(tools_available),
                tool_protocol=str(tool_protocol or ""),
                compare_mode=bool(compare_mode_requested),
                forced_tool_choice=forced_tool_choice is not None,
                # Reflect tools actually available for this turn (tool routing may
                # restrict the schemas we pass to the model).
                repo_tools_enabled=bool(any(name.startswith("repo_") for name in openai_tool_names)),
            )
        except Exception as exc:
            prompt_header_error = f"{type(exc).__name__}: {exc}"

    llm_observability_context = {
        "surface": "support_chat",
        "session_id": session.session_id,
        "message_id": int(user_message.id),
    }

    llm_round = 0

    while True:
        llm_round += 1
        agent_state = context_payload.get("agent") if isinstance(context_payload.get("agent"), dict) else None
        if agent_state is not None:
            agent_state["round"] = llm_round
        context_message = _context_message(payload=context_payload)

        system_prompt = planner_prompt if tools_enabled else answer_prompt
        header_message = (
            [{"role": "system", "content": prompt_header.line}]
            if prompt_header is not None and llm_round == 1
            else []
        )
        if header_message:
            prompt_header_included_round1 = True
        base_messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            *header_message,
            {"role": "system", "content": context_message},
            *([prepared_followup_message] if prepared_followup_message else []),
            *tool_availability_messages,
            *policy_messages,
            *reply_interpretation_policy_messages,
            *comment_intent_messages,
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
        history_for_llm = trimmed_history

        messages_for_llm: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            *header_message,
            {"role": "system", "content": context_message},
            *([prepared_followup_message] if prepared_followup_message else []),
            *tool_availability_messages,
            *policy_messages,
            *reply_interpretation_policy_messages,
            *comment_intent_messages,
            *trimmed_history,
            {"role": "user", "content": payload.message},
            *tool_messages,
        ]

        tool_choice = (
            forced_tool_choice
            if tool_schemas is not None and not tool_calls and used_tool_calls == 0
            else None
        )
        tools_for_call = tool_schemas if tools_enabled else None
        effective_tool_choice: str | dict[str, Any] | None = None
        if tools_for_call is not None:
            effective_tool_choice = tool_choice if tool_choice is not None else "auto"

        trace_step: dict[str, Any] = {
            "round": llm_round,
            "prompt": "planner" if tools_enabled else "answer",
            "tools_provided": tools_for_call is not None,
            "tools_count": tool_schemas_count if tools_for_call is not None else 0,
            "tool_choice": effective_tool_choice,
            "history_messages": len(trimmed_history),
            "tool_messages": len(tool_messages),
        }
        reply = generate_reply(
            messages=messages_for_llm,
            tools=tools_for_call,
            tool_choice=tool_choice,
            observability_context={
                **llm_observability_context,
                "stage": "planner" if tools_enabled else "answer",
            },
        )
        trace_step["provider"] = reply.provider
        trace_step["model"] = reply.model
        if reply.meta and isinstance(reply.meta, dict):
            trace_step["provider_meta"] = {
                "elapsed_ms": reply.meta.get("elapsed_ms"),
                "usage": reply.meta.get("usage"),
                "fallback": reply.meta.get("fallback"),
            }
        trace_step["reply_preview"] = _truncate(reply.content, 320)
        if reply.tool_calls:
            tool_call_names: list[str] = []
            for tool_call_item in reply.tool_calls:
                if not isinstance(tool_call_item, dict):
                    continue
                func_obj = tool_call_item.get("function")
                name_obj: Any = tool_call_item.get("name")
                if isinstance(func_obj, dict):
                    name_obj = func_obj.get("name")
                if isinstance(name_obj, str) and name_obj.strip():
                    tool_call_names.append(name_obj.strip())
            if tool_call_names:
                trace_step["reply_tool_calls"] = tool_call_names
        llm_trace.append(trace_step)

        if reply.tool_calls:
            trace_step["action"] = "openai_tool_calls"
            tool_messages.append(
                {
                    "role": "assistant",
                    "content": reply.content or "",
                    "tool_calls": reply.tool_calls,
                }
            )
            executed_openai_tools: list[str] = []
            for tool_call in reply.tool_calls:
                if not isinstance(tool_call, dict):
                    continue
                call_id = str(tool_call.get("id") or tool_call.get("tool_call_id") or "")
                if not call_id:
                    call_id = f"call_{used_tool_calls + 1}"
                func_obj = tool_call.get("function") if isinstance(tool_call.get("function"), dict) else {}
                tool_name = str(func_obj.get("name") or "").strip()
                if tool_name:
                    executed_openai_tools.append(tool_name)
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

                if (
                    policy_override_tool_args
                    and not tool_calls
                    and tool_name == policy_tool_name
                    and isinstance(parsed_args, dict)
                ):
                    parsed_args = dict(policy_tool_args)

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
                        assistant_db=assistant_db,
                        agent_db=agent_db,
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
            if executed_openai_tools:
                trace_step["executed_tools"] = executed_openai_tools
            continue

        tool_call = parse_tool_call(reply.content)
        if tool_call is None:
            suggested_tool_name = _suggested_tool_from_text(reply.content, candidates=openai_no_arg_tool_names)
            executed_tools = {
                str(call.get("name") or "").strip()
                for call in tool_calls
                if isinstance(call, dict) and call.get("name")
            }
            if (
                tools_enabled
                and used_tool_calls < max_tool_calls
                and suggested_tool_name
                and suggested_tool_name not in executed_tools
            ):
                used_tool_calls += 1
                tool_name = suggested_tool_name
                tool_args: dict[str, Any] = {}
                tool_payload = run_tool(
                    name=tool_name,
                    args=tool_args,
                    core_db=core_db,
                    assistant_db=assistant_db,
                    agent_db=agent_db,
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
                        "protocol": "suggested",
                    }
                )
                tool_call_line = TOOL_CALL_PREFIX + " " + json.dumps(
                    {"name": tool_name, "arguments": tool_args},
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                tool_result_text = format_tool_result_message(tool_name, tool_payload)
                tool_result_messages.append({"role": "system", "content": tool_result_text})
                tool_messages.extend(
                    [
                        {"role": "assistant", "content": tool_call_line},
                        {"role": "system", "content": tool_result_text},
                    ]
                )
                trace_step["action"] = "suggested_tool"
                trace_step["executed_tools"] = [tool_name]
                if used_tool_calls >= max_tool_calls:
                    tools_enabled = False
                continue

            if (
                tools_enabled
                and used_tool_calls == 0
                and policy_tool_name
                and used_tool_calls < max_tool_calls
            ):
                used_tool_calls += 1
                tool_name = policy_tool_name
                tool_args: dict[str, Any] = dict(policy_tool_args)
                tool_payload = run_tool(
                    name=tool_name,
                    args=tool_args,
                    core_db=core_db,
                    assistant_db=assistant_db,
                    agent_db=agent_db,
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
                        "protocol": "policy",
                    }
                )
                tool_call_line = TOOL_CALL_PREFIX + " " + json.dumps(
                    {"name": tool_name, "arguments": tool_args},
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                tool_result_text = format_tool_result_message(tool_name, tool_payload)
                tool_result_messages.append({"role": "system", "content": tool_result_text})
                tool_messages.extend(
                    [
                        {"role": "assistant", "content": tool_call_line},
                        {"role": "system", "content": tool_result_text},
                    ]
                )
                trace_step["action"] = "policy_tool"
                trace_step["executed_tools"] = [tool_name]
                if used_tool_calls >= max_tool_calls:
                    tools_enabled = False
                continue
            if _assistant_leaks_tool_protocol(reply.content):
                if not tool_protocol_retry_used:
                    tool_messages.extend(
                        [
                            {"role": "assistant", "content": reply.content or ""},
                            {
                                "role": "system",
                                "content": (
                                    "Do not expose the TOOL_CALL protocol or ask the user to paste tool-call JSON. "
                                    "Either call the tool yourself or answer in normal user-facing language."
                                ),
                            },
                        ]
                    )
                    trace_step["action"] = "retry_after_tool_protocol_leak"
                    trace_step["guardrail"] = "tool_protocol_leak"
                    tool_protocol_retry_used = True
                    tool_protocol_guard_triggered = True
                    continue
            successful_write_tools = {
                str(call.get("name") or "").strip()
                for call in tool_calls
                if isinstance(call, dict) and call.get("ok") and tool_writes_data(call.get("name"))
            }
            if not successful_write_tools and _assistant_claims_write_success(reply.content):
                if not write_claim_retry_used:
                    corrective_parts = [
                        "Your previous draft claimed a successful write/save/update, but no successful write tool call has occurred in this turn.",
                        "Do not invent saved data, record IDs, or success outcomes.",
                    ]
                    if available_write_tool_names:
                        corrective_parts.append(
                            "If a write is actually needed, choose the appropriate tool now from: "
                            + ", ".join(available_write_tool_names)
                            + "."
                        )
                    else:
                        corrective_parts.append(
                            "No write tool is available in this turn, so explain that nothing was saved yet."
                        )
                    tool_messages.extend(
                        [
                            {"role": "assistant", "content": reply.content or ""},
                            {"role": "system", "content": " ".join(corrective_parts)},
                        ]
                    )
                    trace_step["action"] = "retry_after_unsupported_write_claim"
                    trace_step["guardrail"] = "unsupported_write_claim"
                    write_claim_retry_used = True
                    write_claim_guard_triggered = True
                    continue
            trace_step["action"] = "final_no_tool"
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
            assistant_db=assistant_db,
            agent_db=agent_db,
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
        trace_step["action"] = "line_tool_call"
        trace_step["executed_tools"] = [tool_name]

    if reply is None:
        history_for_llm = history_payload
        reply = generate_reply(
            messages=[
                {"role": "system", "content": answer_prompt},
                {"role": "system", "content": context_message},
                *tool_availability_messages,
                *reply_interpretation_policy_messages,
                *history_payload,
                {"role": "user", "content": payload.message},
                *tool_messages,
                {"role": "system", "content": f"{TOOL_CALL_PREFIX} limit exceeded; answer without tools."},
            ],
            tools=None,
            observability_context={**llm_observability_context, "stage": "answer"},
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

    response_contract_mode = response_contracts_mode()
    response_contract_would_apply_if_live, response_contract_protection_reason = _response_contract_live_eligibility(
        tool_calls=tool_calls,
        comment_intent_decision=comment_intent_decision,
        turn_decision_result=turn_decision_result,
    )
    response_contract_request_meta = _merge_response_contract_request_meta(
        request_meta=payload.meta if isinstance(payload.meta, dict) else None,
        contract_cap=turn_decision_contract_cap if turn_decision_runtime_applied else None,
    )
    controller_pre_send = run_message_pre_send_controller(
        generate_reply_fn=generate_reply,
        source="support_chat",
        context_message=context_message,
        history_messages=history_for_llm,
        user_message=payload.message,
        tool_messages=tool_messages,
        tool_result_messages=tool_result_messages,
        draft_answer=assistant_content,
        draft_reply=final_reply,
        compare_mode=compare_choices is not None,
        request_meta=response_contract_request_meta,
        response_contract_mode=response_contract_mode,
        response_contract_would_apply_if_live=response_contract_would_apply_if_live,
        response_contract_protection_reason=response_contract_protection_reason,
        self_review_enabled=self_review_enabled,
        self_review_decider_enabled=assistant_provider == "vllm" and self_review_decider_enabled,
        used_tool_calls=used_tool_calls,
    )
    assistant_content = controller_pre_send.final_content
    final_reply = controller_pre_send.final_reply
    final_raw_reply_content = controller_pre_send.final_raw_content
    response_contract_meta = controller_pre_send.response_contract_meta
    response_contract_meta["configured_mode"] = response_contract_mode
    response_contract_meta["would_apply_if_live"] = response_contract_would_apply_if_live
    response_contract_meta["protection_reason"] = response_contract_protection_reason
    response_contract_applied = controller_pre_send.response_contract_applied
    self_review_changed = controller_pre_send.self_review_changed
    self_review_error = controller_pre_send.self_review_error
    self_review_mode = controller_pre_send.self_review_mode
    self_review_decision = controller_pre_send.self_review_decision

    successful_write_tools_final = {
        str(call.get("name") or "").strip()
        for call in tool_calls
        if isinstance(call, dict) and call.get("ok") and tool_writes_data(call.get("name"))
    }
    blocked_unsupported_write_claim = False
    if not successful_write_tools_final and _assistant_claims_write_success(assistant_content):
        blocked_unsupported_write_claim = True
        write_claim_guard_triggered = True
        assistant_content = (
            "I can't confirm that I saved that because no write tool actually succeeded. "
            "I haven't made that change yet."
        )

    blocked_tool_protocol_leak = False
    if _assistant_leaks_tool_protocol(assistant_content):
        blocked_tool_protocol_leak = True
        tool_protocol_guard_triggered = True
        assistant_content = (
            "I can help with that directly. Tell me the draft you want, or ask me to save a project note, "
            "and I will handle the tool call myself."
        )

    turn_decision_meta: dict[str, Any] | None = None
    if turn_decision_result is not None:
        turn_decision_meta = turn_decision_result.as_meta()
        runtime_router_decision = tool_router.get("decision") if isinstance(tool_router, dict) else None
        runtime_primary_group = None
        runtime_secondary_groups: list[str] = []
        if isinstance(runtime_router_decision, dict):
            runtime_primary_group = str(runtime_router_decision.get("primary") or "").strip() or None
            runtime_secondary_groups = [
                str(item).strip()
                for item in (runtime_router_decision.get("secondary") or [])
                if isinstance(item, str) and str(item).strip()
            ]
        runtime_forced_tool_name = None
        if isinstance(forced_tool_choice, dict):
            func_obj = forced_tool_choice.get("function")
            if isinstance(func_obj, dict):
                runtime_forced_tool_name = str(func_obj.get("name") or "").strip() or None

        runtime_snapshot = {
            "tool_router_source": (
                str(tool_router.get("source") or "").strip() if isinstance(tool_router, dict) else None
            ),
            "tool_router_primary_group": runtime_primary_group,
            "tool_router_secondary_groups": runtime_secondary_groups,
            "forced_tool_choice": runtime_forced_tool_name,
            "compare_mode_requested": compare_mode_requested,
            "response_contract_policy": _response_contract_policy_meta(response_contract_request_meta),
            "comment_intent_write_mode": _legacy_comment_intent_to_write_mode(comment_intent_decision),
            "awaiting_reply_state": (
                str(awaiting_reply_state.get("name") or "").strip()
                if isinstance(awaiting_reply_state, dict)
                else None
            ),
            "reply_interpretation_kind": reply_interpretation_runtime_kind,
            "reply_interpretation_action": reply_interpretation_runtime_action,
            "reply_interpretation_applied": reply_interpretation_runtime_applied,
        }
        turn_decision_meta["runtime"] = runtime_snapshot

        if turn_decision_result.mode == "shadow" and turn_decision_result.decision is not None:
            divergence: dict[str, Any] = {}
            if turn_decision_result.decision.tool_plan.primary_group != runtime_primary_group:
                divergence["primary_group"] = {
                    "turn_decision": turn_decision_result.decision.tool_plan.primary_group,
                    "runtime": runtime_primary_group,
                }
            if list(turn_decision_result.decision.tool_plan.secondary_groups) != runtime_secondary_groups:
                divergence["secondary_groups"] = {
                    "turn_decision": list(turn_decision_result.decision.tool_plan.secondary_groups),
                    "runtime": runtime_secondary_groups,
                }
            if turn_decision_result.decision.tool_plan.preferred_first_tool != runtime_forced_tool_name:
                divergence["preferred_first_tool"] = {
                    "turn_decision": turn_decision_result.decision.tool_plan.preferred_first_tool,
                    "runtime": runtime_forced_tool_name,
                }
            runtime_response_mode = "compare" if compare_mode_requested else "single"
            if turn_decision_result.decision.response_plan.mode != runtime_response_mode:
                divergence["response_mode"] = {
                    "turn_decision": turn_decision_result.decision.response_plan.mode,
                    "runtime": runtime_response_mode,
                }
            runtime_contract_cap = _response_contract_policy_meta(response_contract_request_meta).get("cap")
            if turn_decision_result.decision.response_plan.contract_cap != runtime_contract_cap:
                divergence["response_contract_cap"] = {
                    "turn_decision": turn_decision_result.decision.response_plan.contract_cap,
                    "runtime": runtime_contract_cap,
                }
            runtime_write_mode = _legacy_comment_intent_to_write_mode(comment_intent_decision)
            if turn_decision_result.decision.write_plan.mode != runtime_write_mode:
                divergence["write_mode"] = {
                    "turn_decision": turn_decision_result.decision.write_plan.mode,
                    "runtime": runtime_write_mode,
                }
            if classifier_reply_interpretation_kind != reply_interpretation_runtime_kind:
                divergence["reply_interpretation_kind"] = {
                    "turn_decision": classifier_reply_interpretation_kind,
                    "runtime": reply_interpretation_runtime_kind,
                }
            if classifier_reply_interpretation_action != reply_interpretation_runtime_action:
                divergence["reply_interpretation_action"] = {
                    "turn_decision": classifier_reply_interpretation_action,
                    "runtime": reply_interpretation_runtime_action,
                }
            if divergence:
                turn_decision_meta["divergence"] = divergence

    meta: dict[str, Any] = {
        "provider": final_reply.provider,
        "model": final_reply.model,
        "references": {
            "projects": referenced_projects
            if referenced_projects
            else ([focused_project_id] if focused_project_id is not None else []),
        },
    }
    controller_thread_key = support_post_send_thread_key(session.session_id)
    meta["controller"] = {
        "schema_version": 1,
        "source": "support_chat",
        "thread_key": controller_thread_key,
        "pre_send": {
            "action": controller_pre_send.action,
            "stages": [stage.as_meta() for stage in controller_pre_send.stages],
            "response_contract_applied": response_contract_applied,
            "self_review_changed": self_review_changed,
        },
        "tool_selector": tool_selector_source,
        "selected_rule_name": selected_policy_rule_name,
        "prepared_followup_injected": bool(prepared_followup_consumed),
    }
    meta["tooling"] = {
        "enabled": bool(tools_enabled_initial),
        "protocol_config": tool_protocol,
        "schemas_provided": tool_schemas_count > 0,
        "schemas_count": tool_schemas_count,
        "max_tool_calls": max_tool_calls,
        "used_tool_calls": used_tool_calls,
        "forced_tool_choice": forced_tool_choice,
        "requested_tool_names": requested_tool_names_known,
        "missing_requested_tool_names": missing_requested_tool_names,
    }
    meta["context"] = {
        "history_messages_used": len(history_payload),
        "used_conversation_memory": bool(state_for_context.get("conversation_memory")),
        "used_conversation_summary": bool(state_for_context.get("conversation_summary")),
        "context_state_keys": sorted(state_for_context.keys()),
        "time": context_payload.get("time"),
    }
    meta["response_contract"] = response_contract_meta
    if tool_router:
        meta["tool_router"] = tool_router
    if turn_decision_meta is not None:
        meta["turn_decision"] = turn_decision_meta
    if isinstance(awaiting_reply_state, dict) or classifier_reply_interpretation_kind != "none":
        meta["reply_interpretation"] = {
            "awaiting_state": (
                str(awaiting_reply_state.get("name") or "").strip()
                if isinstance(awaiting_reply_state, dict)
                else None
            ),
            "legacy_kind": legacy_reply_interpretation_kind,
            "legacy_action": legacy_reply_interpretation_action,
            "classifier_kind": classifier_reply_interpretation_kind,
            "classifier_action": classifier_reply_interpretation_action,
            "classifier_confidence": (
                turn_decision_result.decision.reply_interpretation.confidence
                if turn_decision_result is not None and turn_decision_result.decision is not None
                else 0.0
            ),
            "classifier_reason": (
                turn_decision_result.decision.reply_interpretation.reason
                if turn_decision_result is not None and turn_decision_result.decision is not None
                else ""
            ),
            "runtime_kind": reply_interpretation_runtime_kind,
            "runtime_action": reply_interpretation_runtime_action,
            "applied": reply_interpretation_runtime_applied,
            "removed_write_tools": reply_interpretation_removed_write_tools,
        }
    if comment_intent_decision is not None or comment_intent_reply is not None:
        meta["comment_intent"] = {
            "enabled": _project_comment_intent_hints_enabled(),
            "decision": comment_intent_decision,
            "provider": getattr(comment_intent_reply, "provider", None),
            "model": getattr(comment_intent_reply, "model", None),
        }
    if final_reply.meta:
        meta["provider_meta"] = final_reply.meta
    meta["tool_calls"] = tool_calls
    prompt_header_configured = prompt_header_enabled()
    if prompt_header_configured or prompt_header_error:
        meta["prompt_header"] = {
            "configured": prompt_header_configured,
            "built": prompt_header is not None,
            "included_round1": prompt_header_included_round1,
            "line": prompt_header.line if prompt_header is not None else None,
            "fields": prompt_header.fields if prompt_header is not None else None,
            "legend_version": prompt_header.legend_version if prompt_header is not None else None,
            "error": prompt_header_error,
        }
    if llm_trace:
        meta["llm_trace"] = llm_trace
    if plan_text:
        meta["plan"] = plan_text
    raw_final = final_raw_reply_content.strip()
    if raw_final and raw_final != assistant_content.strip():
        meta["raw_content"] = raw_final
    if self_review_enabled:
        meta["self_review"] = {
            "changed": self_review_changed,
            "error": self_review_error,
            "mode": self_review_mode,
            "decision": self_review_decision,
        }
        if self_review_changed:
            draft_raw = str(controller_pre_send.draft_raw_content or "").strip()
            if draft_raw and draft_raw != raw_final:
                meta["draft_raw_content"] = draft_raw
    if write_claim_guard_triggered:
        meta["write_claim_guard"] = {
            "triggered": True,
            "retried": write_claim_retry_used,
            "blocked_final_claim": blocked_unsupported_write_claim,
            "available_write_tools": available_write_tool_names,
            "successful_write_tools": sorted(successful_write_tools_final),
        }
    if tool_protocol_guard_triggered:
        meta["tool_protocol_guard"] = {
            "triggered": True,
            "retried": tool_protocol_retry_used,
            "blocked_final_leak": blocked_tool_protocol_leak,
        }

    if compare_choices is not None and final_reply.ok and assistant_content.strip():
        meta["controller"]["post_send"] = {"action": "none", "reason": "compare_pending_choice"}
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
            user_meta["user"] = _support_user_summary(user)
        user_message.meta_json = json.dumps(
            {**_load_state(getattr(user_message, "meta_json", None)), **user_meta},
            ensure_ascii=False,
        )
        session.updated_at = utcnow()
        assistant_db.flush()

        state, enqueued = _maybe_enqueue_session_compaction(
            assistant_db=assistant_db,
            agent_db=agent_db,
            session=session,
            state=state,
            triggered_by_message_id=int(user_message.id),
            reason="compare_choices",
        )
        if enqueued:
            session.state_json = _dump_state(state)
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
        meta_json=None,
    )
    assistant_db.add(assistant_message)
    session.updated_at = utcnow()
    assistant_db.flush()

    state, post_send_meta = _enqueue_support_post_send_prepare(
        assistant_db=assistant_db,
        agent_db=agent_db,
        session=session,
        state=state,
        target_message_id=int(assistant_message.id),
    )
    meta["controller"]["post_send"] = post_send_meta
    assistant_message.meta_json = json.dumps(meta, ensure_ascii=False)
    assistant_db.flush()

    state, enqueued = _maybe_enqueue_session_compaction(
        assistant_db=assistant_db,
        agent_db=agent_db,
        session=session,
        state=state,
        triggered_by_message_id=int(assistant_message.id),
        reason="chat_reply",
    )
    if enqueued:
        session.state_json = _dump_state(state)
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
    agent_db: Session = Depends(get_agent_session_dep),
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
        compare_selection["user"] = _support_user_summary(user)
    assistant_meta = {**assistant_meta, "compare_selection": compare_selection}
    controller_meta = assistant_meta.get("controller") if isinstance(assistant_meta.get("controller"), dict) else {}
    if not isinstance(controller_meta, dict):
        controller_meta = {}
    controller_meta = {
        "schema_version": 1,
        "source": "support_chat",
        "thread_key": support_post_send_thread_key(session.session_id),
        **controller_meta,
    }
    assistant_meta["controller"] = controller_meta

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

    state = _load_state(getattr(session, "state_json", None))
    state, post_send_meta = _enqueue_support_post_send_prepare(
        assistant_db=assistant_db,
        agent_db=agent_db,
        session=session,
        state=state,
        target_message_id=int(assistant_message.id),
    )
    assistant_meta["controller"] = {
        **controller_meta,
        "post_send": post_send_meta,
    }
    assistant_message.meta_json = json.dumps(assistant_meta, ensure_ascii=False) if assistant_meta else None
    assistant_db.flush()

    compare_meta["selected_index"] = int(payload.choiceIndex)
    compare_meta["selected_message_id"] = int(assistant_message.id)
    compare_meta["selected_at"] = utcnow().isoformat()
    user_message.meta_json = json.dumps({**_load_state(meta_raw), "compare": compare_meta}, ensure_ascii=False)
    assistant_db.flush()

    state, enqueued = _maybe_enqueue_session_compaction(
        assistant_db=assistant_db,
        agent_db=agent_db,
        session=session,
        state=state,
        triggered_by_message_id=int(assistant_message.id),
        reason="compare_choice_selected",
    )
    if enqueued:
        session.state_json = _dump_state(state)
        assistant_db.flush()

    return ChooseResponse(
        sessionId=session.session_id,
        messageId=int(assistant_message.id),
        message=assistant_message.content,
    )


def _debug_session_user_map(*, core_db: Session, rows: list[tuple[SupportMessage, SupportSession]]) -> dict[int, dict[str, Any]]:
    session_user_ids = {
        int(session.user_id)
        for _message, session in rows
        if isinstance(session.user_id, int) and session.user_id > 0
    }
    user_map: dict[int, dict[str, Any]] = {}
    if session_user_ids:
        auth_rows = core_db.query(AuthUser).filter(AuthUser.id.in_(session_user_ids)).all()
        for auth_user in auth_rows:
            user_map[int(auth_user.id)] = _support_user_summary(auth_user) or {}
    return user_map


def _build_support_debug_message_item(
    *,
    assistant_db: Session,
    message: SupportMessage,
    session: SupportSession,
    session_user: dict[str, Any] | None,
) -> SupportDebugMessageItem:
    meta = _load_state(getattr(message, "meta_json", None))
    prev_user = (
        assistant_db.query(SupportMessage)
        .filter(SupportMessage.session_pk == session.id)
        .filter(SupportMessage.role == "user")
        .filter(SupportMessage.id < int(message.id))
        .order_by(SupportMessage.id.desc())
        .first()
    )
    prev_meta = _load_state(getattr(prev_user, "meta_json", None) if prev_user is not None else None)
    ui = prev_meta.get("ui") if isinstance(prev_meta.get("ui"), dict) else {}
    tool_calls = meta.get("tool_calls") if isinstance(meta.get("tool_calls"), list) else None
    llm_trace = meta.get("llm_trace") if isinstance(meta.get("llm_trace"), list) else None

    return SupportDebugMessageItem(
        sessionId=session.session_id,
        sessionUser=session_user,
        messageId=int(message.id),
        role=str(message.role or ""),
        message=message.content,
        createdAt=message.created_at,
        provider=message.provider,
        model=message.model,
        uiPath=str(ui.get("path") or "") or None,
        uiName=str(ui.get("name") or "") or None,
        previousUserMessageId=int(prev_user.id) if prev_user is not None else None,
        previousUserMessage=prev_user.content if prev_user is not None else None,
        previousUserMeta=prev_meta or None,
        messageMeta=meta or None,
        toolRouter=meta.get("tool_router") if isinstance(meta.get("tool_router"), dict) else None,
        tooling=meta.get("tooling") if isinstance(meta.get("tooling"), dict) else None,
        toolCalls=tool_calls,
        llmTrace=llm_trace,
        promptHeader=meta.get("prompt_header") if isinstance(meta.get("prompt_header"), dict) else None,
    )


def _build_support_debug_context_messages(
    *,
    assistant_db: Session,
    session_pk: int,
    message: SupportMessage,
    before: int,
    after: int,
) -> list[SupportDebugContextItem]:
    before_rows = (
        assistant_db.query(SupportMessage)
        .filter(SupportMessage.session_pk == session_pk)
        .filter(SupportMessage.id < int(message.id))
        .order_by(SupportMessage.id.desc())
        .limit(before)
        .all()
    )
    after_rows = (
        assistant_db.query(SupportMessage)
        .filter(SupportMessage.session_pk == session_pk)
        .filter(SupportMessage.id > int(message.id))
        .order_by(SupportMessage.id.asc())
        .limit(after)
        .all()
    )
    ordered_rows = list(reversed(before_rows)) + [message] + after_rows
    return [
        SupportDebugContextItem(
            messageId=int(row.id),
            role=str(row.role or ""),
            message=row.content,
            createdAt=row.created_at,
            provider=row.provider,
            model=row.model,
            isSelected=int(row.id) == int(message.id),
        )
        for row in ordered_rows
    ]


@router.get("/messages", response_model=list[SupportDebugMessageItem])
def list_messages(
    response: Response,
    role: str | None = None,
    session_id: str | None = Query(default=None, alias="sessionId"),
    limit: int = Query(default=200, ge=1, le=2000),
    offset: int = Query(default=0, ge=0),
    assistant_db: Session = Depends(get_assistant_session_dep),
    core_db: Session = Depends(get_session_dep),
    user: AuthUser | None = Depends(require_access),
):
    if user is None or user.role not in {UserRole.admin, UserRole.editor}:
        raise HTTPException(status_code=403, detail="Admin or editor access required.")

    role_filter = str(role or "").strip().lower() or None
    if role_filter and role_filter not in {"user", "assistant", "system"}:
        raise HTTPException(status_code=400, detail="role must be one of user, assistant, system")

    query = assistant_db.query(SupportMessage, SupportSession).join(
        SupportSession, SupportMessage.session_pk == SupportSession.id
    )
    if session_id:
        query = query.filter(SupportSession.session_id == session_id)
    if role_filter:
        query = query.filter(SupportMessage.role == role_filter)

    total = int(query.count())
    rows = (
        query.order_by(SupportMessage.created_at.desc(), SupportMessage.id.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )
    response.headers["X-Total-Count"] = str(total)

    user_map = _debug_session_user_map(core_db=core_db, rows=rows)

    payload: list[SupportDebugMessageItem] = []
    for message, session in rows:
        session_user = user_map.get(int(session.user_id)) if isinstance(session.user_id, int) else None
        payload.append(
            _build_support_debug_message_item(
                assistant_db=assistant_db,
                message=message,
                session=session,
                session_user=session_user,
            )
        )

    return payload


@router.get("/messages/{message_id}", response_model=SupportDebugMessageDetail)
def get_message_detail(
    message_id: int,
    before: int = Query(default=6, ge=0, le=50),
    after: int = Query(default=6, ge=0, le=50),
    assistant_db: Session = Depends(get_assistant_session_dep),
    core_db: Session = Depends(get_session_dep),
    user: AuthUser | None = Depends(require_access),
):
    if user is None or user.role not in {UserRole.admin, UserRole.editor}:
        raise HTTPException(status_code=403, detail="Admin or editor access required.")

    row = (
        assistant_db.query(SupportMessage, SupportSession)
        .join(SupportSession, SupportMessage.session_pk == SupportSession.id)
        .filter(SupportMessage.id == message_id)
        .first()
    )
    if row is None:
        raise HTTPException(status_code=404, detail="Message not found.")

    message, session = row
    user_map = _debug_session_user_map(core_db=core_db, rows=[row])
    session_user = user_map.get(int(session.user_id)) if isinstance(session.user_id, int) else None
    item = _build_support_debug_message_item(
        assistant_db=assistant_db,
        message=message,
        session=session,
        session_user=session_user,
    )
    context_messages = _build_support_debug_context_messages(
        assistant_db=assistant_db,
        session_pk=int(session.id),
        message=message,
        before=int(before),
        after=int(after),
    )
    session_message_count = int(
        assistant_db.query(func.count(SupportMessage.id))
        .filter(SupportMessage.session_pk == int(session.id))
        .scalar()
        or 0
    )
    session_state = _load_state(getattr(session, "state_json", None)) or None

    return SupportDebugMessageDetail(
        **item.model_dump(),
        sessionState=session_state,
        sessionMessageCount=session_message_count,
        contextMessages=context_messages,
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
        feedback_meta["user"] = _support_user_summary(user)
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
