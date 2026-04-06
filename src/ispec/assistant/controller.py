from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Callable, Literal

from sqlalchemy.orm import Session

from ispec.agent.commands import COMMAND_POST_SEND_PREPARE
from ispec.agent.models import AgentCommand
from ispec.assistant.formatting import split_plan_final
from ispec.assistant.response_contracts import (
    ResponseContractMode,
    ResponseContractResult,
    run_response_contract_pipeline,
)
from ispec.assistant.service import (
    AssistantReply,
    _system_prompt_review,
    _system_prompt_review_decider,
)
from ispec.assistant.tools import parse_tool_call


ArtifactSource = Literal["support_chat", "scheduled_assistant"]
PreSendAction = Literal["pass_through", "review_before_send"]
PostSendAction = Literal["none", "enqueue_post_send_prepare"]
ControllerStageStatus = Literal["applied", "skipped", "error"]
ControllerStageName = Literal["response_contract", "self_review"]


def utcnow() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True)
class ControllerStageResult:
    name: ControllerStageName
    status: ControllerStageStatus
    reason: str | None = None
    changed: bool = False
    meta: dict[str, Any] = field(default_factory=dict)

    def as_meta(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "reason": self.reason,
            "changed": self.changed,
            "meta": dict(self.meta or {}),
        }


@dataclass(frozen=True)
class ControllerPreSendResult:
    action: PreSendAction
    final_content: str
    final_reply: AssistantReply
    final_raw_content: str
    response_contract_meta: dict[str, Any]
    response_contract_applied: bool
    self_review_changed: bool
    self_review_error: str | None
    self_review_mode: str | None
    self_review_decision: str | None
    draft_raw_content: str | None = None
    stages: tuple[ControllerStageResult, ...] = ()


@dataclass(frozen=True)
class ControllerPostSendEnqueueResult:
    action: PostSendAction
    enqueued: bool
    thread_key: str
    command_id: int | None = None
    existing_command_id: int | None = None
    reason: str | None = None

    def as_meta(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "enqueued": self.enqueued,
            "thread_key": self.thread_key,
            "command_id": self.command_id,
            "existing_command_id": self.existing_command_id,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class ControllerContext:
    generate_reply_fn: Callable[..., AssistantReply]
    source: ArtifactSource
    context_message: str
    history_messages: tuple[dict[str, Any], ...]
    user_message: str
    tool_messages: tuple[dict[str, Any], ...]
    tool_result_messages: tuple[dict[str, Any], ...]
    draft_answer: str
    draft_reply: AssistantReply
    compare_mode: bool
    request_meta: dict[str, Any] | None
    response_contract_mode: ResponseContractMode
    response_contract_would_apply_if_live: bool
    response_contract_protection_reason: str | None
    self_review_enabled: bool
    self_review_decider_enabled: bool
    used_tool_calls: int


@dataclass
class ControllerState:
    content: str
    reply: AssistantReply
    raw_content: str
    response_contract_meta: dict[str, Any]
    response_contract_applied: bool = False
    self_review_changed: bool = False
    self_review_error: str | None = None
    self_review_mode: str | None = None
    self_review_decision: str | None = None
    draft_raw_content: str | None = None


@dataclass(frozen=True)
class SourceControllerPolicy:
    pre_send_stages: tuple[ControllerStageName, ...]


_SOURCE_CONTROLLER_POLICIES: dict[ArtifactSource, SourceControllerPolicy] = {
    "support_chat": SourceControllerPolicy(pre_send_stages=("response_contract", "self_review")),
    "scheduled_assistant": SourceControllerPolicy(pre_send_stages=("response_contract",)),
}


def support_post_send_thread_key(session_id: str) -> str:
    return f"support:{str(session_id or '').strip()}"


def scheduled_post_send_thread_key(
    *,
    schedule_key: str | None,
    job_name: str | None,
    command_id: int | None,
) -> str:
    schedule_text = str(schedule_key or "").strip()
    if schedule_text:
        return f"scheduled:{schedule_text}"
    job_text = str(job_name or "").strip() or "unnamed"
    if isinstance(command_id, int) and command_id > 0:
        return f"scheduled:{job_text}:{command_id}"
    return f"scheduled:{job_text}"


def enqueue_post_send_prepare_command(
    *,
    agent_db: Session,
    source: ArtifactSource,
    thread_key: str,
    payload: dict[str, Any],
    priority: int = 0,
    current_command_id: int | None = None,
) -> ControllerPostSendEnqueueResult:
    normalized_thread_key = str(thread_key or "").strip()
    if not normalized_thread_key:
        return ControllerPostSendEnqueueResult(
            action="none",
            enqueued=False,
            thread_key="",
            reason="missing_thread_key",
        )

    rows = (
        agent_db.query(AgentCommand)
        .filter(AgentCommand.command_type == COMMAND_POST_SEND_PREPARE)
        .filter(AgentCommand.status.in_(["queued", "running"]))
        .order_by(AgentCommand.id.desc())
        .all()
    )
    for row in rows:
        if current_command_id is not None and int(row.id) == int(current_command_id):
            continue
        row_payload = dict(row.payload_json or {})
        row_source = str(row_payload.get("source") or "").strip()
        row_thread_key = str(row_payload.get("thread_key") or "").strip()
        if row_source == source and row_thread_key == normalized_thread_key:
            return ControllerPostSendEnqueueResult(
                action="enqueue_post_send_prepare",
                enqueued=False,
                thread_key=normalized_thread_key,
                existing_command_id=int(row.id),
                reason="duplicate_inflight",
            )

    full_payload = {
        "schema_version": 1,
        "source": source,
        "thread_key": normalized_thread_key,
        **dict(payload or {}),
    }
    cmd = AgentCommand(
        command_type=COMMAND_POST_SEND_PREPARE,
        status="queued",
        priority=int(priority),
        payload_json=full_payload,
        result_json={},
    )
    agent_db.add(cmd)
    agent_db.flush()
    return ControllerPostSendEnqueueResult(
        action="enqueue_post_send_prepare",
        enqueued=True,
        thread_key=normalized_thread_key,
        command_id=int(cmd.id),
    )


def _response_contract_stage(ctx: ControllerContext, state: ControllerState) -> ControllerStageResult:
    if ctx.response_contract_mode != "shadow":
        return ControllerStageResult(name="response_contract", status="skipped", reason="disabled")
    if ctx.compare_mode:
        state.response_contract_meta["skipped_reason"] = "compare_mode"
        return ControllerStageResult(name="response_contract", status="skipped", reason="compare_mode")
    if not ctx.draft_reply.ok:
        state.response_contract_meta["skipped_reason"] = "draft_reply_error"
        return ControllerStageResult(name="response_contract", status="skipped", reason="draft_reply_error")
    if not state.content.strip():
        state.response_contract_meta["skipped_reason"] = "empty_draft"
        return ControllerStageResult(name="response_contract", status="skipped", reason="empty_draft")

    result: ResponseContractResult = run_response_contract_pipeline(
        generate_reply_fn=ctx.generate_reply_fn,
        context_message=ctx.context_message,
        history_messages=list(ctx.history_messages),
        user_message=ctx.user_message,
        tool_result_messages=list(ctx.tool_result_messages),
        draft_answer=state.content,
        request_meta=ctx.request_meta,
    )
    state.response_contract_meta = result.as_meta()
    state.response_contract_meta["enabled"] = True
    state.response_contract_meta["applied"] = False
    state.response_contract_meta["configured_mode"] = ctx.response_contract_mode
    state.response_contract_meta["would_apply_if_live"] = bool(ctx.response_contract_would_apply_if_live)
    state.response_contract_meta["protection_reason"] = ctx.response_contract_protection_reason
    if result.ok and result.rendered_content:
        changed = result.rendered_content.strip() != state.content.strip()
        state.response_contract_meta["shadow_candidate"] = result.rendered_content
        return ControllerStageResult(
            name="response_contract",
            status="applied",
            changed=changed,
            meta={
                "mode": ctx.response_contract_mode,
                "selected_contract": result.selected_contract,
                "repair_applied": bool(result.repair_applied),
                "would_apply_if_live": bool(ctx.response_contract_would_apply_if_live),
                "protection_reason": ctx.response_contract_protection_reason,
            },
        )
    reason = result.skipped_reason or (result.errors[0] if result.errors else "contract_pipeline_failed")
    return ControllerStageResult(
        name="response_contract",
        status="error" if result.errors else "skipped",
        reason=reason,
        meta={
            "mode": ctx.response_contract_mode,
            "selected_contract": result.selected_contract,
            "would_apply_if_live": bool(ctx.response_contract_would_apply_if_live),
            "protection_reason": ctx.response_contract_protection_reason,
        },
    )


def _self_review_should_run(ctx: ControllerContext, state: ControllerState) -> tuple[bool, str | None]:
    if not ctx.self_review_enabled:
        return False, "disabled"
    if ctx.source != "support_chat":
        return False, "source_not_supported"
    if state.response_contract_applied:
        state.self_review_mode = "skipped_response_contract"
        return False, "response_contract_applied"
    if not ctx.draft_reply.ok:
        return False, "draft_reply_error"
    if not state.content.strip():
        return False, "empty_draft"
    if ctx.used_tool_calls <= 0:
        state.self_review_mode = "skipped_no_tool_calls"
        return False, "no_tool_calls"
    return True, None


def _self_review_decider(ctx: ControllerContext, *, assistant_content: str) -> tuple[str | None, str | None]:
    if not ctx.self_review_decider_enabled:
        return None, None

    decider_instruction = (
        "Decide if the draft answer needs changes. "
        "Output only KEEP (no changes) or REWRITE (needs changes)."
    )
    decider_reply = ctx.generate_reply_fn(
        messages=[
            {"role": "system", "content": _system_prompt_review_decider()},
            {"role": "system", "content": ctx.context_message},
            *list(ctx.history_messages),
            {"role": "user", "content": ctx.user_message},
            *list(ctx.tool_messages),
            {"role": "assistant", "content": assistant_content},
            {"role": "user", "content": decider_instruction},
        ],
        tools=None,
        vllm_extra_body={
            "structured_outputs": {"choice": ["KEEP", "REWRITE"]},
            "max_tokens": 3,
            "stop": ["\n"],
            "temperature": 0,
        },
    )
    if not decider_reply.ok:
        return "keep", "review_decider_error"

    decider_text = (decider_reply.content or "").strip().upper()
    if decider_text.startswith("KEEP"):
        return "keep", None
    if decider_text.startswith("REWRITE"):
        return "rewrite", None
    return None, "review_decider_invalid_output"


def _self_review_stage(ctx: ControllerContext, state: ControllerState) -> ControllerStageResult:
    should_run, skip_reason = _self_review_should_run(ctx, state)
    if not should_run:
        return ControllerStageResult(name="self_review", status="skipped", reason=skip_reason)

    state.self_review_mode = "rewrite"
    if ctx.self_review_decider_enabled:
        state.self_review_mode = "structured_choice"
        decision, error = _self_review_decider(ctx, assistant_content=state.content)
        state.self_review_decision = decision
        if error:
            state.self_review_error = error
            if error == "review_decider_error":
                return ControllerStageResult(
                    name="self_review",
                    status="skipped",
                    reason=error,
                    meta={"mode": state.self_review_mode, "decision": "keep"},
                )
        if decision == "keep":
            return ControllerStageResult(
                name="self_review",
                status="skipped",
                reason="decider_keep",
                meta={"mode": state.self_review_mode, "decision": decision},
            )

    review_instruction = (
        "Review the assistant answer above for correctness (grounded in CONTEXT/TOOL_RESULT), "
        "clarity, and iSPEC tone. If it's already good, repeat it verbatim. "
        "If it needs changes, rewrite it. Do not call tools.\n"
        "Output only:\nFINAL:\n<answer>"
    )
    review_reply = ctx.generate_reply_fn(
        messages=[
            {"role": "system", "content": _system_prompt_review()},
            {"role": "system", "content": ctx.context_message},
            *list(ctx.history_messages),
            {"role": "user", "content": ctx.user_message},
            *list(ctx.tool_messages),
            {"role": "assistant", "content": state.content},
            {"role": "user", "content": review_instruction},
        ],
        tools=None,
    )

    if not review_reply.ok:
        state.self_review_error = "review_error"
        return ControllerStageResult(name="self_review", status="error", reason="review_error")

    review_tool_call = parse_tool_call(review_reply.content)
    if review_tool_call is not None or review_reply.tool_calls:
        state.self_review_error = "review_requested_tool_call"
        return ControllerStageResult(
            name="self_review",
            status="error",
            reason="review_requested_tool_call",
            meta={"mode": state.self_review_mode},
        )

    review_raw = (review_reply.content or "").strip()
    _review_plan_text, review_final_content = split_plan_final(review_raw)
    reviewed_content = (review_final_content or review_raw).strip()
    if not reviewed_content:
        state.self_review_error = "empty_review_output"
        return ControllerStageResult(name="self_review", status="error", reason="empty_review_output")

    changed = reviewed_content != state.content.strip()
    if changed:
        state.self_review_changed = True
        state.draft_raw_content = state.raw_content or state.content.strip()
        state.content = reviewed_content
        state.reply = review_reply
        state.raw_content = review_raw
    return ControllerStageResult(
        name="self_review",
        status="applied",
        changed=changed,
        meta={"mode": state.self_review_mode, "decision": state.self_review_decision},
    )


_PRE_SEND_STAGE_RUNNERS: dict[ControllerStageName, Callable[[ControllerContext, ControllerState], ControllerStageResult]] = {
    "response_contract": _response_contract_stage,
    "self_review": _self_review_stage,
}


def run_message_pre_send_controller(
    *,
    generate_reply_fn: Callable[..., AssistantReply],
    source: ArtifactSource,
    context_message: str,
    history_messages: list[dict[str, Any]],
    user_message: str,
    tool_messages: list[dict[str, Any]],
    tool_result_messages: list[dict[str, Any]],
    draft_answer: str,
    draft_reply: AssistantReply,
    compare_mode: bool,
    request_meta: dict[str, Any] | None,
    response_contract_mode: ResponseContractMode,
    response_contract_would_apply_if_live: bool,
    response_contract_protection_reason: str | None,
    self_review_enabled: bool,
    self_review_decider_enabled: bool,
    used_tool_calls: int,
) -> ControllerPreSendResult:
    ctx = ControllerContext(
        generate_reply_fn=generate_reply_fn,
        source=source,
        context_message=context_message,
        history_messages=tuple(history_messages),
        user_message=user_message,
        tool_messages=tuple(tool_messages),
        tool_result_messages=tuple(tool_result_messages),
        draft_answer=(draft_answer or "").strip(),
        draft_reply=draft_reply,
        compare_mode=compare_mode,
        request_meta=request_meta,
        response_contract_mode=response_contract_mode,
        response_contract_would_apply_if_live=bool(response_contract_would_apply_if_live),
        response_contract_protection_reason=response_contract_protection_reason,
        self_review_enabled=bool(self_review_enabled),
        self_review_decider_enabled=bool(self_review_decider_enabled),
        used_tool_calls=int(used_tool_calls),
    )
    state = ControllerState(
        content=ctx.draft_answer,
        reply=draft_reply,
        raw_content=(draft_reply.content or "").strip(),
        response_contract_meta={
            "enabled": ctx.response_contract_mode == "shadow",
            "configured_mode": ctx.response_contract_mode,
            "applied": False,
            "would_apply_if_live": bool(ctx.response_contract_would_apply_if_live),
            "protection_reason": ctx.response_contract_protection_reason,
        },
    )
    action: PreSendAction = (
        "review_before_send"
        if (
            not compare_mode
            and draft_reply.ok
            and bool(ctx.draft_answer)
            and (ctx.response_contract_mode == "shadow" or self_review_enabled)
        )
        else "pass_through"
    )

    stage_results: list[ControllerStageResult] = []
    policy = _SOURCE_CONTROLLER_POLICIES.get(source, SourceControllerPolicy(pre_send_stages=()))
    for stage_name in policy.pre_send_stages:
        runner = _PRE_SEND_STAGE_RUNNERS.get(stage_name)
        if runner is None:
            continue
        stage_results.append(runner(ctx, state))

    return ControllerPreSendResult(
        action=action,
        final_content=state.content,
        final_reply=state.reply,
        final_raw_content=state.raw_content,
        response_contract_meta=state.response_contract_meta,
        response_contract_applied=state.response_contract_applied,
        self_review_changed=state.self_review_changed,
        self_review_error=state.self_review_error,
        self_review_mode=state.self_review_mode,
        self_review_decision=state.self_review_decision,
        draft_raw_content=state.draft_raw_content,
        stages=tuple(stage_results),
    )
