from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

from ispec.assistant.classifier_service import generate_classifier_reply
from ispec.assistant.json_utils import parse_json_object
from ispec.assistant.response_contracts import ResponseContractName, response_contract_names
from ispec.assistant.service import AssistantReply
from ispec.assistant.tool_routing import ToolGroup, tool_names_for_groups
from ispec.prompt import load_bound_prompt, prompt_binding, prompt_observability_context


TurnDecisionMode = Literal["off", "shadow", "own"]
TurnDecisionSource = Literal["support_chat", "scheduled_assistant"]
PrimaryGoal = Literal[
    "answer_question",
    "inspect_state",
    "draft_project_comment",
    "save_project_comment",
    "confirm_save",
    "automation_task",
    "devops_task",
]
ClarificationReason = Literal[
    "none",
    "missing_identifier",
    "missing_comment_text",
    "ambiguous_target",
    "missing_confirmation",
]
WritePlanMode = Literal["none", "draft_only", "save_now", "confirm_save"]
ResponseMode = Literal["single", "compare"]
ReplyInterpretationKind = Literal["none", "approve", "deny", "defer", "modify", "unclear"]

_PRIMARY_GOALS: tuple[PrimaryGoal, ...] = (
    "answer_question",
    "inspect_state",
    "draft_project_comment",
    "save_project_comment",
    "confirm_save",
    "automation_task",
    "devops_task",
)
_CLARIFICATION_REASONS: tuple[ClarificationReason, ...] = (
    "none",
    "missing_identifier",
    "missing_comment_text",
    "ambiguous_target",
    "missing_confirmation",
)
_WRITE_PLAN_MODES: tuple[WritePlanMode, ...] = (
    "none",
    "draft_only",
    "save_now",
    "confirm_save",
)
_RESPONSE_MODES: tuple[ResponseMode, ...] = ("single", "compare")
_REPLY_INTERPRETATION_KINDS: tuple[ReplyInterpretationKind, ...] = (
    "none",
    "approve",
    "deny",
    "defer",
    "modify",
    "unclear",
)


@dataclass(frozen=True)
class TurnDecisionToolPlan:
    use_tools: bool
    primary_group: str | None
    secondary_groups: tuple[str, ...]
    preferred_first_tool: str | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "use_tools": self.use_tools,
            "primary_group": self.primary_group,
            "secondary_groups": list(self.secondary_groups),
            "preferred_first_tool": self.preferred_first_tool,
        }


@dataclass(frozen=True)
class TurnDecisionWritePlan:
    mode: WritePlanMode

    def as_dict(self) -> dict[str, Any]:
        return {"mode": self.mode}


@dataclass(frozen=True)
class TurnDecisionResponsePlan:
    mode: ResponseMode
    contract_cap: ResponseContractName | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "contract_cap": self.contract_cap,
        }


@dataclass(frozen=True)
class TurnDecisionReplyInterpretation:
    kind: ReplyInterpretationKind
    confidence: float
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "confidence": self.confidence,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class TurnDecision:
    source: TurnDecisionSource
    primary_goal: PrimaryGoal
    needs_clarification: bool
    clarification_reason: ClarificationReason
    tool_plan: TurnDecisionToolPlan
    write_plan: TurnDecisionWritePlan
    response_plan: TurnDecisionResponsePlan
    reply_interpretation: TurnDecisionReplyInterpretation
    confidence: float
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "primary_goal": self.primary_goal,
            "needs_clarification": self.needs_clarification,
            "clarification_reason": self.clarification_reason,
            "tool_plan": self.tool_plan.as_dict(),
            "write_plan": self.write_plan.as_dict(),
            "response_plan": self.response_plan.as_dict(),
            "reply_interpretation": self.reply_interpretation.as_dict(),
            "confidence": self.confidence,
            "reason": self.reason,
        }


@dataclass
class TurnDecisionResult:
    ok: bool
    mode: TurnDecisionMode
    source: TurnDecisionSource
    applied: bool = False
    decision: TurnDecision | None = None
    raw_decision: dict[str, Any] | None = None
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    skipped_reason: str | None = None
    reply_meta: dict[str, Any] | None = None

    def as_meta(self) -> dict[str, Any]:
        return {
            "enabled": self.mode != "off",
            "mode": self.mode,
            "source": self.source,
            "applied": self.applied,
            "ok": self.ok,
            "decision": self.decision.as_dict() if self.decision is not None else None,
            "raw_decision": self.raw_decision,
            "warnings": list(self.warnings),
            "errors": list(self.errors),
            "skipped_reason": self.skipped_reason,
            "reply": self.reply_meta,
        }


def parse_turn_decision_mode(raw: str | None, *, auto_shadow: bool) -> TurnDecisionMode:
    text = str(raw or "").strip().lower()
    if not text or text == "auto":
        return "shadow" if auto_shadow else "off"
    if text in {"0", "false", "no", "off"}:
        return "off"
    if text in {"1", "true", "yes", "on", "shadow"}:
        return "shadow"
    if text == "own":
        return "own"
    return "off"


def turn_decision_auto_shadow(*, assistant_provider: str, state_dir: str | None) -> bool:
    if str(assistant_provider or "").strip().lower() != "vllm":
        return False
    raw = str(state_dir or "").strip()
    if not raw:
        return False
    try:
        path = Path(raw).expanduser().resolve()
    except Exception:
        return False
    return path.name == ".pids"


def turn_decision_schema(
    *,
    group_names: list[str],
    tool_names: list[str],
    response_modes: list[ResponseMode],
    contract_caps: list[ResponseContractName],
) -> dict[str, Any]:
    primary_group_enum = [""] + [name for name in group_names if name]
    tool_enum = [""] + [name for name in tool_names if name]
    contract_enum = [""] + [name for name in contract_caps if name]
    mode_enum = [mode for mode in response_modes if mode in _RESPONSE_MODES]
    if not mode_enum:
        mode_enum = ["single"]

    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "source": {"type": "string", "enum": ["support_chat", "scheduled_assistant"]},
            "primary_goal": {"type": "string", "enum": list(_PRIMARY_GOALS)},
            "needs_clarification": {"type": "boolean"},
            "clarification_reason": {"type": "string", "enum": list(_CLARIFICATION_REASONS)},
            "tool_plan": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "use_tools": {"type": "boolean"},
                    "primary_group": {"type": "string", "enum": primary_group_enum},
                    "secondary_groups": {
                        "type": "array",
                        "items": {"type": "string", "enum": group_names},
                        "minItems": 0,
                        "maxItems": 2,
                    },
                    "preferred_first_tool": {"type": "string", "enum": tool_enum},
                },
                "required": ["use_tools", "primary_group", "secondary_groups", "preferred_first_tool"],
            },
            "write_plan": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "mode": {"type": "string", "enum": list(_WRITE_PLAN_MODES)},
                },
                "required": ["mode"],
            },
            "response_plan": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "mode": {"type": "string", "enum": mode_enum},
                    "contract_cap": {"type": "string", "enum": contract_enum},
                },
                "required": ["mode", "contract_cap"],
            },
            "reply_interpretation": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "kind": {"type": "string", "enum": list(_REPLY_INTERPRETATION_KINDS)},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "reason": {"type": "string"},
                },
                "required": ["kind", "confidence", "reason"],
            },
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "reason": {"type": "string"},
        },
        "required": [
            "source",
            "primary_goal",
            "needs_clarification",
            "clarification_reason",
            "tool_plan",
            "write_plan",
            "response_plan",
            "reply_interpretation",
            "confidence",
            "reason",
        ],
    }


@prompt_binding("assistant.turn_decision.classifier")
def _turn_decision_prompt(
    *,
    source: TurnDecisionSource,
    groups: list[ToolGroup],
    response_modes: list[ResponseMode],
    contract_caps: list[ResponseContractName],
) -> str:
    values = _turn_decision_prompt_values(
        source=source,
        groups=groups,
        response_modes=response_modes,
        contract_caps=contract_caps,
    )
    return load_bound_prompt(
        _turn_decision_prompt,
        values=values,
    ).text


def _turn_decision_prompt_values(
    *,
    source: TurnDecisionSource,
    groups: list[ToolGroup],
    response_modes: list[ResponseMode],
    contract_caps: list[ResponseContractName],
) -> dict[str, str]:
    return {
        "scheduled_rules_block": (
            "\n\nScheduled-assistant rules:\n"
            "- This is not an end-user conversation.\n"
            "- needs_clarification must be false.\n"
            "- clarification_reason must be none.\n"
            "- response_plan.mode must be single."
            if source == "scheduled_assistant"
            else ""
        ),
        "groups_block": (
            "\n\nAvailable tool groups:\n" + "\n".join(f"- {group.name}: {group.description}" for group in groups)
            if groups
            else ""
        ),
        "response_modes_block": ("\n\nAllowed response modes: " + ", ".join(response_modes)) if response_modes else "",
        "contract_caps_block": ("\nAllowed response contract caps: " + ", ".join(contract_caps)) if contract_caps else "",
    }


def _clamp_confidence(raw: Any) -> float:
    try:
        value = float(raw)
    except Exception:
        value = 0.0
    return max(0.0, min(1.0, value))


def _normalize_write_mode(*, primary_goal: PrimaryGoal, raw_mode: Any, warnings: list[str]) -> WritePlanMode:
    if primary_goal == "draft_project_comment":
        if raw_mode != "draft_only":
            warnings.append("write_plan_normalized_to_draft_only")
        return "draft_only"
    if primary_goal == "save_project_comment":
        if raw_mode != "save_now":
            warnings.append("write_plan_normalized_to_save_now")
        return "save_now"
    if primary_goal == "confirm_save":
        if raw_mode != "confirm_save":
            warnings.append("write_plan_normalized_to_confirm_save")
        return "confirm_save"
    if raw_mode not in _WRITE_PLAN_MODES:
        if raw_mode not in (None, ""):
            warnings.append("invalid_write_plan_mode")
        return "none"
    return raw_mode


def _normalize_reply_interpretation(
    *,
    raw_reply_interpretation: Any,
    awaiting_reply_state: dict[str, Any] | None,
    source: TurnDecisionSource,
    warnings: list[str],
) -> TurnDecisionReplyInterpretation:
    raw = raw_reply_interpretation if isinstance(raw_reply_interpretation, dict) else {}
    raw_kind = str(raw.get("kind") or "").strip()
    raw_reason = str(raw.get("reason") or "").strip()
    confidence = _clamp_confidence(raw.get("confidence"))

    if source == "scheduled_assistant":
        if raw_kind not in {"", "none"}:
            warnings.append("reply_interpretation_forced_none_for_scheduled_assistant")
        return TurnDecisionReplyInterpretation(kind="none", confidence=0.0, reason=raw_reason)

    if awaiting_reply_state is None:
        if raw_kind not in {"", "none"}:
            warnings.append("reply_interpretation_ignored_without_awaiting_state")
        return TurnDecisionReplyInterpretation(kind="none", confidence=confidence, reason=raw_reason)

    if raw_kind not in _REPLY_INTERPRETATION_KINDS:
        if raw_kind:
            warnings.append("invalid_reply_interpretation_kind")
        return TurnDecisionReplyInterpretation(kind="unclear", confidence=confidence, reason=raw_reason)

    if raw_kind == "none":
        warnings.append("reply_interpretation_normalized_to_unclear")
        return TurnDecisionReplyInterpretation(kind="unclear", confidence=confidence, reason=raw_reason)

    return TurnDecisionReplyInterpretation(
        kind=raw_kind,  # type: ignore[arg-type]
        confidence=confidence,
        reason=raw_reason,
    )


def _validate_turn_decision(
    *,
    raw_decision: dict[str, Any],
    source: TurnDecisionSource,
    groups: list[ToolGroup],
    response_modes: list[ResponseMode],
    contract_caps: list[ResponseContractName],
    awaiting_reply_state: dict[str, Any] | None,
) -> tuple[TurnDecision | None, list[str], list[str]]:
    warnings: list[str] = []
    errors: list[str] = []
    group_names = [group.name for group in groups if group.name]
    allowed_group_names = set(group_names)
    group_by_tool: dict[str, str] = {}
    for group in groups:
        for tool_name in group.tool_names:
            if tool_name and tool_name not in group_by_tool:
                group_by_tool[tool_name] = group.name
    allowed_tool_names = set(group_by_tool)

    primary_goal_raw = raw_decision.get("primary_goal")
    if primary_goal_raw not in _PRIMARY_GOALS:
        errors.append("invalid_primary_goal")
        return None, warnings, errors
    primary_goal: PrimaryGoal = primary_goal_raw

    needs_clarification = bool(raw_decision.get("needs_clarification"))
    clarification_reason_raw = raw_decision.get("clarification_reason")
    if clarification_reason_raw not in _CLARIFICATION_REASONS:
        warnings.append("invalid_clarification_reason")
        clarification_reason: ClarificationReason = "none"
    else:
        clarification_reason = clarification_reason_raw
    if not needs_clarification:
        clarification_reason = "none"

    tool_plan_raw = raw_decision.get("tool_plan")
    if not isinstance(tool_plan_raw, dict):
        errors.append("missing_tool_plan")
        return None, warnings, errors
    use_tools = bool(tool_plan_raw.get("use_tools"))
    primary_group_raw = str(tool_plan_raw.get("primary_group") or "").strip()
    primary_group = primary_group_raw if primary_group_raw in allowed_group_names else None
    if primary_group_raw and primary_group is None:
        warnings.append("invalid_primary_group")

    secondary_groups_raw = tool_plan_raw.get("secondary_groups")
    secondary_groups: list[str] = []
    if isinstance(secondary_groups_raw, list):
        for item in secondary_groups_raw:
            if isinstance(item, str) and item in allowed_group_names and item != primary_group and item not in secondary_groups:
                secondary_groups.append(item)
    elif secondary_groups_raw not in (None, []):
        warnings.append("invalid_secondary_groups")
    secondary_groups = secondary_groups[:2]

    preferred_first_tool_raw = str(tool_plan_raw.get("preferred_first_tool") or "").strip()
    preferred_first_tool = preferred_first_tool_raw if preferred_first_tool_raw in allowed_tool_names else None
    if preferred_first_tool_raw and preferred_first_tool is None:
        warnings.append("invalid_preferred_first_tool")

    inferred_primary_group = group_by_tool.get(preferred_first_tool or "")
    if primary_group is None and inferred_primary_group:
        primary_group = inferred_primary_group
        warnings.append("inferred_primary_group_from_preferred_tool")

    if not use_tools and (primary_group or secondary_groups or preferred_first_tool):
        warnings.append("tool_plan_normalized_to_use_tools")
        use_tools = True
    if use_tools and primary_group is None and not preferred_first_tool:
        warnings.append("tool_plan_normalized_to_no_tools")
        use_tools = False
        secondary_groups = []
    if not use_tools:
        primary_group = None
        secondary_groups = []
        preferred_first_tool = None

    write_plan_raw = raw_decision.get("write_plan")
    if not isinstance(write_plan_raw, dict):
        errors.append("missing_write_plan")
        return None, warnings, errors
    write_mode = _normalize_write_mode(
        primary_goal=primary_goal,
        raw_mode=write_plan_raw.get("mode"),
        warnings=warnings,
    )

    response_plan_raw = raw_decision.get("response_plan")
    if not isinstance(response_plan_raw, dict):
        errors.append("missing_response_plan")
        return None, warnings, errors
    allowed_response_modes = [mode for mode in response_modes if mode in _RESPONSE_MODES] or ["single"]
    response_mode_raw = response_plan_raw.get("mode")
    response_mode: ResponseMode = (
        response_mode_raw if response_mode_raw in allowed_response_modes else allowed_response_modes[0]
    )
    if response_mode_raw not in allowed_response_modes:
        warnings.append("invalid_response_mode")

    contract_cap_raw = str(response_plan_raw.get("contract_cap") or "").strip()
    contract_cap: ResponseContractName | None = None
    if contract_cap_raw:
        if contract_cap_raw in contract_caps:
            contract_cap = contract_cap_raw  # type: ignore[assignment]
        else:
            warnings.append("invalid_contract_cap")

    if source == "scheduled_assistant":
        if needs_clarification:
            warnings.append("scheduled_assistant_forced_no_clarification")
        needs_clarification = False
        clarification_reason = "none"
        if response_mode != "single":
            warnings.append("scheduled_assistant_forced_single_mode")
        response_mode = "single"

    reply_interpretation = _normalize_reply_interpretation(
        raw_reply_interpretation=raw_decision.get("reply_interpretation"),
        awaiting_reply_state=awaiting_reply_state,
        source=source,
        warnings=warnings,
    )

    confidence = _clamp_confidence(raw_decision.get("confidence"))
    reason = str(raw_decision.get("reason") or "").strip()
    if not reason:
        warnings.append("empty_reason")

    return (
        TurnDecision(
            source=source,
            primary_goal=primary_goal,
            needs_clarification=needs_clarification,
            clarification_reason=clarification_reason,
            tool_plan=TurnDecisionToolPlan(
                use_tools=use_tools,
                primary_group=primary_group,
                secondary_groups=tuple(secondary_groups),
                preferred_first_tool=preferred_first_tool,
            ),
            write_plan=TurnDecisionWritePlan(mode=write_mode),
            response_plan=TurnDecisionResponsePlan(
                mode=response_mode,
                contract_cap=contract_cap,
            ),
            reply_interpretation=reply_interpretation,
            confidence=confidence,
            reason=reason,
        ),
        warnings,
        errors,
    )


def run_turn_decision_pipeline(
    *,
    generate_reply_fn: Callable[..., AssistantReply],
    mode: TurnDecisionMode,
    source: TurnDecisionSource,
    user_message: str,
    last_assistant_message: str | None,
    focused_project_id: int | None,
    referenced_project_ids: list[int] | None,
    groups: list[ToolGroup],
    response_modes: list[ResponseMode],
    contract_caps: list[ResponseContractName] | None = None,
    extra_context: dict[str, Any] | None = None,
) -> TurnDecisionResult:
    if mode == "off":
        return TurnDecisionResult(ok=False, mode=mode, source=source, skipped_reason="mode_off")

    contract_cap_names = list(contract_caps or response_contract_names())
    group_names = [group.name for group in groups if group.name]
    tool_names = [
        tool_name
        for group in groups
        for tool_name in group.tool_names
        if isinstance(tool_name, str) and tool_name
    ]
    schema = turn_decision_schema(
        group_names=group_names,
        tool_names=tool_names,
        response_modes=response_modes,
        contract_caps=contract_cap_names,
    )
    prompt = load_bound_prompt(
        _turn_decision_prompt,
        values=_turn_decision_prompt_values(
            source=source,
            groups=groups,
            response_modes=response_modes,
            contract_caps=contract_cap_names,
        ),
    )
    messages = [
        {"role": "system", "content": prompt.text},
        {
            "role": "user",
            "content": json.dumps(
                {
                    "source": source,
                    "user_message": user_message,
                    "last_assistant_message": last_assistant_message or "",
                    "focused_project_id": focused_project_id,
                    "referenced_project_ids": list(referenced_project_ids or []),
                    "available_groups": group_names,
                    "available_tools": tool_names,
                    "response_modes": response_modes,
                    "response_contract_caps": contract_cap_names,
                    "context": extra_context or {},
                },
                ensure_ascii=False,
            ),
        },
    ]
    reply = generate_classifier_reply(
        base_generate_reply_fn=generate_reply_fn,
        messages=messages,
        vllm_extra_body={
            "structured_outputs": {"json": schema},
            "max_tokens": 300,
            "temperature": 0,
        },
        observability_context=prompt_observability_context(
            prompt,
            extra={
                "surface": "turn_decision",
                "task": source,
                "stage": (
                    "reply_interpretation"
                    if isinstance(extra_context, dict) and isinstance(extra_context.get("awaiting_reply_state"), dict)
                    else "turn_decision"
                ),
            },
        ),
    )
    result = TurnDecisionResult(
        ok=False,
        mode=mode,
        source=source,
        applied=mode == "own",
        reply_meta={
            "provider": reply.provider,
            "model": reply.model,
            "meta": reply.meta,
        },
    )
    parsed = parse_json_object(reply.content)
    result.raw_decision = parsed
    if not isinstance(parsed, dict):
        result.errors.append("invalid_json")
        return result

    validated, warnings, errors = _validate_turn_decision(
        raw_decision=parsed,
        source=source,
        groups=groups,
        response_modes=response_modes,
        contract_caps=contract_cap_names,
        awaiting_reply_state=(
            dict(extra_context.get("awaiting_reply_state"))
            if isinstance(extra_context, dict) and isinstance(extra_context.get("awaiting_reply_state"), dict)
            else None
        ),
    )
    result.warnings.extend(warnings)
    result.errors.extend(errors)
    if validated is None:
        return result

    result.ok = True
    result.decision = validated
    return result


def selected_tool_names_from_decision(
    *,
    groups: list[ToolGroup],
    decision: TurnDecision,
    explicit_requested_tool_names: list[str] | None = None,
    always_include: list[str] | None = None,
) -> set[str]:
    selected: set[str] = set(always_include or [])
    if decision.tool_plan.primary_group:
        selected |= tool_names_for_groups(
            groups=groups,
            primary=decision.tool_plan.primary_group,
            secondary=list(decision.tool_plan.secondary_groups),
        )
    if decision.tool_plan.preferred_first_tool:
        selected.add(decision.tool_plan.preferred_first_tool)
    if explicit_requested_tool_names:
        selected |= {
            str(name).strip()
            for name in explicit_requested_tool_names
            if isinstance(name, str) and str(name).strip()
        }
    return selected
