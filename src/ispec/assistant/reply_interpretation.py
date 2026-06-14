from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal


ReplyInterpretationKind = Literal["none", "approve", "deny", "defer", "modify", "unclear"]
ReplyInterpretationAction = Literal[
    "none",
    "approve_save",
    "deny_save",
    "defer_save",
    "modify_before_save",
    "clarify",
]

PENDING_PROJECT_COMMENT_SAVE = "project_comment_save_confirmation"
PENDING_PROJECT_COMMENT_TOOL = "create_project_comment"

_AFFIRMATIVE_TERMS = {"yes", "y", "yeah", "yep", "yup", "ok", "okay", "sure", "confirm"}
_NEGATIVE_TERMS = {"no", "n", "nope", "nah"}
_SHORT_APPROVAL_PHRASES = {
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
}
_SHORT_DENIAL_PHRASES = {"no", "n", "nope", "nah"}

_REPLY_INTERPRETATION_ACTIONS: dict[str, dict[str, ReplyInterpretationAction]] = {
    PENDING_PROJECT_COMMENT_SAVE: {
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


@dataclass(frozen=True, kw_only=True)
class ShortReplyInterpretation:
    normalized: str
    is_confirmation: bool
    is_affirmative: bool
    kind: ReplyInterpretationKind


@dataclass(frozen=True, kw_only=True)
class PendingReplyState:
    name: str
    project_id: int
    pending_tool: str
    source: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": self.name,
            "project_id": self.project_id,
            "pending_tool": self.pending_tool,
        }
        if self.source:
            payload["source"] = self.source
        return payload

    @classmethod
    def from_dict(cls, value: dict[str, Any] | None) -> "PendingReplyState | None":
        if not isinstance(value, dict):
            return None
        name = str(value.get("name") or "").strip()
        pending_tool = str(value.get("pending_tool") or "").strip()
        project_id = _safe_positive_int(value.get("project_id"))
        if not name or not pending_tool or project_id is None:
            return None
        return cls(
            name=name,
            project_id=project_id,
            pending_tool=pending_tool,
            source=str(value.get("source") or "").strip() or None,
        )


AwaitingProjectCommentSaveState = PendingReplyState


@dataclass(frozen=True, kw_only=True)
class ReplyInterpretation:
    short_reply: ShortReplyInterpretation
    awaiting_save: AwaitingProjectCommentSaveState | None
    kind: ReplyInterpretationKind
    action: ReplyInterpretationAction
    classifier_kind: ReplyInterpretationKind = "none"
    classifier_action: ReplyInterpretationAction = "none"
    classifier_confidence: float = 0.0
    classifier_reason: str = ""
    runtime_kind: ReplyInterpretationKind = "none"
    runtime_action: ReplyInterpretationAction = "none"
    applied: bool = False
    policy_messages: tuple[dict[str, str], ...] = ()

    @property
    def is_confirmation_reply(self) -> bool:
        return self.short_reply.is_confirmation

    @property
    def is_affirmative_reply(self) -> bool:
        return self.short_reply.is_affirmative

    @property
    def has_pending_save(self) -> bool:
        return self.awaiting_save is not None

    @property
    def confirms_project_comment_save(self) -> bool:
        return self.runtime_action == "approve_save"

    @property
    def should_log_meta(self) -> bool:
        return self.has_pending_save or self.classifier_kind != "none"

    def awaiting_state_dict(self) -> dict[str, Any] | None:
        return self.awaiting_save.to_dict() if self.awaiting_save is not None else None

    def with_turn_decision(
        self,
        *,
        turn_decision_result: Any,
        turn_decision_runtime_applied: bool,
        available_tool_names: set[str],
        focused_project_id: int | None,
    ) -> "ReplyInterpretation":
        decision = getattr(turn_decision_result, "decision", None)
        classifier_reply = getattr(decision, "reply_interpretation", None) if decision is not None else None
        classifier_kind = _reply_interpretation_kind_or_none(getattr(classifier_reply, "kind", None))
        classifier_action: ReplyInterpretationAction = "none"
        awaiting_save = self.awaiting_save
        if turn_decision_runtime_applied and awaiting_save is None:
            awaiting_save = turn_decision_implies_project_comment_confirmation(
                turn_decision_result=turn_decision_result,
                available_tool_names=available_tool_names,
                focused_project_id=focused_project_id,
            )
        mapped_classifier_action = reply_interpretation_action(awaiting_save, classifier_kind)
        if mapped_classifier_action is not None:
            classifier_action = mapped_classifier_action

        applied = bool(turn_decision_runtime_applied and awaiting_save is not None)
        runtime_kind = self.kind
        runtime_action = self.action
        if applied:
            runtime_kind, runtime_action = _merge_pending_save_reply_decision(
                lexical_kind=self.kind,
                lexical_action=self.action,
                classifier_kind=classifier_kind,
                classifier_action=classifier_action,
            )
        policy_messages = tuple(
            reply_interpretation_messages(
                action=runtime_action if applied else None,
                awaiting_state=awaiting_save,
            )
        )

        return ReplyInterpretation(
            short_reply=self.short_reply,
            awaiting_save=awaiting_save,
            kind=self.kind,
            action=self.action,
            classifier_kind=classifier_kind,
            classifier_action=classifier_action,
            classifier_confidence=_safe_float(getattr(classifier_reply, "confidence", 0.0)),
            classifier_reason=str(getattr(classifier_reply, "reason", "") or ""),
            runtime_kind=runtime_kind,
            runtime_action=runtime_action,
            applied=applied,
            policy_messages=policy_messages,
        )

    def meta(self, *, removed_write_tools: list[str] | tuple[str, ...] = ()) -> dict[str, Any]:
        return {
            "awaiting_state": self.awaiting_save.name if self.awaiting_save is not None else None,
            "legacy_kind": self.kind,
            "legacy_action": self.action,
            "classifier_kind": self.classifier_kind,
            "classifier_action": self.classifier_action,
            "classifier_confidence": self.classifier_confidence,
            "classifier_reason": self.classifier_reason,
            "runtime_kind": self.runtime_kind,
            "runtime_action": self.runtime_action,
            "applied": self.applied,
            "removed_write_tools": list(removed_write_tools),
        }


def normalize_short_reply(message: str | None) -> str:
    normalized = re.sub(r"[^\w\s]", "", str(message or "").strip().lower())
    return re.sub(r"\s+", " ", normalized).strip()


def classify_short_reply(message: str | None) -> ShortReplyInterpretation:
    normalized = normalize_short_reply(message)
    if not normalized or len(normalized) > 64:
        return ShortReplyInterpretation(
            normalized=normalized,
            is_confirmation=False,
            is_affirmative=False,
            kind="unclear",
        )

    if normalized in _SHORT_APPROVAL_PHRASES:
        return ShortReplyInterpretation(
            normalized=normalized,
            is_confirmation=True,
            is_affirmative=True,
            kind="approve",
        )
    if normalized in _SHORT_DENIAL_PHRASES:
        return ShortReplyInterpretation(
            normalized=normalized,
            is_confirmation=True,
            is_affirmative=False,
            kind="deny",
        )

    tokens = normalized.split()
    if len(tokens) > 6:
        return ShortReplyInterpretation(
            normalized=normalized,
            is_confirmation=False,
            is_affirmative=False,
            kind="unclear",
        )

    has_negative = any(token in _NEGATIVE_TERMS for token in tokens)
    has_affirmative = any(token in _AFFIRMATIVE_TERMS for token in tokens)
    is_confirmation = bool(has_affirmative or has_negative)
    is_affirmative = bool(has_affirmative and not has_negative)
    return ShortReplyInterpretation(
        normalized=normalized,
        is_confirmation=is_confirmation,
        is_affirmative=is_affirmative,
        kind=("approve" if is_affirmative else "deny") if is_confirmation else "unclear",
    )


def is_confirmation_reply(message: str | None) -> bool:
    return classify_short_reply(message).is_confirmation


def is_affirmative_reply(message: str | None) -> bool:
    return classify_short_reply(message).is_affirmative


def assistant_requested_project_history_save(message: str | None) -> bool:
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


def awaiting_project_comment_save_state(
    *,
    tool_protocol: str,
    available_tool_names: set[str],
    focused_project_id: int | None,
    last_assistant_message: str | None,
) -> PendingReplyState | None:
    if str(tool_protocol or "").strip().lower() != "openai":
        return None
    if PENDING_PROJECT_COMMENT_TOOL not in available_tool_names:
        return None
    if not isinstance(focused_project_id, int) or focused_project_id <= 0:
        return None
    if not assistant_requested_project_history_save(last_assistant_message):
        return None
    return PendingReplyState(
        name=PENDING_PROJECT_COMMENT_SAVE,
        project_id=focused_project_id,
        pending_tool=PENDING_PROJECT_COMMENT_TOOL,
    )


def interpret_reply_for_project_comment_save(
    *,
    tool_protocol: str,
    available_tool_names: set[str],
    focused_project_id: int | None,
    last_assistant_message: str | None,
    user_message: str | None,
) -> ReplyInterpretation:
    short_reply = classify_short_reply(user_message)
    awaiting_save = awaiting_project_comment_save_state(
        tool_protocol=tool_protocol,
        available_tool_names=available_tool_names,
        focused_project_id=focused_project_id,
        last_assistant_message=last_assistant_message,
    )
    if awaiting_save is None:
        return ReplyInterpretation(
            short_reply=short_reply,
            awaiting_save=None,
            kind="none",
            action="none",
            runtime_kind="none",
            runtime_action="none",
        )

    action = reply_interpretation_action(awaiting_save, short_reply.kind) or "none"
    return ReplyInterpretation(
        short_reply=short_reply,
        awaiting_save=awaiting_save,
        kind=short_reply.kind,
        action=action,
        runtime_kind=short_reply.kind,
        runtime_action=action,
    )


def legacy_reply_interpretation_kind(message: str | None) -> ReplyInterpretationKind:
    return classify_short_reply(message).kind


def reply_interpretation_action(
    awaiting_state: PendingReplyState | dict[str, Any] | None,
    kind: str | None,
) -> ReplyInterpretationAction | None:
    state = awaiting_state if isinstance(awaiting_state, PendingReplyState) else PendingReplyState.from_dict(awaiting_state)
    if state is None:
        return None
    mapping = _REPLY_INTERPRETATION_ACTIONS.get(state.name, {})
    return mapping.get(str(kind or "").strip())


def _merge_pending_save_reply_decision(
    *,
    lexical_kind: ReplyInterpretationKind,
    lexical_action: ReplyInterpretationAction,
    classifier_kind: ReplyInterpretationKind,
    classifier_action: ReplyInterpretationAction,
) -> tuple[ReplyInterpretationKind, ReplyInterpretationAction]:
    if lexical_action in {"deny_save", "defer_save", "modify_before_save"}:
        return lexical_kind, lexical_action
    if classifier_action in {"approve_save", "deny_save", "defer_save", "modify_before_save"}:
        return classifier_kind, classifier_action
    if lexical_action == "approve_save":
        return lexical_kind, lexical_action
    if classifier_action == "clarify":
        return classifier_kind, classifier_action
    return classifier_kind, classifier_action


def turn_decision_implies_project_comment_confirmation(
    *,
    turn_decision_result: Any,
    available_tool_names: set[str],
    focused_project_id: int | None,
) -> PendingReplyState | None:
    decision = getattr(turn_decision_result, "decision", None)
    if decision is None:
        return None
    if PENDING_PROJECT_COMMENT_TOOL not in available_tool_names:
        return None
    if not isinstance(focused_project_id, int) or focused_project_id <= 0:
        return None
    write_plan = getattr(decision, "write_plan", None)
    if getattr(write_plan, "mode", None) != "confirm_save":
        return None
    tool_plan = getattr(decision, "tool_plan", None)
    primary_group = str(getattr(tool_plan, "primary_group", "") or "").strip()
    preferred_tool = str(getattr(tool_plan, "preferred_first_tool", "") or "").strip()
    if primary_group and primary_group != "projects":
        return None
    if preferred_tool and preferred_tool != PENDING_PROJECT_COMMENT_TOOL:
        return None
    return PendingReplyState(
        name=PENDING_PROJECT_COMMENT_SAVE,
        project_id=focused_project_id,
        pending_tool=PENDING_PROJECT_COMMENT_TOOL,
        source="turn_decision_confirm_save",
    )


def reply_interpretation_messages(
    *,
    action: str | None,
    awaiting_state: PendingReplyState | dict[str, Any] | None,
) -> list[dict[str, str]]:
    if action == "approve_save":
        return []
    state = awaiting_state if isinstance(awaiting_state, PendingReplyState) else PendingReplyState.from_dict(awaiting_state)
    if state is None:
        return []
    text = _REPLY_INTERPRETATION_POLICY_TEXT.get(str(action or "").strip())
    if not text:
        return []
    if state.name == PENDING_PROJECT_COMMENT_SAVE:
        text += f" The pending draft is for project {state.project_id}."
    return [{"role": "system", "content": text}]


def _reply_interpretation_kind_or_none(value: Any) -> ReplyInterpretationKind:
    text = str(value or "").strip()
    if text in {"approve", "deny", "defer", "modify", "unclear"}:
        return text  # type: ignore[return-value]
    return "none"


def _safe_float(value: Any) -> float:
    if isinstance(value, bool) or value is None:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _safe_positive_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = int(text)
    except ValueError:
        return None
    return parsed if parsed > 0 else None
