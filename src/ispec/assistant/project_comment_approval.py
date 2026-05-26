from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Callable, Literal

from ispec.assistant.context import extract_project_ids
from ispec.assistant.json_utils import parse_json_object
from ispec.assistant.service import AssistantReply
from ispec.assistant.small_classifier import (
    build_small_classifier_task,
    normalize_small_classifier_decision,
)


ApprovalMode = Literal["off", "shadow", "own"]
TriggerKind = Literal[
    "none",
    "explicit_save",
    "direct_note_request",
    "draft_only",
    "revise",
    "deny",
    "defer",
    "confirmation",
]
GateKind = Literal["none", "pending_save_confirmation", "direct_write_candidate"]

_TRUTHY_ENV = {"1", "true", "yes", "y", "on"}
_FALSY_ENV = {"0", "false", "no", "n", "off"}
_PENDING_TOOL = "create_project_comment"


@dataclass(frozen=True, kw_only=True)
class ProjectCommentApprovalSettings:
    mode: ApprovalMode
    timeout_ms: int = 6000
    approve_threshold: float = 0.80
    decision_threshold: float = 0.60
    ticket_ttl_seconds: int = 120

    @classmethod
    def from_env(cls, *, assistant_provider: str, state_dir_is_dev: bool) -> "ProjectCommentApprovalSettings":
        raw_mode = (os.getenv("ISPEC_ASSISTANT_PROJECT_COMMENT_APPROVAL_EVAL_MODE") or "auto").strip().lower()
        if raw_mode in {"off", "shadow", "own"}:
            mode: ApprovalMode = raw_mode  # type: ignore[assignment]
        elif raw_mode in _FALSY_ENV:
            mode = "off"
        elif raw_mode in _TRUTHY_ENV:
            mode = "shadow"
        elif str(assistant_provider or "").strip().lower() == "vllm" and state_dir_is_dev:
            mode = "shadow"
        else:
            mode = "off"
        return cls(
            mode=mode,
            timeout_ms=_int_env("ISPEC_ASSISTANT_PROJECT_COMMENT_APPROVAL_TIMEOUT_MS", default=6000, low=500, high=30_000),
            approve_threshold=_float_env(
                "ISPEC_ASSISTANT_PROJECT_COMMENT_APPROVAL_APPROVE_THRESHOLD",
                default=0.80,
                low=0.0,
                high=1.0,
            ),
            decision_threshold=_float_env(
                "ISPEC_ASSISTANT_PROJECT_COMMENT_APPROVAL_DECISION_THRESHOLD",
                default=0.60,
                low=0.0,
                high=1.0,
            ),
            ticket_ttl_seconds=_int_env(
                "ISPEC_ASSISTANT_PROJECT_COMMENT_APPROVAL_TICKET_TTL_SECONDS",
                default=120,
                low=1,
                high=3600,
            ),
        )


@dataclass(frozen=True, kw_only=True)
class ProjectCommentTrigger:
    kind: TriggerKind
    phrase: str | None
    command_protocol: str
    explicit_save_terms: tuple[str, ...]
    denial_terms: tuple[str, ...]
    revision_terms: tuple[str, ...]
    defer_terms: tuple[str, ...]
    draft_only_terms: tuple[str, ...]
    comment_object_terms: tuple[str, ...]
    project_ids: tuple[int, ...]
    action_terms: tuple[str, ...]
    near_action_terms: tuple[dict[str, str], ...]
    legacy_confirmation_kind: str
    legacy_save_requested: bool
    source_message_hash: str | None

    @property
    def is_write_candidate(self) -> bool:
        return self.kind in {"explicit_save", "direct_note_request", "confirmation"}

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "kind": self.kind,
            "phrase": self.phrase,
            "command_protocol": self.command_protocol,
            "explicit_save_terms": list(self.explicit_save_terms),
            "denial_terms": list(self.denial_terms),
            "revision_terms": list(self.revision_terms),
            "defer_terms": list(self.defer_terms),
            "draft_only_terms": list(self.draft_only_terms),
            "comment_object_terms": list(self.comment_object_terms),
            "project_ids": list(self.project_ids),
            "project_comment_action_terms": list(self.action_terms),
            "project_comment_near_action_terms": [dict(item) for item in self.near_action_terms],
            "project_comment_write_candidate": self.is_write_candidate,
            "legacy_confirmation_kind": self.legacy_confirmation_kind,
            "legacy_save_requested": bool(self.legacy_save_requested),
            "source_message_hash": self.source_message_hash,
        }

    def to_lexical_features(self) -> dict[str, Any]:
        payload = self.to_dict()
        payload.pop("kind", None)
        payload.pop("phrase", None)
        return payload


@dataclass(frozen=True, kw_only=True)
class ProjectCommentGate:
    eligible: bool
    kind: GateKind
    reason: str
    project_id: int | None
    pending_tool: str
    session_id: str
    thread_key: str
    current_turn_id: str
    prior_assistant_message_id: int | None
    pending_draft_hash: str | None
    source_message_hash: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "eligible": bool(self.eligible),
            "kind": self.kind,
            "reason": self.reason,
            "project_id": self.project_id,
            "pending_tool": self.pending_tool,
            "session_id": self.session_id,
            "thread_key": self.thread_key,
            "current_turn_id": self.current_turn_id,
            "prior_assistant_message_id": self.prior_assistant_message_id,
            "pending_draft_hash": self.pending_draft_hash,
            "source_message_hash": self.source_message_hash,
        }


@dataclass(frozen=True, kw_only=True)
class ProjectCommentClassifierResult:
    ran: bool
    ok: bool
    label: str | None
    confidence: float
    reason: str
    provider: str | None
    model: str | None
    latency_ms: int
    timeout_ms: int
    error: str | None

    @classmethod
    def empty(cls, *, timeout_ms: int) -> "ProjectCommentClassifierResult":
        return cls(
            ran=False,
            ok=False,
            label=None,
            confidence=0.0,
            reason="",
            provider=None,
            model=None,
            latency_ms=0,
            timeout_ms=int(timeout_ms),
            error=None,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "ran": self.ran,
            "ok": self.ok,
            "label": self.label,
            "confidence": self.confidence,
            "reason": self.reason,
            "provider": self.provider,
            "model": self.model,
            "latency_ms": self.latency_ms,
            "timeout_ms": self.timeout_ms,
            "error": self.error,
        }


@dataclass(frozen=True, kw_only=True)
class ProjectCommentPolicyDecision:
    decision: str
    live_applied: bool
    approve_threshold: float
    decision_threshold: float
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision,
            "live_applied": self.live_applied,
            "approve_threshold": self.approve_threshold,
            "decision_threshold": self.decision_threshold,
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True, kw_only=True)
class ProjectCommentWriteTicket:
    issued: bool
    ticket_id: str | None
    tool_name: str
    project_id: int | None
    session_id: str | None
    thread_key: str | None
    current_turn_id: str | None
    pending_draft_hash: str | None
    candidate_comment_hash: str | None
    issued_at: str | None
    expires_at: str | None
    ttl_seconds: int
    single_use: bool
    shadow_only: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "issued": self.issued,
            "ticket_id": self.ticket_id,
            "tool_name": self.tool_name,
            "project_id": self.project_id,
            "session_id": self.session_id,
            "thread_key": self.thread_key,
            "current_turn_id": self.current_turn_id,
            "pending_draft_hash": self.pending_draft_hash,
            "candidate_comment_hash": self.candidate_comment_hash,
            "issued_at": self.issued_at,
            "expires_at": self.expires_at,
            "ttl_seconds": self.ttl_seconds,
            "single_use": self.single_use,
            "shadow_only": self.shadow_only,
        }


def _int_env(key: str, *, default: int, low: int, high: int) -> int:
    raw = (os.getenv(key) or "").strip()
    try:
        return max(low, min(high, int(raw))) if raw else default
    except ValueError:
        return default


def _float_env(key: str, *, default: float, low: float, high: float) -> float:
    raw = (os.getenv(key) or "").strip()
    try:
        return max(low, min(high, float(raw))) if raw else default
    except ValueError:
        return default


def stable_hash(value: Any) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def lexical_terms_present(message: str | None, terms: tuple[str, ...]) -> tuple[str, ...]:
    text = str(message or "").lower()
    found: list[str] = []
    for term in terms:
        if re.search(rf"(?<![A-Za-z0-9_]){re.escape(term)}(?![A-Za-z0-9_])", text):
            found.append(term)
    return tuple(found)


def token_near_term(token: str, term: str) -> bool:
    token = str(token or "").strip().lower()
    term = str(term or "").strip().lower()
    if token == term:
        return True
    if len(token) < 3 or len(term) < 3 or abs(len(token) - len(term)) > 1:
        return False
    if len(token) == len(term):
        diffs = [idx for idx, pair in enumerate(zip(token, term, strict=True)) if pair[0] != pair[1]]
        if len(diffs) <= 1:
            return True
        if len(diffs) == 2:
            first, second = diffs
            return (
                second == first + 1
                and token[first] == term[second]
                and token[second] == term[first]
            )
        return False

    shorter, longer = (token, term) if len(token) < len(term) else (term, token)
    skipped = False
    short_idx = 0
    for char in longer:
        if short_idx < len(shorter) and shorter[short_idx] == char:
            short_idx += 1
            continue
        if skipped:
            return False
        skipped = True
    return True


def near_lexical_terms_present(message: str | None, terms: tuple[str, ...]) -> tuple[dict[str, str], ...]:
    tokens = [token for token in re.findall(r"[A-Za-z][A-Za-z']{1,31}", str(message or "").lower()) if token]
    found: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for token in tokens:
        token = token.strip()
        if not token:
            continue
        for term in terms:
            if not token_near_term(token, term):
                continue
            key = (token, term)
            if key not in seen:
                seen.add(key)
                found.append({"token": token, "matched": term})
            break
    return tuple(found)


def project_comment_force_tool_choice_requested(message: str | None) -> bool:
    text = str(message or "").strip().lower()
    if not text:
        return False
    if re.search(r"\b(do not|don't|dont)\s+(save|log|record|add|commit)\b", text):
        return False
    normalized = re.sub(r"[^\w\s]", " ", text)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    if not re.search(r"\b(save|commit)\b", normalized):
        return False
    return bool(re.search(r"\b(project|history|comment|comments|note|notes|memo|it|this)\b", normalized))


def detect_project_comment_trigger(
    *,
    message: str | None,
    legacy_confirmation_kind: str,
    legacy_save_requested: bool,
) -> ProjectCommentTrigger:
    explicit_save_terms = lexical_terms_present(message, ("save", "log", "record", "add", "commit"))
    denial_terms = lexical_terms_present(message, ("no", "nope", "nah", "deny", "cancel", "not", "dont", "don't"))
    revision_terms = lexical_terms_present(
        message,
        ("summary", "summarize", "revise", "rewrite", "edit", "change", "tweak", "shorter", "modify"),
    )
    defer_terms = lexical_terms_present(message, ("later", "wait", "hold", "postpone"))
    draft_only_terms = lexical_terms_present(message, ("draft", "review", "suggest", "propose"))
    comment_object_terms = lexical_terms_present(message, ("note", "notes", "comment", "comments", "memo", "history"))
    project_ids = tuple(extract_project_ids(message or "")[:10])
    action_terms = lexical_terms_present(
        message,
        ("make", "leave", "create", "log", "record", "add", "track", "remember", "note"),
    )
    near_action_terms = near_lexical_terms_present(
        message,
        ("make", "leave", "create", "log", "record", "add", "track", "remember", "note"),
    )

    command_protocol = "none"
    lowered = str(message or "").lower()
    if re.search(r"\b(do not|don't|dont|not)\s+(save|log|record|add|commit)\b", lowered):
        command_protocol = "explicit_deny"
    elif project_comment_force_tool_choice_requested(message):
        command_protocol = "explicit_save"
    elif draft_only_terms and not explicit_save_terms:
        command_protocol = "explicit_draft_only"

    kind: TriggerKind = "none"
    phrase: str | None = None
    if command_protocol == "explicit_deny":
        kind = "deny"
        phrase = "explicit_deny"
    elif command_protocol == "explicit_draft_only":
        kind = "draft_only"
        phrase = draft_only_terms[0] if draft_only_terms else None
    elif defer_terms and not explicit_save_terms:
        kind = "defer"
        phrase = defer_terms[0]
    elif revision_terms and not explicit_save_terms:
        kind = "revise"
        phrase = revision_terms[0]
    elif str(legacy_confirmation_kind or "").strip() not in {"", "none"}:
        kind = "confirmation"
        phrase = str(legacy_confirmation_kind)
    elif bool(legacy_save_requested) or command_protocol == "explicit_save":
        kind = "explicit_save"
        phrase = explicit_save_terms[0] if explicit_save_terms else "explicit_save"
    elif project_ids and comment_object_terms and (action_terms or near_action_terms):
        kind = "direct_note_request"
        if action_terms:
            phrase = action_terms[0]
        elif near_action_terms:
            phrase = f"{near_action_terms[0]['token']}~{near_action_terms[0]['matched']}"

    return ProjectCommentTrigger(
        kind=kind,
        phrase=phrase,
        command_protocol=command_protocol,
        explicit_save_terms=explicit_save_terms,
        denial_terms=denial_terms,
        revision_terms=revision_terms,
        defer_terms=defer_terms,
        draft_only_terms=draft_only_terms,
        comment_object_terms=comment_object_terms,
        project_ids=project_ids,
        action_terms=action_terms,
        near_action_terms=near_action_terms,
        legacy_confirmation_kind=str(legacy_confirmation_kind or "none"),
        legacy_save_requested=bool(legacy_save_requested),
        source_message_hash=stable_hash(message),
    )


def build_project_comment_gate(
    *,
    settings: ProjectCommentApprovalSettings,
    trigger: ProjectCommentTrigger,
    tool_protocol: str,
    available_tool_names: set[str],
    focused_project_id: int | None,
    session_id: str,
    thread_key: str,
    current_turn_id: str,
    prior_assistant_message_id: int | None,
    prior_assistant_message: str | None,
    user_message: str | None,
    awaiting_reply_state: dict[str, Any] | None,
) -> ProjectCommentGate:
    base = {
        "project_id": focused_project_id,
        "pending_tool": _PENDING_TOOL,
        "session_id": session_id,
        "thread_key": thread_key,
        "current_turn_id": current_turn_id,
        "prior_assistant_message_id": prior_assistant_message_id,
        "pending_draft_hash": stable_hash(prior_assistant_message),
        "source_message_hash": stable_hash(user_message),
    }
    if settings.mode == "off":
        return ProjectCommentGate(eligible=False, kind="none", reason="mode_off", **base)
    if str(tool_protocol or "").strip().lower() != "openai":
        return ProjectCommentGate(eligible=False, kind="none", reason="tool_protocol_not_openai", **base)
    if _PENDING_TOOL not in available_tool_names:
        return ProjectCommentGate(eligible=False, kind="none", reason="project_comment_tool_unavailable", **base)
    if not isinstance(focused_project_id, int) or focused_project_id <= 0:
        return ProjectCommentGate(eligible=False, kind="none", reason="missing_project_id", **base)
    if isinstance(awaiting_reply_state, dict):
        return ProjectCommentGate(eligible=True, kind="pending_save_confirmation", reason="awaiting_project_comment_save_confirmation", **base)
    if trigger.kind in {"explicit_save", "direct_note_request"}:
        return ProjectCommentGate(eligible=True, kind="direct_write_candidate", reason="project_comment_write_candidate", **base)
    return ProjectCommentGate(eligible=False, kind="none", reason="no_pending_approval_state", **base)


def empty_project_comment_classifier(*, settings: ProjectCommentApprovalSettings) -> ProjectCommentClassifierResult:
    return ProjectCommentClassifierResult.empty(timeout_ms=settings.timeout_ms)


def run_project_comment_approval_classifier(
    *,
    settings: ProjectCommentApprovalSettings,
    gate: ProjectCommentGate,
    trigger: ProjectCommentTrigger,
    user_message: str,
    prior_assistant_message: str | None,
    focused_project_id: int | None,
    generate_reply_fn: Callable[..., AssistantReply],
    generate_classifier_reply_fn: Callable[..., AssistantReply],
) -> ProjectCommentClassifierResult:
    if not gate.eligible:
        return ProjectCommentClassifierResult.empty(timeout_ms=settings.timeout_ms)
    try:
        prepared = build_small_classifier_task(
            "project_comment_approval",
            payload={
                "user_message": user_message,
                "prior_assistant_message": prior_assistant_message or "",
                "state_gate": gate.to_dict(),
                "trigger": trigger.to_dict(),
                "lexical_features": trigger.to_lexical_features(),
                "pending_action": _PENDING_TOOL,
                "focused_project_id": focused_project_id,
            },
        )
        started = time.monotonic()
        reply = generate_classifier_reply_fn(
            base_generate_reply_fn=generate_reply_fn,
            messages=prepared.messages,
            vllm_extra_body={
                "structured_outputs": {"json": prepared.schema},
                "max_tokens": prepared.task.max_tokens,
                "temperature": prepared.task.temperature,
            },
            observability_context=prepared.observability_context(
                extra={"surface": "support_chat", "stage": "project_comment_approval_eval"}
            ),
            timeout_seconds=max(0.5, float(settings.timeout_ms) / 1000.0),
        )
        latency_ms = int((time.monotonic() - started) * 1000)
        parsed = parse_json_object(reply.content)
        decision = normalize_small_classifier_decision(
            "project_comment_approval",
            parsed if isinstance(parsed, dict) else None,
        )
        if decision is None:
            return ProjectCommentClassifierResult(
                ran=True,
                ok=False,
                label=None,
                confidence=0.0,
                reason="",
                provider=reply.provider,
                model=reply.model,
                latency_ms=latency_ms,
                timeout_ms=settings.timeout_ms,
                error=reply.error or "invalid_classifier_json",
            )
        return ProjectCommentClassifierResult(
            ran=True,
            ok=bool(reply.ok),
            label=decision.label,
            confidence=decision.confidence,
            reason=_truncate(decision.reason, 400) or "",
            provider=reply.provider,
            model=reply.model,
            latency_ms=latency_ms,
            timeout_ms=settings.timeout_ms,
            error=reply.error or (None if reply.ok else "classifier_reply_not_ok"),
        )
    except Exception as exc:
        return ProjectCommentClassifierResult(
            ran=True,
            ok=False,
            label=None,
            confidence=0.0,
            reason="",
            provider=None,
            model=None,
            latency_ms=0,
            timeout_ms=settings.timeout_ms,
            error=f"{type(exc).__name__}: {exc}",
        )


def decide_project_comment_policy(
    *,
    settings: ProjectCommentApprovalSettings,
    gate: ProjectCommentGate,
    trigger: ProjectCommentTrigger,
    classifier: ProjectCommentClassifierResult,
) -> tuple[ProjectCommentPolicyDecision, ProjectCommentWriteTicket]:
    reasons: list[str] = []
    policy_decision = "not_gated"
    issued = False
    if settings.mode == "off" or not gate.eligible:
        reasons.append(gate.reason or "not_gated")
    elif not classifier.ran or not classifier.ok:
        policy_decision = "classifier_unavailable"
        reasons.append(classifier.error or "classifier_unavailable")
    else:
        label = str(classifier.label or "")
        confidence = float(classifier.confidence or 0.0)
        if trigger.command_protocol == "explicit_deny":
            policy_decision = "no_ticket"
            reasons.append("explicit_command_protocol_deny")
        elif confidence < float(settings.decision_threshold):
            policy_decision = "ask_explicit_confirmation"
            reasons.append("confidence_below_decision_threshold")
        elif label == "approve_save" and confidence >= float(settings.approve_threshold):
            policy_decision = "shadow_ticket"
            issued = True
            reasons.append("approve_save_above_threshold")
        elif label == "approve_save":
            policy_decision = "ask_explicit_confirmation"
            reasons.append("approve_save_below_approve_threshold")
        elif label in {"deny_save", "draft_only", "revise_draft"}:
            policy_decision = "no_ticket"
            reasons.append(label)
        else:
            policy_decision = "ask_explicit_confirmation"
            reasons.append(label or "unrelated_or_unclear")

    from ispec.assistant.models import utcnow

    issued_at_dt = utcnow() if issued else None
    expires_at_dt = issued_at_dt + timedelta(seconds=int(settings.ticket_ttl_seconds)) if issued_at_dt is not None else None
    issued_at = issued_at_dt.isoformat() if issued_at_dt is not None else None
    expires_at = expires_at_dt.isoformat() if expires_at_dt is not None else None
    ticket_seed = {
        "tool_name": _PENDING_TOOL,
        "project_id": gate.project_id,
        "session_id": gate.session_id,
        "thread_key": gate.thread_key,
        "current_turn_id": gate.current_turn_id,
        "pending_draft_hash": gate.pending_draft_hash,
        "source_message_hash": gate.source_message_hash,
        "issued_at": issued_at,
    }
    ticket_id = "pct_" + hashlib.sha256(
        json.dumps(ticket_seed, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:24] if issued else None
    return (
        ProjectCommentPolicyDecision(
            decision=policy_decision,
            live_applied=False,
            approve_threshold=float(settings.approve_threshold),
            decision_threshold=float(settings.decision_threshold),
            reasons=tuple(reasons),
        ),
        ProjectCommentWriteTicket(
            issued=issued,
            ticket_id=ticket_id,
            tool_name=_PENDING_TOOL,
            project_id=gate.project_id,
            session_id=gate.session_id,
            thread_key=gate.thread_key,
            current_turn_id=gate.current_turn_id,
            pending_draft_hash=gate.pending_draft_hash,
            candidate_comment_hash=None,
            issued_at=issued_at,
            expires_at=expires_at,
            ttl_seconds=int(settings.ticket_ttl_seconds),
            single_use=True,
            shadow_only=True,
        ),
    )


def project_comment_write_outcome(
    *,
    tool_calls: list[dict[str, Any]],
    focused_project_id: int | None,
    safe_int_fn: Callable[[Any], int | None],
) -> dict[str, Any]:
    outcome = {
        "status": "not_attempted",
        "tool_name": None,
        "project_id": focused_project_id,
        "comment_id": None,
        "error": None,
    }
    for call in tool_calls:
        if not isinstance(call, dict) or str(call.get("name") or "").strip() != _PENDING_TOOL:
            continue
        args = call.get("arguments") if isinstance(call.get("arguments"), dict) else {}
        project_id = safe_int_fn(args.get("project_id")) if isinstance(args, dict) else None
        outcome.update(
            {
                "status": "succeeded" if call.get("ok") else "failed",
                "tool_name": _PENDING_TOOL,
                "project_id": project_id or focused_project_id,
                "error": call.get("error"),
            }
        )
        preview = str(call.get("result_preview") or "").strip()
        if preview:
            try:
                result = json.loads(preview)
            except Exception:
                result = None
            if isinstance(result, dict):
                outcome["comment_id"] = safe_int_fn(result.get("comment_id"))
                outcome["project_id"] = safe_int_fn(result.get("project_id")) or outcome.get("project_id")
        return outcome
    return outcome


def forced_tool_choice_name(forced_tool_choice: str | dict[str, Any] | None) -> str | None:
    if isinstance(forced_tool_choice, str):
        return forced_tool_choice.strip() or None
    if not isinstance(forced_tool_choice, dict):
        return None
    func_obj = forced_tool_choice.get("function")
    if not isinstance(func_obj, dict):
        return None
    return str(func_obj.get("name") or "").strip() or None


def _truncate(value: str | None, limit: int = 400) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if limit <= 0 or len(text) <= limit:
        return text
    return text[: limit - 1] + "..."
