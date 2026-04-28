from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Callable


PaneState = str
NotificationAction = str

SENTINEL_STATE_VERSION = 1

_ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

_WAITING_RE = re.compile(
    r"\b("
    r"waiting for (?:approval|input|confirmation|you|human)|"
    r"awaiting (?:approval|input|confirmation)|"
    r"need(?:s)? (?:input|approval|confirmation|review)|"
    r"please (?:confirm|approve|review)|"
    r"should i|do you want|proceed\?|continue\?|"
    r"\by/n\b|\byes/no\b"
    r")\b",
    re.IGNORECASE,
)
_BLOCKED_RE = re.compile(
    r"\b("
    r"blocked|stuck|cannot proceed|can't proceed|permission denied|"
    r"missing (?:dependency|credential|token|file)|"
    r"not authorized|authentication failed"
    r")\b",
    re.IGNORECASE,
)
_ERROR_RE = re.compile(
    r"\b("
    r"traceback|exception|segmentation fault|fatal error|"
    r"tests? failed|failed tests?|pytest.*failed|"
    r"assertionerror|keyerror|valueerror|runtimeerror|"
    r"exit code [1-9][0-9]*|command failed"
    r")\b",
    re.IGNORECASE | re.DOTALL,
)
_COMPLETE_RE = re.compile(
    r"\b("
    r"tests? passed|passed in [0-9.]+s|build (?:succeeded|completed)|"
    r"completed successfully|finished successfully|all done|done\.?|"
    r"ready for review|report (?:written|saved|generated)|"
    r"saved to|successfully (?:saved|added|created|uploaded)"
    r")\b",
    re.IGNORECASE,
)
_RISK_RE = re.compile(
    r"("
    r"\brsync\b[^\n]*(?:--delete|--remove-source-files)|"
    r"\brm\s+-[^\n]*r[^\n]*f|"
    r"\bdelete\b[^\n]*(?:recursive|permanent|all)|"
    r"\bmkfs\b|\bdd\b\s+if=|"
    r"\bchmod\s+-R\b|\bchown\s+-R\b"
    r")",
    re.IGNORECASE,
)
_ORDINARY_PROGRESS_RE = re.compile(
    r"\b(running|processing|downloading|installing|building|compiling|training|syncing|copying)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class SentinelObservation:
    target: str
    pane_id: str | None
    session: str | None
    window_name: str | None
    pane_title: str | None
    current_command: str | None
    pane_active: bool
    pane_dead: bool
    content: str
    normalized_text: str
    content_hash: str
    last_nonempty_line: str | None
    captured_at: str

    def as_report_dict(self, *, include_content: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "target": self.target,
            "pane_id": self.pane_id,
            "session": self.session,
            "window_name": self.window_name,
            "pane_title": self.pane_title,
            "current_command": self.current_command,
            "pane_active": self.pane_active,
            "pane_dead": self.pane_dead,
            "content_hash": self.content_hash,
            "last_nonempty_line": self.last_nonempty_line,
            "captured_at": self.captured_at,
        }
        if include_content:
            payload["content"] = self.content
            payload["normalized_text"] = self.normalized_text
        return payload


@dataclass(frozen=True)
class SentinelPaneState:
    target: str
    state: PaneState
    observed: str
    reason: str
    attention: str
    risk: bool
    changed: bool
    content_hash: str
    notification_action: NotificationAction
    importance_score: int
    notification_fingerprint: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "state": self.state,
            "observed": self.observed,
            "reason": self.reason,
            "attention": self.attention,
            "risk": bool(self.risk),
            "changed": bool(self.changed),
            "content_hash": self.content_hash,
            "notification_action": self.notification_action,
            "importance_score": int(self.importance_score),
            "notification_fingerprint": self.notification_fingerprint,
        }


def utcnow_iso() -> str:
    return datetime.now(UTC).isoformat()


def normalize_pane_text(text: str | None) -> str:
    raw = "" if text is None else str(text)
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")
    raw = _ANSI_RE.sub("", raw)
    raw = _CONTROL_RE.sub("", raw)
    lines = [line.rstrip() for line in raw.split("\n")]
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines)


def content_hash(text: str | None) -> str:
    normalized = normalize_pane_text(text)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def last_nonempty_line(text: str | None) -> str | None:
    normalized = normalize_pane_text(text)
    for line in reversed(normalized.splitlines()):
        stripped = line.strip()
        if stripped:
            return stripped
    return None


def build_observation(
    *,
    pane: dict[str, Any],
    snapshot: dict[str, Any],
    captured_at: str | None = None,
) -> SentinelObservation:
    raw_content = str(snapshot.get("content") or "")
    normalized = normalize_pane_text(raw_content)
    target = (
        str(snapshot.get("target") or "").strip()
        or str(snapshot.get("preferred_alias") or "").strip()
        or str(snapshot.get("capture_target") or "").strip()
        or str(pane.get("target") or "").strip()
        or str(pane.get("pane_id") or "").strip()
        or "unknown"
    )
    return SentinelObservation(
        target=target,
        pane_id=_optional_str(snapshot.get("pane_id") or pane.get("pane_id")),
        session=_optional_str(snapshot.get("session") or pane.get("session")),
        window_name=_optional_str(snapshot.get("window_name") or pane.get("window_name")),
        pane_title=_optional_str(snapshot.get("pane_title") or pane.get("pane_title")),
        current_command=_optional_str(snapshot.get("current_command") or pane.get("current_command")),
        pane_active=bool(snapshot.get("pane_active") if "pane_active" in snapshot else pane.get("pane_active")),
        pane_dead=bool(snapshot.get("pane_dead") if "pane_dead" in snapshot else pane.get("pane_dead")),
        content=raw_content,
        normalized_text=normalized,
        content_hash=hashlib.sha256(normalized.encode("utf-8")).hexdigest(),
        last_nonempty_line=_optional_str(snapshot.get("last_nonempty_line")) or last_nonempty_line(normalized),
        captured_at=captured_at or utcnow_iso(),
    )


def previous_hashes(previous_state: dict[str, Any] | None) -> dict[str, str]:
    state = previous_state if isinstance(previous_state, dict) else {}
    panes = state.get("panes")
    if not isinstance(panes, dict):
        return {}
    result: dict[str, str] = {}
    for target, payload in panes.items():
        if not isinstance(target, str) or not isinstance(payload, dict):
            continue
        hash_value = payload.get("content_hash")
        if isinstance(hash_value, str) and hash_value:
            result[target] = hash_value
    return result


def classify_observation(
    observation: SentinelObservation,
    *,
    changed: bool,
    previous_state: dict[str, Any] | None = None,
    min_notify_seconds: int = 900,
    now: datetime | None = None,
) -> SentinelPaneState:
    text = observation.normalized_text
    line = observation.last_nonempty_line or ""
    searchable = f"{text}\n{line}"
    risk = bool(_RISK_RE.search(searchable))

    if observation.pane_dead:
        state = "complete"
        reason = "pane is marked dead"
    elif _ERROR_RE.search(searchable):
        state = "error"
        reason = "error/failure text observed"
    elif _BLOCKED_RE.search(searchable):
        state = "blocked"
        reason = "blocked/stuck text observed"
    elif _WAITING_RE.search(searchable):
        state = "waiting_for_human"
        reason = "human input or approval appears requested"
    elif _COMPLETE_RE.search(searchable):
        state = "complete"
        reason = "completion or milestone text observed"
    elif observation.pane_active or _ORDINARY_PROGRESS_RE.search(searchable):
        state = "active"
        reason = "pane is active or ordinary progress is visible"
    elif text.strip():
        state = "idle"
        reason = "pane has content but no active/progress signal"
    else:
        state = "unknown"
        reason = "no pane content observed"

    attention = _attention_for_state(state=state, risk=risk)
    observed = _observed_summary(observation)
    fingerprint = notification_fingerprint(
        target=observation.target,
        state=state,
        risk=risk,
        observed=observed,
    )
    score = importance_score(
        state=state,
        risk=risk,
        changed=changed,
        previous_state=previous_state,
        fingerprint=fingerprint,
        now=now,
        min_notify_seconds=min_notify_seconds,
    )
    action = notification_action(score=score, changed=changed)

    return SentinelPaneState(
        target=observation.target,
        state=state,
        observed=observed,
        reason=reason,
        attention=attention,
        risk=risk,
        changed=changed,
        content_hash=observation.content_hash,
        notification_action=action,
        importance_score=score,
        notification_fingerprint=fingerprint,
    )


def build_sentinel_report(
    *,
    observations: list[SentinelObservation],
    previous_state: dict[str, Any] | None = None,
    resources: dict[str, Any] | None = None,
    min_notify_seconds: int = 900,
    now: datetime | None = None,
) -> dict[str, Any]:
    prev_hashes = previous_hashes(previous_state)
    pane_states: list[SentinelPaneState] = []
    for observation in observations:
        changed = prev_hashes.get(observation.target) != observation.content_hash
        pane_states.append(
            classify_observation(
                observation,
                changed=changed,
                previous_state=previous_state,
                min_notify_seconds=min_notify_seconds,
                now=now,
            )
        )

    changed_states = [state for state in pane_states if state.changed]
    notification_candidates = [
        state for state in pane_states if state.notification_action == "slack_candidate"
    ]
    report = {
        "schema_version": SENTINEL_STATE_VERSION,
        "read_only": True,
        "resources": dict(resources or {}),
        "pane_states": [state.as_dict() for state in pane_states],
        "changed_panes": [state.target for state in changed_states],
        "notifications": {
            "slack": "simulated",
            "sent": False,
            "candidates": [state.as_dict() for state in notification_candidates],
        },
        "summary": summarize_report(pane_states),
    }
    if notification_candidates:
        report["notifications"]["message_preview"] = format_slack_candidate_text(report)
    return report


def next_state_from_report(
    *,
    report: dict[str, Any],
    observations: list[SentinelObservation],
    previous_state: dict[str, Any] | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    now_iso = (now or datetime.now(UTC)).isoformat()
    previous = previous_state if isinstance(previous_state, dict) else {}
    notification_state = dict(previous.get("notifications") or {})
    for candidate in report.get("notifications", {}).get("candidates", []):
        if not isinstance(candidate, dict):
            continue
        fingerprint = str(candidate.get("notification_fingerprint") or "").strip()
        if fingerprint:
            notification_state[fingerprint] = {
                "last_candidate_at": now_iso,
                "target": candidate.get("target"),
                "state": candidate.get("state"),
            }

    pane_state_by_target = {
        str(item.get("target")): item
        for item in report.get("pane_states", [])
        if isinstance(item, dict) and item.get("target")
    }
    panes = {}
    for observation in observations:
        panes[observation.target] = {
            "content_hash": observation.content_hash,
            "last_seen_at": now_iso,
            "last_nonempty_line": observation.last_nonempty_line,
            "state": pane_state_by_target.get(observation.target, {}).get("state"),
        }

    return {
        "schema_version": SENTINEL_STATE_VERSION,
        "last_observed_at": now_iso,
        "last_report": {
            "summary": report.get("summary"),
            "model_summary": report.get("model_summary"),
            "changed_panes": report.get("changed_panes", []),
            "notifications": report.get("notifications", {}),
        },
        "panes": panes,
        "notifications": notification_state,
    }


def format_slack_candidate_text(report: dict[str, Any]) -> str:
    candidates = report.get("notifications", {}).get("candidates", [])
    lines = ["orchestrator sentinel: attention candidate"]
    for item in candidates[:5] if isinstance(candidates, list) else []:
        if not isinstance(item, dict):
            continue
        lines.append(
            f"- {item.get('target')}: {item.get('state')} "
            f"({item.get('attention')}) - {item.get('observed')}"
        )
    return "\n".join(lines)


def observe_tmux_panes(
    *,
    list_panes: Callable[[], list[dict[str, Any]]],
    capture_snapshot: Callable[..., dict[str, Any]],
    lines: int = 80,
    captured_at: str | None = None,
) -> tuple[list[SentinelObservation], list[dict[str, Any]]]:
    observations: list[SentinelObservation] = []
    errors: list[dict[str, Any]] = []
    for pane in list_panes():
        try:
            snapshot = capture_snapshot(
                pane=pane,
                lines=int(lines),
                include_history=False,
                history_lines=None,
            )
            observations.append(build_observation(pane=pane, snapshot=snapshot, captured_at=captured_at))
        except Exception as exc:
            errors.append(
                {
                    "target": pane.get("target") or pane.get("pane_id") or "unknown",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    return observations, errors


def summarize_report(pane_states: list[SentinelPaneState]) -> str:
    if not pane_states:
        return "No readable panes observed."
    changed = sum(1 for state in pane_states if state.changed)
    attention = [state for state in pane_states if state.notification_action == "slack_candidate"]
    state_counts: dict[str, int] = {}
    for state in pane_states:
        state_counts[state.state] = int(state_counts.get(state.state, 0)) + 1
    count_text = ", ".join(f"{key}={value}" for key, value in sorted(state_counts.items()))
    if attention:
        return f"Observed {len(pane_states)} pane(s), {changed} changed; attention candidates={len(attention)}; {count_text}."
    return f"Observed {len(pane_states)} pane(s), {changed} changed; {count_text}."


def notification_fingerprint(*, target: str, state: str, risk: bool, observed: str) -> str:
    base = normalize_pane_text(observed).lower()[:240]
    raw = f"{target}|{state}|{int(bool(risk))}|{base}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def importance_score(
    *,
    state: str,
    risk: bool,
    changed: bool,
    previous_state: dict[str, Any] | None,
    fingerprint: str,
    now: datetime | None,
    min_notify_seconds: int,
) -> int:
    score = 0
    if changed:
        score += 2
    if risk:
        score += 5
    if state in {"blocked", "waiting_for_human", "error"}:
        score += 4
    elif state == "complete":
        score += 3
    elif state == "active":
        score += 1
    elif state == "idle":
        score += 1

    if _recently_seen_notification(
        previous_state=previous_state,
        fingerprint=fingerprint,
        now=now,
        min_notify_seconds=min_notify_seconds,
    ):
        score -= 5
    if not changed and state in {"active", "idle", "unknown"}:
        score -= 3
    return max(0, int(score))


def notification_action(*, score: int, changed: bool) -> NotificationAction:
    if score >= 6 and changed:
        return "slack_candidate"
    if score >= 2 or changed:
        return "log_only"
    return "none"


def _recently_seen_notification(
    *,
    previous_state: dict[str, Any] | None,
    fingerprint: str,
    now: datetime | None,
    min_notify_seconds: int,
) -> bool:
    previous = previous_state if isinstance(previous_state, dict) else {}
    notifications = previous.get("notifications")
    if not isinstance(notifications, dict):
        return False
    payload = notifications.get(fingerprint)
    if not isinstance(payload, dict):
        return False
    raw = payload.get("last_candidate_at") or payload.get("last_sent_at")
    if not isinstance(raw, str) or not raw.strip():
        return False
    try:
        previous_at = datetime.fromisoformat(raw)
        if previous_at.tzinfo is None:
            previous_at = previous_at.replace(tzinfo=UTC)
        current = now or datetime.now(UTC)
        if current.tzinfo is None:
            current = current.replace(tzinfo=UTC)
        age = (current - previous_at).total_seconds()
    except Exception:
        return False
    return age < max(1, int(min_notify_seconds))


def _attention_for_state(*, state: str, risk: bool) -> str:
    if risk:
        return "human_review"
    if state in {"blocked", "waiting_for_human", "error"}:
        return "human_review"
    if state == "complete":
        return "possible_issue"
    return "none"


def _observed_summary(observation: SentinelObservation) -> str:
    line = (observation.last_nonempty_line or "").strip()
    if line:
        return _truncate(line, 240)
    if observation.current_command:
        return f"current command: {_truncate(observation.current_command, 120)}"
    return "no visible pane content"


def _optional_str(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _truncate(value: str, limit: int) -> str:
    text = str(value or "")
    if len(text) <= int(limit):
        return text
    return text[: max(0, int(limit) - 3)].rstrip() + "..."
