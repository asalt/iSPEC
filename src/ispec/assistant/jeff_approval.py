"""Isolated approval experiments. Nothing here issues tickets or executes tools."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import time
from typing import Callable, Literal
from urllib.parse import urlsplit

import requests

from ispec.assistant.reply_interpretation import classify_short_reply
from ispec.prompt import prompt_binding, render_prompt


Label = Literal["yes", "no", "unsure"]
LABELS = ("yes", "no", "unsure")
BINARY_LABELS = ("yes", "no")
PROMPT_PREFIX = "experimental.project_comment.jeff_approval"
QUESTION_SUFFIXES = {
    "baseline": "",
    "affirmation": ".affirmation",
    "authorize": ".authorize",
    "approval": ".approval",
    "save_now": ".save_now",
}
CONTEXT_SUFFIXES = {
    "object": ".context_messages",
    "messages": ".context_array",
    "live_last": ".context_live_last",
    "transcript": ".context_transcript",
    "reply": ".context_reply",
}
CHOICE_SUFFIXES = {
    "original": ".choices",
    "descriptive": ".choices_descriptive",
    "binary": ".choices_binary",
}


@dataclass(frozen=True, kw_only=True)
class ApprovalPromptVariant:
    question: str = "baseline"
    context: str = "object"
    choices: str = "original"

    def __post_init__(self):
        if (
            self.question not in QUESTION_SUFFIXES
            or self.context not in CONTEXT_SUFFIXES
            or self.choices not in CHOICE_SUFFIXES
        ):
            raise ValueError("Unknown approval prompt variant")

    @property
    def families(self) -> dict[str, str]:
        return {
            "question": PROMPT_PREFIX + QUESTION_SUFFIXES[self.question],
            "context": PROMPT_PREFIX + CONTEXT_SUFFIXES[self.context],
            "choices": PROMPT_PREFIX + CHOICE_SUFFIXES[self.choices],
        }

    @property
    def labels(self) -> tuple[str, ...]:
        return BINARY_LABELS if self.choices == "binary" else LABELS


@dataclass(frozen=True, kw_only=True)
class PendingNoteExchange:
    project_id: int
    note: str
    initial_request: str
    assistant_draft: str
    latest_reply: str
    pending: bool = True

    def context(self) -> dict:
        return {
            "pending_project_id": self.project_id,
            "messages": [
                {"role": "user", "content": self.initial_request},
                {"role": "assistant", "content": self.assistant_draft},
                {"role": "user", "content": self.latest_reply},
            ],
        }

    def gate_error(self) -> str | None:
        if self.pending is not True:
            return "no_pending_note"
        if (
            isinstance(self.project_id, bool)
            or not isinstance(self.project_id, int)
            or self.project_id <= 0
        ):
            return "missing_project_id"
        if any(
            not isinstance(value, str) or not value.strip()
            for value in (
                self.note,
                self.initial_request,
                self.assistant_draft,
                self.latest_reply,
            )
        ):
            return "incomplete_exchange"
        if self.note not in self.assistant_draft:
            return "pending_note_not_shown"
        return None


@dataclass(frozen=True, kw_only=True)
class ApprovalTrialSettings:
    min_probability: float = 0.75
    min_margin: float = 0.20
    max_context_chars: int = 16_000
    allow_lexical_fallback: bool = True

    def __post_init__(self):
        for value in (self.min_probability, self.min_margin):
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("Probability and margin thresholds must be in [0, 1]")
        if self.max_context_chars < 1:
            raise ValueError("Context limit must be positive")


@dataclass(frozen=True, kw_only=True)
class ShadowApprovalResult:
    decision: Label
    source: str
    reason: str
    raw_label: Label | None = None
    probabilities: dict[str, float] | None = None
    latency_ms: int = 0
    error: str | None = None

    @property
    def would_save(self) -> bool:
        return self.decision == "yes"

    def to_dict(self) -> dict:
        return {
            **asdict(self),
            "shadow_only": True,
            "would_save": self.would_save,
            "write_outcome": {"status": "not_attempted"},
        }


class JeffUnavailable(RuntimeError):
    """Transport unavailable, timed out, loading, or busy; fallback may be evaluated."""


class JeffContractError(RuntimeError):
    """A configuration/protocol failure, not permission to try another approver."""


@prompt_binding("experimental.project_comment.jeff_approval")
def prepare_approval_request(
    exchange: PendingNoteExchange,
    *,
    model: str = "jeff-latest",
    variant: ApprovalPromptVariant = ApprovalPromptVariant(),
) -> dict:
    families = variant.families
    prompt = render_prompt(families["question"])
    choices = json.loads(render_prompt(families["choices"]).text)
    if (
        not isinstance(choices, dict)
        or tuple(choices) != variant.labels
        or any(
            not isinstance(value, str) or not value.strip()
            for value in choices.values()
        )
    ):
        raise ValueError(
            f"Approval choices must describe {variant.labels} in that order"
        )
    values = {
        "project_id": exchange.project_id,
        "initial_request": exchange.initial_request,
        "assistant_draft": exchange.assistant_draft,
        "latest_reply": exchange.latest_reply,
    }
    values.update(
        {
            f"{key}_json": json.dumps(value, ensure_ascii=False)
            for key, value in tuple(values.items())
            if key != "project_id"
        }
    )
    context = render_prompt(families["context"], values=values).text.rstrip("\n")
    state = (
        context if variant.context in {"transcript", "reply"} else json.loads(context)
    )
    return {
        "state": state,
        "questions": {
            "save_approval": {
                "type": "choice",
                "criteria": choices,
                "instructions": prompt.text.strip(),
            },
        },
        "model": model,
    }


def lexical_pending_approval(reply: str) -> Label:
    """Keep the old interpreter, but do not authorize compound replies by substring."""
    result = classify_short_reply(reply)
    if result.normalized in {"yes", "confirm", "save it"}:
        return "yes"
    return "no" if result.kind == "deny" else "unsure"


def parse_approval_response(
    response: dict, *, labels: tuple[str, ...] = LABELS
) -> tuple[Label, dict[str, float]]:
    if labels not in (LABELS, BINARY_LABELS):
        raise ValueError("Use an explicitly supported approval choice contract")
    try:
        answer = response["answers"]["save_approval"]
        probabilities = answer["probabilities"]
        label = answer["choice"]
        if (
            answer["type"] != "choice"
            or set(probabilities) != set(labels)
            or label not in labels
        ):
            raise ValueError("Wrong choice schema")
        values = list(probabilities.values())
        if any(
            type(value) not in (float, int)
            or not math.isfinite(value)
            or not 0 <= value <= 1
            for value in values
        ):
            raise ValueError("Invalid probabilities")
        if not math.isclose(sum(values), 1.0, abs_tol=1e-5):
            raise ValueError("Probabilities do not sum to one")
        if probabilities[label] != max(values):
            raise ValueError("Choice does not match probabilities")
        return label, dict(probabilities)
    except (KeyError, TypeError, ValueError) as exc:
        raise JeffContractError("invalid_choice_response") from exc


def evaluate_pending_approval(
    exchange: PendingNoteExchange,
    *,
    query: Callable[[dict], dict],
    settings: ApprovalTrialSettings = ApprovalTrialSettings(),
    model: str = "jeff-latest",
    variant: ApprovalPromptVariant = ApprovalPromptVariant(),
    lexical_fallback: Callable[[str], Label] = lexical_pending_approval,
) -> ShadowApprovalResult:
    if error := exchange.gate_error():
        return ShadowApprovalResult(decision="unsure", source="none", reason=error)
    # A character cap is a local guard, not an estimate of the tokenizer's limit.
    # Reject rather than dropping a qualifier or truncating the user's latest reply.
    if (
        len(json.dumps(exchange.context(), ensure_ascii=False))
        > settings.max_context_chars
    ):
        return ShadowApprovalResult(
            decision="unsure", source="none", reason="context_too_large"
        )
    request = prepare_approval_request(exchange, model=model, variant=variant)
    if (
        len(json.dumps(request["state"], ensure_ascii=False))
        > settings.max_context_chars
    ):
        return ShadowApprovalResult(
            decision="unsure", source="none", reason="context_too_large"
        )
    start = time.monotonic()
    try:
        label, probabilities = parse_approval_response(
            query(request), labels=variant.labels
        )
    except JeffUnavailable as exc:
        decision = (
            lexical_fallback(exchange.latest_reply)
            if settings.allow_lexical_fallback
            else "unsure"
        )
        if decision not in LABELS:
            raise ValueError("Invalid lexical fallback result")
        return ShadowApprovalResult(
            decision=decision,
            source="lexical" if settings.allow_lexical_fallback else "none",
            reason="jeff_unavailable",
            error=str(exc),
            latency_ms=round((time.monotonic() - start) * 1000),
        )
    except JeffContractError as exc:
        return ShadowApprovalResult(
            decision="unsure",
            source="none",
            reason="jeff_contract_error",
            error=str(exc),
            latency_ms=round((time.monotonic() - start) * 1000),
        )
    ranked = sorted(probabilities.values(), reverse=True)
    sufficient = (
        ranked[0] >= settings.min_probability
        and ranked[0] - ranked[1] >= settings.min_margin
    )
    return ShadowApprovalResult(
        decision=label if sufficient else "unsure",
        source="jeff",
        reason="choice" if sufficient else "insufficient_probability_or_margin",
        raw_label=label,
        probabilities=probabilities,
        latency_ms=round((time.monotonic() - start) * 1000),
    )


@dataclass(frozen=True, kw_only=True)
class LocalJeffEndpoint:
    """Eval-only HTTP transport matching Client.ask, with no redirects, auth, or retries."""

    url: str
    timeout_seconds: float = 15.0

    def __post_init__(self):
        parsed = urlsplit(self.url)
        if (
            parsed.scheme != "http"
            or parsed.hostname not in {"127.0.0.1", "::1"}
            or parsed.port is None
            or parsed.username
            or parsed.password
            or parsed.path not in {"", "/"}
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("Use an explicit loopback HTTP URL and port")
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("Timeout must be positive")

    def request(self, *, method: str, path: str, payload: dict | None = None) -> dict:
        if (method, path) not in {("GET", "/health"), ("POST", "/v1/systemone")}:
            raise ValueError("Endpoint is not part of this read-only experiment")
        try:
            with requests.Session() as session:
                session.trust_env = False
                response = session.request(
                    method,
                    self.url.rstrip("/") + path,
                    json=payload,
                    timeout=self.timeout_seconds,
                    allow_redirects=False,
                )
        except (requests.ConnectionError, requests.Timeout) as exc:
            raise JeffUnavailable(type(exc).__name__) from exc
        if response.status_code in {503, 529}:
            raise JeffUnavailable(f"http_{response.status_code}")
        if response.status_code != 200:
            raise JeffContractError(f"http_{response.status_code}")
        try:
            result = response.json()
        except ValueError as exc:
            raise JeffContractError("invalid_response_json") from exc
        if not isinstance(result, dict):
            raise JeffContractError("response_not_object")
        return result

    def __call__(self, payload: dict) -> dict:
        return self.request(method="POST", path="/v1/systemone", payload=payload)
