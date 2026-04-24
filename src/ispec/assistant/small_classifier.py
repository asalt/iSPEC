from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from ispec.prompt import RenderedPrompt, load_bound_prompt, prompt_binding, prompt_observability_context


@dataclass(frozen=True)
class SmallClassifierTaskSpec:
    name: str
    labels: tuple[str, ...]
    required_input_keys: tuple[str, ...]
    binding_callable: Callable[[], str]
    max_tokens: int = 160
    temperature: float = 0.0

    @property
    def prompt_family(self) -> str:
        return str(getattr(self.binding_callable, "__prompt_family__", "") or "")


@dataclass(frozen=True)
class SmallClassifierDecision:
    task_name: str
    label: str
    confidence: float
    reason: str


@dataclass(frozen=True)
class PreparedSmallClassifierTask:
    task: SmallClassifierTaskSpec
    prompt: RenderedPrompt
    messages: list[dict[str, Any]]
    schema: dict[str, Any]

    def observability_context(self, *, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        payload = {"surface": "small_classifier_scaffold", "task": self.task.name}
        if extra:
            payload.update(extra)
        return prompt_observability_context(self.prompt, extra=payload)


_SAVE_REPLY_INTERPRETATION_LABELS = ("approve", "deny", "defer", "modify", "unclear")


@prompt_binding("assistant.small_classifier.save_reply_interpretation")
def _save_reply_interpretation_prompt() -> str:
    return load_bound_prompt(_save_reply_interpretation_prompt).text


_SMALL_CLASSIFIER_TASKS: dict[str, SmallClassifierTaskSpec] = {
    "save_reply_interpretation": SmallClassifierTaskSpec(
        name="save_reply_interpretation",
        labels=_SAVE_REPLY_INTERPRETATION_LABELS,
        required_input_keys=(
            "user_message",
            "last_assistant_message",
            "awaiting_state",
            "pending_action",
        ),
        binding_callable=_save_reply_interpretation_prompt,
        max_tokens=160,
        temperature=0.0,
    )
}


def small_classifier_task_names() -> tuple[str, ...]:
    return tuple(sorted(_SMALL_CLASSIFIER_TASKS))


def get_small_classifier_task(task_name: str) -> SmallClassifierTaskSpec:
    cleaned = str(task_name or "").strip()
    task = _SMALL_CLASSIFIER_TASKS.get(cleaned)
    if task is None:
        names = ", ".join(small_classifier_task_names()) or "<none>"
        raise KeyError(f"Unknown small-classifier task {cleaned!r}. Known tasks: {names}")
    return task


def small_classifier_schema(task_name: str) -> dict[str, Any]:
    task = get_small_classifier_task(task_name)
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "label": {"type": "string", "enum": list(task.labels)},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "reason": {"type": "string"},
        },
        "required": ["label", "confidence", "reason"],
    }


def build_small_classifier_task(
    task_name: str,
    *,
    payload: Mapping[str, Any],
) -> PreparedSmallClassifierTask:
    task = get_small_classifier_task(task_name)
    missing = [key for key in task.required_input_keys if key not in payload]
    if missing:
        missing_text = ", ".join(missing)
        raise ValueError(f"Missing required small-classifier payload keys for {task.name!r}: {missing_text}")
    prompt = load_bound_prompt(task.binding_callable)
    messages = [
        {"role": "system", "content": prompt.text},
        {"role": "user", "content": json.dumps(dict(payload), ensure_ascii=False)},
    ]
    return PreparedSmallClassifierTask(
        task=task,
        prompt=prompt,
        messages=messages,
        schema=small_classifier_schema(task.name),
    )


def normalize_small_classifier_decision(
    task_name: str,
    decision: Mapping[str, Any] | None,
) -> SmallClassifierDecision | None:
    if not isinstance(decision, Mapping):
        return None
    task = get_small_classifier_task(task_name)
    label = str(decision.get("label") or "").strip()
    if label not in task.labels:
        return None
    try:
        confidence = float(decision.get("confidence") or 0.0)
    except Exception:
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))
    reason = str(decision.get("reason") or "").strip()
    return SmallClassifierDecision(task_name=task.name, label=label, confidence=confidence, reason=reason)
