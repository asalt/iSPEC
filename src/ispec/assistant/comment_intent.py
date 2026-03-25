from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from ispec.assistant.service import AssistantReply


def project_comment_intent_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "intent": {
                "type": "string",
                "enum": ["draft_only", "save_now", "confirm_save", "other"],
            },
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "reason": {"type": "string"},
        },
        "required": ["intent", "confidence", "reason"],
    }


def _project_comment_intent_prompt() -> str:
    return (
        "Classify the user's intent for iSPEC project-comment handling.\n"
        "Return only a JSON object matching the schema.\n"
        "\n"
        "intent meanings:\n"
        "- draft_only: the user wants help drafting/rewording a comment or note, but not saving it yet.\n"
        "- save_now: the user explicitly wants a project note/comment/history entry saved now.\n"
        "- confirm_save: the user is confirming that a previously drafted note should now be saved.\n"
        "- other: not really a project-comment drafting/saving request.\n"
        "\n"
        "Prefer draft_only when the request is about wording, drafting, rewriting, or improving a comment.\n"
        "Prefer confirm_save only when the user is clearly confirming an earlier draft/save question."
    )


def _parse_json_object(text: str | None) -> dict[str, Any] | None:
    if not text:
        return None
    raw = str(text).strip()
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        pass

    start = raw.find("{")
    end = raw.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        parsed = json.loads(raw[start : end + 1])
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None


def _validate_intent_decision(decision: dict[str, Any]) -> dict[str, Any] | None:
    intent = decision.get("intent")
    confidence = decision.get("confidence")
    reason = decision.get("reason")
    if intent not in {"draft_only", "save_now", "confirm_save", "other"}:
        return None
    try:
        confidence_value = float(confidence)
    except Exception:
        confidence_value = 0.0
    confidence_value = max(0.0, min(1.0, confidence_value))
    reason_text = str(reason or "").strip()
    return {
        "intent": str(intent),
        "confidence": confidence_value,
        "reason": reason_text,
    }


def decide_project_comment_intent_vllm(
    *,
    user_message: str,
    last_assistant_message: str | None,
    focused_project_id: int | None,
    generate_reply_fn: Callable[..., AssistantReply],
) -> tuple[dict[str, Any] | None, AssistantReply]:
    messages = [
        {"role": "system", "content": _project_comment_intent_prompt()},
        {
            "role": "user",
            "content": json.dumps(
                {
                    "user_message": user_message,
                    "last_assistant_message": last_assistant_message or "",
                    "focused_project_id": focused_project_id,
                },
                ensure_ascii=False,
            ),
        },
    ]
    reply = generate_reply_fn(
        messages=messages,
        tools=None,
        vllm_extra_body={
            "guided_json": project_comment_intent_schema(),
            "max_tokens": 200,
            "temperature": 0,
        },
    )
    parsed = _parse_json_object(reply.content)
    validated = _validate_intent_decision(parsed) if isinstance(parsed, dict) else None
    return validated, reply
