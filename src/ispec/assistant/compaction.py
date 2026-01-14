from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ispec.assistant.service import AssistantReply


MEMORY_SCHEMA_VERSION = 1


def conversation_memory_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "schema_version": {"type": "integer", "enum": [MEMORY_SCHEMA_VERSION]},
            "summary": {"type": "string", "maxLength": 1500},
            "facts": {
                "type": "array",
                "items": {"type": "string", "maxLength": 240},
                "maxItems": 24,
            },
            "open_tasks": {
                "type": "array",
                "items": {"type": "string", "maxLength": 240},
                "maxItems": 24,
            },
            "preferences": {
                "type": "array",
                "items": {"type": "string", "maxLength": 240},
                "maxItems": 12,
            },
            "entities": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "projects": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "id": {"type": "integer"},
                                "title": {"type": "string", "maxLength": 200},
                            },
                            "required": ["id"],
                        },
                        "maxItems": 24,
                    },
                    "people": {
                        "type": "array",
                        "items": {"type": "string", "maxLength": 120},
                        "maxItems": 24,
                    },
                    "experiments": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "id": {"type": "integer"},
                                "title": {"type": "string", "maxLength": 200},
                            },
                            "required": ["id"],
                        },
                        "maxItems": 24,
                    },
                },
                "required": ["projects", "people", "experiments"],
            },
        },
        "required": ["schema_version", "summary", "facts", "open_tasks", "preferences", "entities"],
    }


def _distiller_system_prompt() -> str:
    return (
        "You are a conversation memory distiller for the iSPEC support assistant.\n"
        "Update a structured 'conversation_memory' JSON object given:\n"
        "- previous_memory: existing memory JSON (may be empty)\n"
        "- new_messages: a list of new chat messages to incorporate\n"
        "\n"
        "Rules:\n"
        "- Output ONLY a JSON object matching the provided schema.\n"
        "- Be concise; keep lists short.\n"
        "- Do not store secrets or credentials. Omit API keys, passwords, env vars, file paths, or tokens.\n"
        "- Only include entity IDs if explicitly mentioned in the messages.\n"
        "- Prefer stable facts and ongoing tasks over transient phrasing.\n"
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


def _normalize_memory(memory: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(memory, dict):
        memory = {}
    if memory.get("schema_version") != MEMORY_SCHEMA_VERSION:
        memory = {**memory, "schema_version": MEMORY_SCHEMA_VERSION}
    for key in ("facts", "open_tasks", "preferences"):
        value = memory.get(key)
        if not isinstance(value, list):
            memory[key] = []
    entities = memory.get("entities")
    if not isinstance(entities, dict):
        entities = {}
    for key in ("projects", "people", "experiments"):
        value = entities.get(key)
        if not isinstance(value, list):
            entities[key] = []
    memory["entities"] = entities
    if not isinstance(memory.get("summary"), str):
        memory["summary"] = ""
    return memory


@dataclass(frozen=True)
class MemoryDistillResult:
    memory: dict[str, Any] | None
    reply: AssistantReply


def distill_conversation_memory(
    *,
    previous_memory: dict[str, Any] | None,
    new_messages: list[dict[str, str]],
    generate_reply_fn: Callable[..., AssistantReply],
    max_tokens: int = 900,
) -> MemoryDistillResult:
    schema = conversation_memory_schema()
    previous = _normalize_memory(previous_memory)
    payload = {"previous_memory": previous, "new_messages": list(new_messages or [])}
    messages = [
        {"role": "system", "content": _distiller_system_prompt()},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]

    reply = generate_reply_fn(
        messages=messages,
        tools=None,
        vllm_extra_body={"guided_json": schema, "temperature": 0, "max_tokens": int(max_tokens)},
    )
    parsed = _parse_json_object(reply.content)
    if not isinstance(parsed, dict):
        return MemoryDistillResult(memory=None, reply=reply)

    normalized = _normalize_memory(parsed)
    return MemoryDistillResult(memory=normalized, reply=reply)

