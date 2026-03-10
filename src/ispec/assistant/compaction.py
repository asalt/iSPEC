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

    # Some model outputs incorrectly wrap the payload in a "conversation_memory"
    # field. Prefer the inner object when present.
    wrapped = memory.get("conversation_memory")
    if isinstance(wrapped, dict) and wrapped:
        memory = wrapped

    def clean_str(value: Any, *, max_len: int) -> str:
        if not isinstance(value, str):
            value = str(value) if value is not None else ""
        value = value.strip()
        if not value:
            return ""
        if len(value) > max_len:
            value = value[:max_len]
        return value

    def clean_str_list(value: Any, *, max_items: int, max_len: int) -> list[str]:
        if not isinstance(value, list):
            return []
        items: list[str] = []
        for item in value:
            if not isinstance(item, str):
                continue
            cleaned = clean_str(item, max_len=max_len)
            if not cleaned:
                continue
            items.append(cleaned)
            if len(items) >= max_items:
                break
        return items

    def clean_id_item(item: Any, *, max_title_len: int) -> dict[str, Any] | None:
        pid: int | None = None
        title: str | None = None
        if isinstance(item, dict):
            raw_id = item.get("id")
            if isinstance(raw_id, int):
                pid = raw_id
            elif isinstance(raw_id, str) and raw_id.strip().isdigit():
                pid = int(raw_id.strip())
            raw_title = item.get("title")
            if isinstance(raw_title, str) and raw_title.strip():
                title = clean_str(raw_title, max_len=max_title_len)
        elif isinstance(item, int):
            pid = item
        elif isinstance(item, str) and item.strip().isdigit():
            pid = int(item.strip())
        if pid is None:
            return None
        if pid < 0:
            return None
        payload: dict[str, Any] = {"id": int(pid)}
        if title:
            payload["title"] = title
        return payload

    entities_raw = memory.get("entities") if isinstance(memory.get("entities"), dict) else {}

    projects_raw = entities_raw.get("projects") if isinstance(entities_raw.get("projects"), list) else []
    projects: list[dict[str, Any]] = []
    for item in projects_raw:
        cleaned = clean_id_item(item, max_title_len=200)
        if cleaned is None:
            continue
        projects.append(cleaned)
        if len(projects) >= 24:
            break

    experiments_raw = entities_raw.get("experiments") if isinstance(entities_raw.get("experiments"), list) else []
    experiments: list[dict[str, Any]] = []
    for item in experiments_raw:
        cleaned = clean_id_item(item, max_title_len=200)
        if cleaned is None:
            continue
        experiments.append(cleaned)
        if len(experiments) >= 24:
            break

    people = clean_str_list(entities_raw.get("people"), max_items=24, max_len=120)

    normalized: dict[str, Any] = {
        "schema_version": MEMORY_SCHEMA_VERSION,
        "summary": clean_str(memory.get("summary"), max_len=1500),
        "facts": clean_str_list(memory.get("facts"), max_items=24, max_len=240),
        "open_tasks": clean_str_list(memory.get("open_tasks"), max_items=24, max_len=240),
        "preferences": clean_str_list(memory.get("preferences"), max_items=12, max_len=240),
        "entities": {
            "projects": projects,
            "people": people,
            "experiments": experiments,
        },
    }

    # Collapse a fully-empty memory to {} to keep prompts smaller.
    entities = normalized.get("entities") if isinstance(normalized.get("entities"), dict) else {}
    has_entities = any(bool(entities.get(k)) for k in ("projects", "people", "experiments"))
    has_content = bool(normalized.get("summary")) or bool(normalized.get("facts")) or bool(
        normalized.get("open_tasks")
    ) or bool(normalized.get("preferences")) or bool(has_entities)
    return normalized if has_content else {}


def normalize_conversation_memory(memory: dict[str, Any] | None) -> dict[str, Any]:
    """Normalize conversation_memory blobs for safe use in prompts/state.

    This enforces the schema shape, unwraps accidental wrapper objects, and
    returns {} when there's no meaningful content.
    """

    return _normalize_memory(memory)


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
