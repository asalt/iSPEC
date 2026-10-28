"""Validate normalized server output. Never parse or execute model-native code."""

from __future__ import annotations

import json
from typing import Any


def with_system_preamble(messages: list[dict[str, Any]], text: str) -> list[dict[str, Any]]:
    """Restore a template preamble without losing context or promoting tool/user text."""
    result = [dict(message) for message in messages]
    text = text.strip()
    if not text:
        raise ValueError("Tool preamble cannot be empty")
    for message in result:
        if message.get("role") == "system":
            content = message.get("content")
            if not isinstance(content, str):
                raise ValueError("Tool preamble requires a text system message")
            if not content.startswith(text):
                message["content"] = text + "\n" + content
            return result
    return [{"role": "system", "content": text}] + result


def structured_reply(
    data: Any, *, tools: list[dict[str, Any]] | None, tool_choice: Any,
) -> tuple[str, list[dict[str, Any]] | None, str]:
    if not isinstance(data, dict) or not isinstance(data.get("choices"), list) or not data["choices"]:
        raise ValueError("Invalid vLLM response envelope")
    choice = data["choices"][0]
    if not isinstance(choice, dict) or not isinstance(choice.get("message"), dict):
        raise ValueError("Invalid vLLM response message")
    finish_reason = choice.get("finish_reason")
    if finish_reason not in {"stop", "tool_calls"}:
        raise ValueError("vLLM response incomplete or stopped without a usable result")
    message = choice["message"]
    content = message.get("content") or ""
    if not isinstance(content, str):
        raise ValueError("Invalid vLLM answer content")
    allowed = {item["function"]["name"] for item in tools or []}
    if tool_choice == "none":
        allowed = set()
    if isinstance(tool_choice, dict):
        allowed &= {tool_choice.get("function", {}).get("name")}
    raw_calls = message.get("tool_calls")
    if raw_calls is None:
        raw_calls = []
    if not isinstance(raw_calls, list):
        raise ValueError("Invalid vLLM tool-call list")
    calls = []
    ids: set[str] = set()
    for call in raw_calls:
        if not isinstance(call, dict) or call.get("type") != "function":
            raise ValueError("Invalid vLLM tool call")
        function = call.get("function")
        if not isinstance(function, dict) or function.get("name") not in allowed:
            raise ValueError("vLLM returned an unavailable tool")
        call_id = call.get("id")
        if not isinstance(call_id, str) or not call_id.strip() or call_id in ids:
            raise ValueError("Invalid or duplicate vLLM tool-call id")
        ids.add(call_id)
        arguments = function.get("arguments")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except (TypeError, ValueError):
                raise ValueError("Invalid vLLM tool arguments") from None
        if not isinstance(arguments, dict):
            raise ValueError("vLLM tool arguments must be an object")
        calls.append({"id": call_id, "type": "function", "function": {
            "name": function["name"], "arguments": json.dumps(arguments, ensure_ascii=False),
        }})
    if (tool_choice == "required" or isinstance(tool_choice, dict)) and not calls:
        raise ValueError("vLLM did not return the required tool call")
    if not content.strip() and not calls:
        raise ValueError("vLLM returned no final answer or tool call")
    return content, calls or None, finish_reason
