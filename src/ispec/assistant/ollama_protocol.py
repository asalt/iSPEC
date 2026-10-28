"""Native Ollama wire translation; never executes tools or grants permission."""

from __future__ import annotations

import json
from typing import Any
from uuid import uuid4


def native_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    names = {}
    for item in messages:
        role = item.get("role", "user")
        if role not in {"system", "user", "assistant", "tool"}:
            raise ValueError("Unsupported Ollama message role")
        message = {"role": role, "content": str(item.get("content") or "")}
        calls = item.get("tool_calls")
        if calls:
            message["tool_calls"] = []
            for call in calls:
                function = call["function"]
                arguments = function.get("arguments", {})
                if isinstance(arguments, str):
                    arguments = json.loads(arguments)
                if not isinstance(arguments, dict):
                    raise ValueError("Tool arguments must be an object")
                name = function["name"]
                if call.get("id"):
                    names[call["id"]] = name
                message["tool_calls"].append({"type": "function", "function": {
                    "name": name, "arguments": arguments,
                }})
        if role == "tool":
            name = item.get("tool_name") or item.get("name") or names.get(item.get("tool_call_id"))
            if not name:
                raise ValueError("Tool result has no matching tool name")
            message["tool_name"] = name
        result.append(message)
    # Some native renderers (including OLMo 3) retain only the first system
    # message. Preserve all system instructions, without promoting user/tool text.
    systems = [message["content"] for message in result if message["role"] == "system"]
    if len(systems) > 1:
        result = [{"role": "system", "content": "\n\n".join(systems)}] + [
            message for message in result if message["role"] != "system"
        ]
    return result


def selected_tools(tools: list[dict[str, Any]] | None, choice: Any) -> list[dict[str, Any]]:
    tools = list(tools or [])
    if choice == "none":
        return []
    if isinstance(choice, dict):
        name = choice.get("function", {}).get("name")
        tools = [tool for tool in tools if tool.get("function", {}).get("name") == name]
        if not name or not tools:
            raise ValueError("Requested tool is not available")
    elif choice not in (None, "auto", "required"):
        raise ValueError("Unsupported tool choice")
    if choice == "required" and not tools:
        raise ValueError("Required tool choice has no tools")
    return tools


def request_options(extra: dict[str, Any] | None) -> tuple[dict[str, Any], dict[str, Any], bool]:
    """Translate existing caller options without accepting payload overrides."""
    extra = extra or {}
    options = {}
    for source, target in (("max_tokens", "num_predict"), ("temperature", "temperature"),
                           ("top_p", "top_p"), ("seed", "seed"), ("stop", "stop")):
        if source in extra:
            options[target] = extra[source]
    structured = extra.get("structured_outputs") or {}
    schema = structured.get("json", extra.get("guided_json"))
    choices = structured.get("choice", extra.get("guided_choice"))
    response_format = extra.get("response_format") or {}
    if response_format.get("type") == "json_schema":
        schema = response_format["json_schema"]["schema"]
    if choices is not None:
        schema = {"type": "string", "enum": choices}
    fields = {}
    if schema is not None:
        fields["format"] = schema
    elif response_format.get("type") == "json_object":
        fields["format"] = "json"
    if any(key in structured for key in ("regex", "grammar")) or any(
        key in extra for key in ("guided_regex", "guided_grammar")
    ):
        raise ValueError("Ollama native adapter does not support regex/grammar constraints")
    return options, fields, choices is not None


def response_tools(raw: Any, tools: list[dict[str, Any]], choice: Any) -> list[dict[str, Any]] | None:
    result = []
    allowed = {tool["function"]["name"] for tool in tools}
    if raw is not None and not isinstance(raw, list):
        raise ValueError("Invalid Ollama tool-call list")
    for call in raw or []:
        function = call.get("function") or {}
        name = function.get("name")
        arguments = function.get("arguments")
        if name not in allowed or not isinstance(arguments, dict):
            raise ValueError("Ollama returned an unavailable tool or invalid arguments")
        result.append({"id": f"ollama_{uuid4().hex}", "type": "function", "function": {
            "name": name, "arguments": json.dumps(arguments, ensure_ascii=False),
        }})
    if (choice == "required" or isinstance(choice, dict)) and not result:
        raise ValueError("Ollama did not return the required tool call")
    return result or None
