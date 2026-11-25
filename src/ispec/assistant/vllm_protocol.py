"""Validate normalized server output. Never parse or execute model-native code."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any

UNAVAILABLE_CALL_KEY = "ispec_unavailable"


def with_single_system(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge system instructions so templates that attach tool schemas per system message emit them once."""
    result = [dict(message) for message in messages]
    systems = [message for message in result if message.get("role") == "system"]
    if len(systems) < 2:
        return result
    if any(not isinstance(message.get("content"), str) for message in systems):
        raise ValueError("System merge requires text system messages")
    return [{"role": "system", "content": "\n\n".join(message["content"] for message in systems)}] + [
        message for message in result if message.get("role") != "system"
    ]


@dataclass(frozen=True, kw_only=True)
class RenderedPromptBudget:
    """Server-tokenized input and the space reserved for generated output."""

    input_tokens: int
    output_tokens: int
    context_limit: int
    input_limit: int | None

    @classmethod
    def from_response(cls, *, response: Any, output_tokens: int,
                      input_limit: int | None) -> RenderedPromptBudget:
        """Validate the tokenization server's count and model context limit."""
        if not isinstance(response, dict):
            raise ValueError("Invalid vLLM tokenization response")
        count = response.get("count")
        limit = response.get("max_model_len")
        if type(count) is not int or count < 0 or type(limit) is not int or limit <= 0:
            raise ValueError("Invalid vLLM tokenization count/context limit")
        return cls(input_tokens=count, output_tokens=output_tokens,
                   context_limit=limit, input_limit=input_limit)

    def validate(self) -> None:
        """Reject requests exceeding the input budget or total model context."""
        if self.input_limit is not None and self.input_tokens > self.input_limit:
            raise ValueError(
                f"Rendered prompt has {self.input_tokens} tokens; input budget is {self.input_limit}"
            )
        if self.input_tokens + self.output_tokens > self.context_limit:
            raise ValueError(
                f"Rendered prompt ({self.input_tokens}) plus output reserve ({self.output_tokens}) "
                f"exceeds model context ({self.context_limit})"
            )

    def to_dict(self) -> dict[str, int | None]:
        return asdict(self)


def tokenization_request(*, payload: dict[str, Any]) -> dict[str, Any]:
    """Send the same messages, schemas and template options to vLLM's tokenizer."""
    fields = (
        "model", "messages", "tools", "chat_template", "chat_template_kwargs",
        "add_generation_prompt", "continue_final_message", "add_special_tokens",
    )
    return {key: payload[key] for key in fields if key in payload}


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
    for index, call in enumerate(raw_calls):
        if not isinstance(call, dict) or call.get("type") != "function":
            raise ValueError("Invalid vLLM tool call")
        function = call.get("function") if isinstance(call.get("function"), dict) else {}
        name = function.get("name")
        call_id = call.get("id")
        if not isinstance(call_id, str) or not call_id.strip() or call_id in ids:
            call_id = f"call_unavailable_{index}"
        ids.add(call_id)
        arguments = function.get("arguments")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except (TypeError, ValueError):
                arguments = None
        # A call the turn cannot run is kept and answered as an unavailable tool,
        # so the model learns what happened instead of the turn failing.
        problem = (
            "unavailable_tool" if name not in allowed
            else "invalid_arguments" if not isinstance(arguments, dict)
            else None
        )
        entry = {"id": call_id, "type": "function", "function": {
            "name": name if isinstance(name, str) and name else "unknown_tool",
            "arguments": json.dumps(arguments if problem is None else {}, ensure_ascii=False),
        }}
        if problem:
            entry[UNAVAILABLE_CALL_KEY] = problem
        calls.append(entry)
    usable = [call for call in calls if UNAVAILABLE_CALL_KEY not in call]
    if (tool_choice == "required" or isinstance(tool_choice, dict)) and not usable:
        raise ValueError("vLLM did not return the required tool call")
    if not content.strip() and not calls:
        raise ValueError("vLLM returned no final answer or tool call")
    return content, calls or None, finish_reason
