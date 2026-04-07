from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import requests

from ispec.assistant.tools import TOOL_CALL_PREFIX, TOOL_RESULT_PREFIX, tool_prompt
from ispec.assistant.usage_logging import record_inference_usage_event
from ispec.logging import get_logger
from ispec.prompt import PromptSource, PromptVersionInfo, RenderedPrompt, load_bound_prompt, prompt_binding, prompt_observability_context


logger = get_logger(__name__)

ResponseFormat = Literal["single", "compare"]

@dataclass(frozen=True)
class AssistantReply:
    content: str
    provider: str
    model: str | None = None
    meta: dict[str, Any] | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_parser_fallback_used = False
    tool_parser_fallback_shape: str | None = None
    ok: bool = True
    error: str | None = None


_PROMPT_FILE_MAX_CHARS = 80_000


def _normalize_openai_base_url(value: str) -> str:
    """Normalize OpenAI-style base URLs.

    Accepts values like:
      - http://host:8000
      - http://host:8000/v1
      - http://host:8000/v1/chat/completions

    Returns the base URL without the /v1 suffix or endpoint path.
    """

    raw = (value or "").strip().rstrip("/")
    if not raw:
        return ""

    for suffix in ("/v1/chat/completions", "/v1/models", "/v1"):
        if raw.endswith(suffix):
            raw = raw[: -len(suffix)].rstrip("/")
            break

    return raw


def _truncate_text(value: str, *, limit: int = 2000) -> str:
    if limit <= 0:
        return ""
    text = (value or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit]


def _read_text_file(path: str, *, max_chars: int = _PROMPT_FILE_MAX_CHARS) -> str | None:
    try:
        content = Path(path).expanduser().read_text(encoding="utf-8")
    except Exception:
        return None
    content = (content or "").strip()
    if not content:
        return None
    if max_chars > 0 and len(content) > max_chars:
        content = content[:max_chars]
    return content


def _read_prompt_from_env(path_env: str) -> str | None:
    raw = (os.getenv(path_env) or "").strip()
    if not raw:
        return None
    return _read_text_file(raw)


def _prompt_extras() -> str | None:
    """Return optional prompt additions appended to the default system prompts."""

    chunks: list[str] = []

    file_extra = _read_prompt_from_env("ISPEC_ASSISTANT_SYSTEM_PROMPT_EXTRA_PATH")
    if file_extra:
        chunks.append(file_extra.strip())

    extra = (os.getenv("ISPEC_ASSISTANT_SYSTEM_PROMPT_EXTRA") or "").strip()
    if extra:
        chunks.append(extra)

    combined = "\n\n".join(chunk for chunk in chunks if chunk)
    return combined.strip() or None


def _assistant_identity() -> str:
    return (os.getenv("ISPEC_ASSISTANT_NAME") or "iSPEC").strip() or "iSPEC"


def _prompt_text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _overlay_rendered_prompt(
    prompt: RenderedPrompt,
    *,
    text: str,
    source_path: str | None = None,
) -> RenderedPrompt:
    normalized = (text or "").strip()
    source = PromptSource(
        family=prompt.source.family,
        source_path=source_path or prompt.source.source_path,
        title=prompt.source.title,
        notes=prompt.source.notes,
        body=normalized,
        body_sha256=_prompt_text_sha256(normalized),
    )
    return RenderedPrompt(text=normalized, source=source, binding=prompt.binding, version=PromptVersionInfo())


def _apply_prompt_extras(prompt: RenderedPrompt) -> RenderedPrompt:
    extras = _prompt_extras()
    if not extras:
        return prompt
    return _overlay_rendered_prompt(prompt, text=prompt.text.rstrip() + "\n\n" + extras)


def _legacy_answer_prompt_override() -> tuple[str | None, str | None]:
    prompt_text: str | None = None
    source_path: str | None = None

    file_override = _read_prompt_from_env("ISPEC_ASSISTANT_SYSTEM_PROMPT_PATH")
    if file_override:
        prompt_text = file_override.strip()
        source_path = "env:ISPEC_ASSISTANT_SYSTEM_PROMPT_PATH"

    override = (os.getenv("ISPEC_ASSISTANT_SYSTEM_PROMPT") or "").strip()
    if override:
        prompt_text = override.strip()
        source_path = "env:ISPEC_ASSISTANT_SYSTEM_PROMPT"

    return prompt_text, source_path


def _final_template(*, response_format: ResponseFormat) -> str:
    if response_format == "compare":
        return (
            "  FINAL_A:\n"
            "  <draft answer A (concise)>\n"
            "  FINAL_B:\n"
            "  <draft answer B (alternative phrasing/structure; can be slightly more detailed)>\n"
        )
    return "  FINAL:\n  <your user-facing answer>\n"


@prompt_binding("assistant.base.system")
def _base_system_prompt() -> str:
    return _render_base_system_prompt().text


def _render_base_system_prompt() -> RenderedPrompt:
    return load_bound_prompt(_base_system_prompt, values={"identity": _assistant_identity()})


def _planner_tool_use_block(*, tool_names: set[str] | None = None) -> str:
    def has(name: str) -> bool:
        return tool_names is None or name in tool_names

    tool_use_lines = [
        "Tool use (optional):",
        "- If you need more iSPEC DB info than CONTEXT provides, request a tool.",
        "- Never invent database values, IDs, or outcomes.",
    ]
    if has("count_all_projects"):
        tool_use_lines.append(
            "- For global count/list questions (e.g. 'how many projects'), do not infer from CONTEXT; use count_all_projects."
        )
    if has("count_current_projects"):
        tool_use_lines.append("- For current/active project counts, use count_current_projects.")
    if has("project_counts_snapshot"):
        tool_use_lines.append(
            "- For a single snapshot that includes both total+current counts plus status breakdown, use project_counts_snapshot."
        )
    if has("latest_projects") or has("latest_project_comments"):
        tool_use_lines.append(
            "- For 'latest projects' / 'recent changes', use latest_projects and latest_project_comments."
        )
    if has("my_projects"):
        tool_use_lines.append(
            "- For questions like 'what projects can I view' / 'my projects', use my_projects."
        )
    if has("project_files_for_project"):
        tool_use_lines.append(
            "- For questions about uploaded files/plots attached to a project, use project_files_for_project."
        )
    if has("experiments_for_project"):
        tool_use_lines.append("- For experiments in a specific project, use experiments_for_project.")
    if has("create_project_comment"):
        tool_use_lines.append(
            "- For collaborative project work, draft notes first; only write to project history if the user explicitly asks you to save."
        )
        tool_use_lines.append(
            "- Requests like 'help me write a comment about project 1531' are drafting requests. Offer a draft and ask for confirmation or tweaks before saving anything."
        )
        tool_use_lines.append(
            "- For drafting requests, write the draft text directly. Do not ask the user to emit TOOL_CALL syntax or name the create_project_comment tool."
        )
        tool_use_lines.append(
            "- If the user explicitly asks you to add/save/log a project note/comment (e.g. 'make a note for project 1478'), call create_project_comment immediately (confirm=true) and then confirm using the tool result (include comment_id)."
        )
        tool_use_lines.append(
            "- If you asked the user to confirm saving a drafted note and they confirm (e.g. 'yes' / 'ok'), call create_project_comment and then confirm using the tool result (include comment_id)."
        )
    if has("repo_search") or has("repo_list_files") or has("repo_read_file"):
        tool_use_lines.append(
            "- For code searches in the iSPEC repo (dev-only), use repo_search/repo_list_files/repo_read_file."
        )
    if has("assistant_list_tools"):
        tool_use_lines.append(
            "- If you're unsure which tool can solve a request, call assistant_list_tools(query=...)."
        )
    tool_use_lines.extend(
        [
            "- If the user explicitly asks you to use a tool, call the appropriate tool.",
            "- If the user requests a specific tool by name but it is not listed under Available tools, say you cannot call it (unavailable/permissions/config) and ask to enable it. Do not substitute a different tool silently.",
            "- Do not tell the user to run CLI commands or use the web app to run tools; you can call tools directly.",
            "- Never show the literal TOOL_CALL protocol or ask the user to paste tool-call JSON.",
            "- For any write action, do not claim it succeeded unless you already have an ok=true TOOL_RESULT from the write tool. If a write is needed, choose the tool instead of narrating a fake success.",
            "- When answering from a TOOL_RESULT, restate the scope (e.g. total vs current). Do not call a subset count 'total'.",
        ]
    )
    return "\n".join(tool_use_lines)


def _planner_tool_protocol_block(*, tools_available: bool, provider: str | None = None) -> str:
    normalized_provider = str(provider or _assistant_provider()).strip().lower()
    if tools_available:
        if normalized_provider == "vllm":
            return (
                "Tool calling protocol:\n"
                "- The active local parser expects parser-native JSON tool calls.\n"
                "- When calling a single tool, output only a JSON object like:\n"
                '  {"name":"<tool>","arguments":{...}}\n'
                "- When calling multiple tools, output only a JSON array of those objects.\n"
                '- Use top-level "name" and "arguments" keys. Do not wrap calls inside '
                '"tool_calls", "type":"function", or another envelope.\n'
                "- When calling tools, do not include PLAN/FINAL or any extra prose.\n"
                f"- If raw JSON tool calls are not supported, fallback to one line starting with {TOOL_CALL_PREFIX}:\n"
                f'  {TOOL_CALL_PREFIX} {{"name":"<tool>","arguments":{{...}}}}\n'
                f"- Tool results may arrive as a {TOOL_RESULT_PREFIX} system message or a role=tool message; treat them as authoritative."
            )
        return (
            "Tool calling protocol:\n"
            "- Use OpenAI-style tool_calls (tools/tool_choice are provided).\n"
            "- When calling tools, do not include PLAN/FINAL in the content.\n"
            f"- If structured tool_calls are not supported, fallback to one line starting with {TOOL_CALL_PREFIX}:\n"
            f'  {TOOL_CALL_PREFIX} {{"name":"<tool>","arguments":{{...}}}}\n'
            f"- Tool results may arrive as a {TOOL_RESULT_PREFIX} system message or a role=tool message; treat them as authoritative."
        )
    return (
        "Tool calling protocol:\n"
        f"- Request a tool by outputting exactly one line starting with {TOOL_CALL_PREFIX}:\n"
        f'  {TOOL_CALL_PREFIX} {{"name":"<tool>","arguments":{{...}}}}\n'
        f"- Tool results arrive as a {TOOL_RESULT_PREFIX} system message; treat them as authoritative."
    )

def _planner_response_format_block(*, response_format: ResponseFormat) -> str:
    return (
        "Response format:\n"
        "- If you call a tool: output only the tool call (or tool_calls), with no extra text.\n"
        "- Otherwise, output only:\n"
        + _final_template(response_format=response_format).rstrip()
    )


@prompt_binding("assistant.answer.system")
def _system_prompt_answer(*, response_format: ResponseFormat = "single") -> str:
    return _render_system_prompt_answer(response_format=response_format).text


def _render_system_prompt_answer(*, response_format: ResponseFormat = "single") -> RenderedPrompt:
    prompt = load_bound_prompt(
        _system_prompt_answer,
        values={
            "base_prompt": _render_base_system_prompt().text,
            "response_format_block": _final_template(response_format=response_format).rstrip(),
        },
    )
    override, override_source = _legacy_answer_prompt_override()
    if override:
        prompt = _overlay_rendered_prompt(prompt, text=override, source_path=override_source)
    return _apply_prompt_extras(prompt)


@prompt_binding("assistant.planner.system")
def _system_prompt_planner(
    *,
    tools_available: bool,
    response_format: ResponseFormat = "single",
    tool_names: set[str] | None = None,
) -> str:
    return _render_system_prompt_planner(
        tools_available=tools_available,
        response_format=response_format,
        tool_names=tool_names,
    ).text


def _render_system_prompt_planner(
    *,
    tools_available: bool,
    response_format: ResponseFormat = "single",
    tool_names: set[str] | None = None,
) -> RenderedPrompt:
    prompt = load_bound_prompt(
        _system_prompt_planner,
        values={
            "base_prompt": _render_base_system_prompt().text,
            "tool_use_block": _planner_tool_use_block(tool_names=tool_names),
            "tool_protocol_block": _planner_tool_protocol_block(
                tools_available=tools_available,
                provider=(os.getenv("ISPEC_ASSISTANT_PROVIDER") or "stub"),
            ),
            "tool_catalog_block": tool_prompt(tool_names=tool_names).strip() + "\n",
            "response_format_block": _planner_response_format_block(response_format=response_format),
        },
    )
    return _apply_prompt_extras(prompt)


@prompt_binding("assistant.review.system")
def _system_prompt_review() -> str:
    return _render_system_prompt_review().text


def _render_system_prompt_review() -> RenderedPrompt:
    prompt = load_bound_prompt(
        _system_prompt_review,
        values={"base_prompt": _render_base_system_prompt().text},
    )
    return _apply_prompt_extras(prompt)


@prompt_binding("assistant.review_decider.system")
def _system_prompt_review_decider() -> str:
    return _render_system_prompt_review_decider().text


def _render_system_prompt_review_decider() -> RenderedPrompt:
    prompt = load_bound_prompt(
        _system_prompt_review_decider,
        values={"base_prompt": _render_base_system_prompt().text},
    )
    return _apply_prompt_extras(prompt)


def _system_prompt() -> str:
    """Backwards-compatible alias (answer stage)."""

    return _system_prompt_answer()


def _ollama_url() -> str:
    return (os.getenv("ISPEC_OLLAMA_URL") or "http://127.0.0.1:11434").rstrip("/")


def _ollama_model() -> str:
    return (os.getenv("ISPEC_OLLAMA_MODEL") or "llama3.2:2b").strip()


def _ollama_timeout_seconds() -> float:
    raw = (os.getenv("ISPEC_OLLAMA_TIMEOUT_SECONDS") or "").strip()
    if not raw:
        return 30.0
    try:
        return max(1.0, float(raw))
    except ValueError:
        return 30.0


def _vllm_url() -> str:
    raw = os.getenv("ISPEC_VLLM_URL") or "http://127.0.0.1:8000"
    normalized = _normalize_openai_base_url(raw)
    return (normalized or raw).rstrip("/")


def _vllm_model() -> str | None:
    raw = (os.getenv("ISPEC_VLLM_MODEL") or "").strip()
    return raw or None


def _vllm_api_key() -> str | None:
    raw = (os.getenv("ISPEC_VLLM_API_KEY") or "").strip()
    return raw or None


def _vllm_timeout_seconds() -> float:
    raw = (os.getenv("ISPEC_VLLM_TIMEOUT_SECONDS") or "").strip()
    if not raw:
        return 30.0
    try:
        return max(1.0, float(raw))
    except ValueError:
        return 30.0


def _vllm_headers() -> dict[str, str]:
    api_key = _vllm_api_key()
    if not api_key:
        return {}
    return {"Authorization": f"Bearer {api_key}"}


def _resolve_vllm_model(*, base_url: str, headers: dict[str, str], timeout: float) -> str | None:
    model = _vllm_model()
    if model:
        return model

    url = f"{base_url}/v1/models"
    response = requests.get(url, headers=headers, timeout=timeout)
    response.raise_for_status()
    payload = response.json()

    if not isinstance(payload, dict):
        return None
    data = payload.get("data")
    if not isinstance(data, list) or not data:
        return None
    first = data[0]
    if not isinstance(first, dict):
        return None
    value = (first.get("id") or "").strip()
    return value or None


def _history_limit() -> int:
    raw = (os.getenv("ISPEC_ASSISTANT_HISTORY_LIMIT") or "").strip()
    if not raw:
        return 20
    try:
        return max(0, int(raw))
    except ValueError:
        return 20


def _assistant_temperature() -> float | None:
    raw = (os.getenv("ISPEC_ASSISTANT_TEMPERATURE") or "").strip()
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    if value < 0:
        value = 0.0
    if value > 2:
        value = 2.0
    return value


def generate_reply(
    *,
    message: str | None = None,
    history: list[dict[str, Any]] | None = None,
    context: str | None = None,
    messages: list[dict[str, Any]] | None = None,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | dict[str, Any] | None = None,
    stage: Literal["planner", "answer", "review"] = "answer",
    vllm_extra_body: dict[str, Any] | None = None,
    observability_context: dict[str, Any] | None = None,
) -> AssistantReply:
    provider = (os.getenv("ISPEC_ASSISTANT_PROVIDER") or "stub").strip().lower()
    if messages is None:
        if message is None:
            raise ValueError("message is required when messages is not provided")
        stage_prompt = _render_stage_prompt(stage=stage, tools_available=bool(tools))
        messages = _build_messages(
            message=message,
            history=history,
            context=context,
            stage=stage,
            system_prompt=stage_prompt.text,
        )
        observability_context = prompt_observability_context(stage_prompt, extra=observability_context)
    if provider == "ollama":
        return _generate_ollama_reply(messages=messages, tools=tools, observability_context=observability_context)
    if provider == "vllm":
        return _generate_vllm_reply(
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            extra_body=vllm_extra_body,
            observability_context=observability_context,
        )
    return AssistantReply(
        content=(
            "Support assistant is running in stub mode. "
            "Set `ISPEC_ASSISTANT_PROVIDER=ollama` or `ISPEC_ASSISTANT_PROVIDER=vllm` "
            "to enable a local model."
        ),
        provider="stub",
        model=None,
        meta=None,
    )


def _render_stage_prompt(
    *,
    stage: Literal["planner", "answer", "review"],
    tools_available: bool,
) -> RenderedPrompt:
    if stage == "planner":
        return _render_system_prompt_planner(tools_available=tools_available)
    if stage == "review":
        return _render_system_prompt_review()
    return _render_system_prompt_answer()


def _build_messages(
    *,
    message: str,
    history: list[dict[str, Any]] | None,
    context: str | None,
    stage: Literal["planner", "answer", "review"] = "answer",
    tools_available: bool = False,
    system_prompt: str | None = None,
) -> list[dict[str, Any]]:
    if system_prompt is None:
        system_prompt = _render_stage_prompt(stage=stage, tools_available=tools_available).text
    messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
    if context:
        messages.append({"role": "system", "content": context})

    if stage == "review":
        history = None

    history_items: list[dict[str, Any]] = []
    limit = _history_limit()
    if history:
        for item in history[-limit:] if limit else []:
            if not isinstance(item, dict):
                continue
            role = str(item.get("role", "user"))
            content = str(item.get("content", "") or "")
            content_stripped = content.strip()
            extra: dict[str, Any] = {}
            if "tool_call_id" in item:
                extra["tool_call_id"] = item.get("tool_call_id")
            if "tool_calls" in item:
                extra["tool_calls"] = item.get("tool_calls")

            if not content_stripped and not extra:
                continue

            history_items.append({"role": role, "content": content, **extra})

        # The frontend includes the current user message in history *and* as
        # the `message` field; avoid duplicating it in the prompt.
        if (
            history_items
            and history_items[-1].get("role") == "user"
            and str(history_items[-1].get("content", "") or "").strip() == message.strip()
        ):
            history_items.pop()

        messages.extend(history_items)

    messages.append({"role": "user", "content": message})
    return messages


def _generate_ollama_reply(*, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None, observability_context: dict[str, Any] | None = None) -> AssistantReply:
    url = f"{_ollama_url()}/api/chat"
    model = _ollama_model()
    timeout = _ollama_timeout_seconds()

    allowed_roles = {"system", "user", "assistant"}
    normalized_messages: list[dict[str, Any]] = []
    for item in messages:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "user")
        if role not in allowed_roles:
            role = "system"
        normalized_messages.append({"role": role, "content": str(item.get("content", "") or "")})

    payload = {"model": model, "messages": normalized_messages, "stream": False}
    temperature = _assistant_temperature()
    if temperature is not None:
        payload["options"] = {"temperature": temperature}
    started = time.monotonic()
    try:
        response = requests.post(url, json=payload, timeout=timeout)
        response.raise_for_status()
        data = response.json()
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        logger.warning("Ollama request failed (%s): %s", url, error)
        meta = {"url": url, "error": repr(exc)}
        record_inference_usage_event(
            provider="ollama",
            model=model,
            meta=meta,
            ok=False,
            error=error,
            observability_context=observability_context,
        )
        return AssistantReply(
            content=f"Assistant error: {error}",
            provider="ollama",
            model=model,
            meta=meta,
            ok=False,
            error=error,
        )
    elapsed_ms = int((time.monotonic() - started) * 1000)

    content = ""
    tool_calls = None
    if isinstance(data, dict):
        message_obj = data.get("message") or {}
        if isinstance(message_obj, dict):
            content = str(message_obj.get("content") or "")
    if not content and not tool_calls:
        content = json.dumps(data)[:4000]

    meta = {"url": url, "elapsed_ms": elapsed_ms}
    record_inference_usage_event(
        provider="ollama",
        model=model,
        meta=meta,
        ok=True,
        observability_context=observability_context,
    )
    return AssistantReply(
        content=content,
        provider="ollama",
        model=model,
        meta=meta,
        tool_calls=None,
    )


def _normalize_vllm_structured_outputs(extra_body: dict[str, Any]) -> dict[str, Any]:
    cleaned = dict(extra_body)
    structured_outputs_raw = cleaned.get("structured_outputs")
    structured_outputs: dict[str, Any] = (
        dict(structured_outputs_raw) if isinstance(structured_outputs_raw, dict) else {}
    )

    if "guided_json" in cleaned and "json" not in structured_outputs and "response_format" not in cleaned:
        structured_outputs["json"] = cleaned.pop("guided_json")
    else:
        cleaned.pop("guided_json", None)

    if "guided_choice" in cleaned and "choice" not in structured_outputs:
        structured_outputs["choice"] = cleaned.pop("guided_choice")
    else:
        cleaned.pop("guided_choice", None)

    if structured_outputs:
        cleaned["structured_outputs"] = structured_outputs
    elif "structured_outputs" in cleaned and not isinstance(cleaned.get("structured_outputs"), dict):
        cleaned.pop("structured_outputs", None)

    return cleaned


def _sanitize_vllm_extra_body(extra_body: dict[str, Any]) -> dict[str, Any]:
    forbidden = {"model", "messages", "stream", "tools", "tool_choice"}
    cleaned: dict[str, Any] = {}
    for key, value in extra_body.items():
        if not isinstance(key, str):
            continue
        if key in forbidden:
            continue
        cleaned[key] = value
    return _normalize_vllm_structured_outputs(cleaned)


def _coerce_salvaged_tool_arguments(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return {}
        try:
            parsed = json.loads(raw)
        except Exception:
            return None
        if isinstance(parsed, dict):
            return parsed
    return None


def _normalize_salvaged_tool_call_object(raw: Any) -> tuple[dict[str, Any] | None, str | None]:
    if not isinstance(raw, dict):
        return None, None

    if "tool_calls" in raw and isinstance(raw.get("tool_calls"), list):
        tool_calls: list[dict[str, Any]] = []
        child_shape: str | None = None
        for item in raw.get("tool_calls") or []:
            normalized, normalized_shape = _normalize_salvaged_tool_call_object(item)
            if normalized is None:
                return None, None
            tool_calls.append(normalized)
            child_shape = child_shape or normalized_shape
        if tool_calls:
            return {"tool_calls": tool_calls}, f"tool_calls_wrapper:{child_shape or 'unknown'}"
        return None, None

    if raw.get("type") == "function" and isinstance(raw.get("function"), dict):
        func_obj = raw.get("function") or {}
        name = str(func_obj.get("name") or "").strip()
        arguments = _coerce_salvaged_tool_arguments(func_obj.get("arguments"))
        if name and arguments is not None:
            return (
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(arguments, ensure_ascii=False, separators=(",", ":")),
                    },
                },
                "function_wrapper",
            )
        return None, None

    name = str(raw.get("name") or "").strip()
    if not name:
        return None, None
    if "arguments" in raw:
        arguments = _coerce_salvaged_tool_arguments(raw.get("arguments"))
        shape = "name_arguments"
    elif "parameters" in raw:
        arguments = _coerce_salvaged_tool_arguments(raw.get("parameters"))
        shape = "name_parameters"
    else:
        return None, None
    if arguments is None:
        return None, None
    return (
        {
            "type": "function",
            "function": {
                "name": name,
                "arguments": json.dumps(arguments, ensure_ascii=False, separators=(",", ":")),
            },
        },
        shape,
    )


def _salvage_vllm_tool_calls_from_content(content: str | None) -> tuple[list[dict[str, Any]] | None, str | None]:
    raw = str(content or "").strip()
    if not raw or raw[0] not in {"{", "["}:
        return None, None
    try:
        parsed = json.loads(raw)
    except Exception:
        return None, None

    if isinstance(parsed, list):
        tool_calls: list[dict[str, Any]] = []
        child_shape: str | None = None
        for item in parsed:
            normalized, normalized_shape = _normalize_salvaged_tool_call_object(item)
            if normalized is None or "tool_calls" in normalized:
                return None, None
            tool_calls.append(normalized)
            child_shape = child_shape or normalized_shape
        if tool_calls:
            return tool_calls, f"list:{child_shape or 'unknown'}"
        return None, None

    normalized, shape = _normalize_salvaged_tool_call_object(parsed)
    if normalized is None:
        return None, None
    if "tool_calls" in normalized:
        wrapped = normalized.get("tool_calls")
        if isinstance(wrapped, list) and wrapped:
            return wrapped, shape
        return None, None
    return [normalized], shape


def _generate_vllm_reply(
    *,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    tool_choice: str | dict[str, Any] | None = None,
    extra_body: dict[str, Any] | None = None,
    observability_context: dict[str, Any] | None = None,
) -> AssistantReply:
    base_url = _vllm_url()
    url = f"{base_url}/v1/chat/completions"
    timeout = _vllm_timeout_seconds()
    headers = _vllm_headers()

    try:
        model = _resolve_vllm_model(base_url=base_url, headers=headers, timeout=timeout)
    except Exception as exc:
        model = _vllm_model()
        error = f"{type(exc).__name__}: {exc}"
        logger.warning("vLLM model resolution failed (%s): %s", base_url, error)
        meta = {"url": base_url, "error": repr(exc)}
        record_inference_usage_event(
            provider="vllm",
            model=model,
            meta=meta,
            ok=False,
            error=error,
            observability_context=observability_context,
        )
        return AssistantReply(
            content=f"Assistant error: {error}",
            provider="vllm",
            model=model,
            meta=meta,
            tool_calls=None,
            ok=False,
            error=error,
        )
    model = model or "unknown"

    payload_base: dict[str, Any] = {"model": model, "messages": messages, "stream": False}
    temperature = _assistant_temperature()
    if temperature is not None:
        payload_base["temperature"] = temperature
    if tools is not None:
        payload_base["tools"] = tools
        payload_base["tool_choice"] = tool_choice if tool_choice is not None else "auto"
    cleaned_extra_body: dict[str, Any] | None = None
    payload = dict(payload_base)
    if extra_body:
        cleaned_extra_body = _sanitize_vllm_extra_body(extra_body)
        payload.update(cleaned_extra_body)
    started = time.monotonic()
    fallback: dict[str, Any] = {}
    rejections: list[dict[str, Any]] = []
    try:
        response = None
        data = None
        candidates: list[tuple[str, dict[str, Any], dict[str, Any]]] = [("full", payload, {})]
        if cleaned_extra_body:
            candidates.append(("drop_extra_fields", payload_base, {"dropped_extra_fields": True}))
        if tools is not None:
            payload_no_tools = dict(payload_base)
            payload_no_tools.pop("tools", None)
            payload_no_tools.pop("tool_choice", None)
            candidates.append(
                (
                    "drop_tools",
                    payload_no_tools,
                    {"dropped_tools": True, "dropped_extra_fields": bool(cleaned_extra_body)},
                )
            )

        last_exc: Exception | None = None
        for label, candidate, candidate_fallback in candidates:
            try:
                response = requests.post(url, json=candidate, headers=headers, timeout=timeout)
                response.raise_for_status()
                data = response.json()
                fallback = candidate_fallback
                last_exc = None
                break
            except requests.exceptions.HTTPError as exc:
                status_code = getattr(getattr(exc, "response", None), "status_code", None)
                if status_code in {400, 422}:
                    response_text = ""
                    response_obj = getattr(exc, "response", None)
                    if response_obj is not None:
                        try:
                            response_text = str(getattr(response_obj, "text", "") or "")
                        except Exception:
                            response_text = ""
                    response_text = _truncate_text(response_text)
                    rejections.append(
                        {
                            "attempt": label,
                            "status_code": status_code,
                            "response_text": response_text or None,
                        }
                    )
                    last_exc = exc
                    continue
                raise

        if last_exc is not None or data is None:
            raise last_exc or RuntimeError("No response data from vLLM.")
    except requests.exceptions.HTTPError as exc:
        response_obj = getattr(exc, "response", None)
        status_code = getattr(response_obj, "status_code", None)
        response_text = ""
        if response_obj is not None:
            try:
                response_text = str(getattr(response_obj, "text", "") or "")
            except Exception:
                response_text = ""

        error = f"{type(exc).__name__}: {exc}"
        meta: dict[str, Any] = {"url": url, "error": repr(exc)}
        if status_code is not None:
            meta["status_code"] = status_code
        response_text = _truncate_text(response_text)
        if response_text:
            meta["response_text"] = response_text
        if rejections:
            meta["attempts"] = rejections[-3:]
        logger.warning(
            "vLLM request failed (%s, status=%s): %s",
            url,
            meta.get("status_code"),
            meta.get("response_text") or error,
        )
        record_inference_usage_event(
            provider="vllm",
            model=model,
            meta=meta,
            ok=False,
            error=error,
            observability_context=observability_context,
        )
        return AssistantReply(
            content=f"Assistant error: {error}",
            provider="vllm",
            model=model,
            meta=meta,
            tool_calls=None,
            ok=False,
            error=error,
        )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        logger.warning("vLLM request failed (%s): %s", url, error)
        meta = {"url": url, "error": repr(exc)}
        record_inference_usage_event(
            provider="vllm",
            model=model,
            meta=meta,
            ok=False,
            error=error,
            observability_context=observability_context,
        )
        return AssistantReply(
            content=f"Assistant error: {error}",
            provider="vllm",
            model=model,
            meta=meta,
            tool_calls=None,
            ok=False,
            error=error,
        )
    elapsed_ms = int((time.monotonic() - started) * 1000)

    content = ""
    usage: dict[str, Any] | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_parser_fallback_used = False
    tool_parser_fallback_shape: str | None = None
    if isinstance(data, dict):
        usage_obj = data.get("usage")
        if isinstance(usage_obj, dict):
            usage = usage_obj

        choices = data.get("choices")
        if isinstance(choices, list) and choices:
            choice0 = choices[0]
            if isinstance(choice0, dict):
                message_obj = choice0.get("message")
                if isinstance(message_obj, dict):
                    content = str(message_obj.get("content") or "")
                    tool_calls_obj = message_obj.get("tool_calls")
                    if isinstance(tool_calls_obj, list) and tool_calls_obj:
                        tool_calls = [tc for tc in tool_calls_obj if isinstance(tc, dict)]
                if not content and "text" in choice0:
                    content = str(choice0.get("text") or "")
    if tools is not None and not tool_calls:
        salvaged_tool_calls, salvage_shape = _salvage_vllm_tool_calls_from_content(content)
        if salvaged_tool_calls:
            tool_calls = salvaged_tool_calls
            tool_parser_fallback_used = True
            tool_parser_fallback_shape = salvage_shape
            content = ""
    if not content and not tool_calls:
        content = json.dumps(data)[:4000]

    meta: dict[str, Any] = {"url": url, "elapsed_ms": elapsed_ms}
    if usage is not None:
        meta["usage"] = usage
    if tools is not None:
        meta["tool_call_dialect"] = "parser_native_name_arguments"
    if tool_parser_fallback_used:
        meta["tool_parser_fallback_used"] = True
        meta["tool_parser_fallback_shape"] = tool_parser_fallback_shape
    if fallback:
        meta["fallback"] = dict(fallback)
        if rejections:
            meta["fallback"]["rejected"] = rejections[-3:]
        logger.info("vLLM request succeeded with fallback %s", meta["fallback"])

    record_inference_usage_event(
        provider="vllm",
        model=model,
        meta=meta,
        ok=True,
        observability_context=observability_context,
    )
    return AssistantReply(
        content=content,
        provider="vllm",
        model=model,
        meta=meta,
        tool_calls=tool_calls,
    )
