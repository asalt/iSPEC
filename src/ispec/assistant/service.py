from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import requests

from ispec.assistant.tools import TOOL_CALL_PREFIX, TOOL_RESULT_PREFIX, tool_prompt
from ispec.logging import get_logger


logger = get_logger(__name__)

ResponseFormat = Literal["single", "compare"]

@dataclass(frozen=True)
class AssistantReply:
    content: str
    provider: str
    model: str | None = None
    meta: dict[str, Any] | None = None
    tool_calls: list[dict[str, Any]] | None = None
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


def _default_prompt_base() -> str:
    identity = (os.getenv("ISPEC_ASSISTANT_NAME") or "iSPEC").strip() or "iSPEC"
    return (
        f"You are {identity}, the built-in support assistant for the iSPEC web app.\n"
        "Your job is to help staff use iSPEC to track projects, people, experiments, and runs.\n"
        "\n"
        "Behavior:\n"
        "- Be concise, practical, and action-oriented.\n"
        "- Ask a single clarifying question when needed.\n"
        "- Never invent database values, IDs, or outcomes.\n"
        "- If you reference a record, include its id and title when available.\n"
        "- CONTEXT is a partial snapshot; do not assume lists are exhaustive or infer global counts from them.\n"
        "- When tool calling is available, use tools for database lookups; do not claim you can't access iSPEC data.\n"
        "- If users share product feedback or feature requests, thank them and ask for specifics (page/route, what they expected).\n"
        "- Do not reveal secrets (API keys, env vars, credentials) or internal paths.\n"
        "\n"
        "You may be provided an additional system message called CONTEXT that contains\n"
        "read-only JSON from the iSPEC database and your chat session state. Treat that\n"
        "context as authoritative.\n"
        "If CONTEXT.session.state.conversation_summary is present, it is a rolling summary\n"
        "of older turns that may be omitted from the message history.\n"
    )

def _final_template(*, response_format: ResponseFormat) -> str:
    if response_format == "compare":
        return (
            "  FINAL_A:\n"
            "  <draft answer A (concise)>\n"
            "  FINAL_B:\n"
            "  <draft answer B (alternative phrasing/structure; can be slightly more detailed)>\n"
        )
    return "  FINAL:\n  <your user-facing answer>\n"


def _system_prompt_answer(*, response_format: ResponseFormat = "single") -> str:
    prompt = (
        _default_prompt_base().rstrip()
        + "\n\n"
        + "Response format:\n"
        + "- Output only:\n"
        + _final_template(response_format=response_format)
        + "- Do not include PLAN.\n"
        + "\n"
        + "UI routes (common): /projects, /project/<id>, /people, /experiments,\n"
        + "/experiment/<id>, /experiment-runs, /experiment-run/<id>.\n"
        + "Project status values: inquiry, consultation, waiting, processing, analysis,\n"
        + "summary, closed, hibernate.\n"
    )

    file_override = _read_prompt_from_env("ISPEC_ASSISTANT_SYSTEM_PROMPT_PATH")
    if file_override:
        prompt = file_override.strip()

    override = (os.getenv("ISPEC_ASSISTANT_SYSTEM_PROMPT") or "").strip()
    if override:
        prompt = override.strip()

    extras = _prompt_extras()
    if extras:
        prompt = prompt.rstrip() + "\n\n" + extras

    return prompt.strip()


def _system_prompt_planner(
    *,
    tools_available: bool,
    response_format: ResponseFormat = "single",
) -> str:
    prompt = (
        _default_prompt_base().rstrip()
        + "\n\n"
        + "Tool use (optional):\n"
        + "- If you need more iSPEC DB info than CONTEXT provides, request a tool.\n"
        + "- Never invent database values, IDs, or outcomes.\n"
        + "- For global count/list questions (e.g. 'how many projects'), do not infer from CONTEXT; use count_projects.\n"
        + "- For 'latest projects' / 'recent changes', use latest_projects and latest_project_comments.\n"
        + "- For experiments in a specific project, use experiments_for_project.\n"
        + "- For collaborative project work, draft notes first; only write to project history if the user explicitly asks you to save.\n"
        + "- For code searches in the iSPEC repo (dev-only), use repo_search/repo_list_files/repo_read_file.\n"
        + "- If the user explicitly asks you to use a tool, call the appropriate tool.\n"
    )

    if tools_available:
        prompt += (
            "\n"
            "Tool calling protocol:\n"
            "- Use OpenAI-style tool_calls (tools/tool_choice are provided).\n"
            "- When calling tools, do not include PLAN/FINAL in the content.\n"
            f"- If structured tool_calls are not supported, fallback to one line starting with {TOOL_CALL_PREFIX}:\n"
            f'  {TOOL_CALL_PREFIX} {{"name":"<tool>","arguments":{{...}}}}\n'
            f"- Tool results may arrive as a {TOOL_RESULT_PREFIX} system message or a role=tool message; treat them as authoritative.\n"
        )
    else:
        prompt += (
            "\n"
            "Tool calling protocol:\n"
            f"- Request a tool by outputting exactly one line starting with {TOOL_CALL_PREFIX}:\n"
            f'  {TOOL_CALL_PREFIX} {{"name":"<tool>","arguments":{{...}}}}\n'
            f"- Tool results arrive as a {TOOL_RESULT_PREFIX} system message; treat them as authoritative.\n"
            "\n"
            f"{tool_prompt()}\n"
        )

    prompt += (
        "\n"
        "Response format:\n"
        "- If you call a tool: output only the tool call (or tool_calls), with no extra text.\n"
        "- Otherwise, output only:\n"
        + _final_template(response_format=response_format)
    )

    extras = _prompt_extras()
    if extras:
        prompt = prompt.rstrip() + "\n\n" + extras

    return prompt.strip()


def _system_prompt_review() -> str:
    prompt = (
        _default_prompt_base().rstrip()
        + "\n\n"
        + "You are in review mode.\n"
        + "- Review the draft answer for correctness (grounded in CONTEXT / tool results), clarity, and iSPEC tone.\n"
        + "- Do not call tools.\n"
        + "- If the draft is already good, repeat it verbatim.\n"
        + "- Otherwise, rewrite it.\n"
        + "\n"
        + "Response format:\n"
        + "- Output only:\n"
        + "  FINAL:\n"
        + "  <answer>\n"
    )

    extras = _prompt_extras()
    if extras:
        prompt = prompt.rstrip() + "\n\n" + extras

    return prompt.strip()


def _system_prompt_review_decider() -> str:
    prompt = (
        _default_prompt_base().rstrip()
        + "\n\n"
        + "You are in review decision mode.\n"
        + "- Decide if the draft answer needs changes.\n"
        + "- Do not call tools.\n"
        + "- Output exactly one token: KEEP or REWRITE.\n"
    )

    extras = _prompt_extras()
    if extras:
        prompt = prompt.rstrip() + "\n\n" + extras

    return prompt.strip()


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
) -> AssistantReply:
    provider = (os.getenv("ISPEC_ASSISTANT_PROVIDER") or "stub").strip().lower()
    if messages is None:
        if message is None:
            raise ValueError("message is required when messages is not provided")
        messages = _build_messages(
            message=message,
            history=history,
            context=context,
            stage=stage,
            tools_available=bool(tools),
        )
    if provider == "ollama":
        return _generate_ollama_reply(messages=messages, tools=tools)
    if provider == "vllm":
        return _generate_vllm_reply(
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            extra_body=vllm_extra_body,
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


def _build_messages(
    *,
    message: str,
    history: list[dict[str, Any]] | None,
    context: str | None,
    stage: Literal["planner", "answer", "review"] = "answer",
    tools_available: bool = False,
) -> list[dict[str, Any]]:
    if stage == "planner":
        system_prompt = _system_prompt_planner(tools_available=tools_available)
    elif stage == "review":
        system_prompt = _system_prompt_review()
    else:
        system_prompt = _system_prompt_answer()

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


def _generate_ollama_reply(*, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None) -> AssistantReply:
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
        return AssistantReply(
            content=f"Assistant error: {error}",
            provider="ollama",
            model=model,
            meta={"url": url, "error": repr(exc)},
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

    return AssistantReply(
        content=content,
        provider="ollama",
        model=model,
        meta={"url": url, "elapsed_ms": elapsed_ms},
        tool_calls=None,
    )


def _sanitize_vllm_extra_body(extra_body: dict[str, Any]) -> dict[str, Any]:
    forbidden = {"model", "messages", "stream", "tools", "tool_choice"}
    cleaned: dict[str, Any] = {}
    for key, value in extra_body.items():
        if not isinstance(key, str):
            continue
        if key in forbidden:
            continue
        cleaned[key] = value
    return cleaned


def _generate_vllm_reply(
    *,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    tool_choice: str | dict[str, Any] | None = None,
    extra_body: dict[str, Any] | None = None,
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
        return AssistantReply(
            content=f"Assistant error: {error}",
            provider="vllm",
            model=model,
            meta={"url": base_url, "error": repr(exc)},
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
            candidates.append(("drop_extra_body", payload_base, {"dropped_extra_body": True}))
        if tools is not None:
            payload_no_tools = dict(payload_base)
            payload_no_tools.pop("tools", None)
            payload_no_tools.pop("tool_choice", None)
            candidates.append(
                (
                    "drop_tools",
                    payload_no_tools,
                    {"dropped_tools": True, "dropped_extra_body": bool(cleaned_extra_body)},
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
        return AssistantReply(
            content=f"Assistant error: {error}",
            provider="vllm",
            model=model,
            meta={"url": url, "error": repr(exc)},
            tool_calls=None,
            ok=False,
            error=error,
        )
    elapsed_ms = int((time.monotonic() - started) * 1000)

    content = ""
    usage: dict[str, Any] | None = None
    tool_calls: list[dict[str, Any]] | None = None
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
    if not content:
        content = json.dumps(data)[:4000]

    meta: dict[str, Any] = {"url": url, "elapsed_ms": elapsed_ms}
    if usage is not None:
        meta["usage"] = usage
    if fallback:
        meta["fallback"] = dict(fallback)
        if rejections:
            meta["fallback"]["rejected"] = rejections[-3:]
        logger.info("vLLM request succeeded with fallback %s", meta["fallback"])

    return AssistantReply(
        content=content,
        provider="vllm",
        model=model,
        meta=meta,
        tool_calls=tool_calls,
    )
