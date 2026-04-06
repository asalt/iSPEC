from __future__ import annotations

import json
from pathlib import Path
from string import Template
from typing import Any

from .bindings import binding_meta_for_callable
from .connect import lookup_prompt_version
from .models import PromptBindingMeta, PromptSource, RenderedPrompt
from .parser import parse_prompt_file


_PROMPT_SOURCE_CACHE: dict[str, tuple[int, PromptSource]] = {}


def resolve_prompt_root() -> Path:
    return Path(__file__).resolve().parents[1] / "prompts"


def prompt_source_path_for_family(family: str) -> Path:
    cleaned = str(family or "").strip()
    if not cleaned:
        raise ValueError("Prompt family is required.")
    return resolve_prompt_root() / f"{cleaned}.md"


def load_prompt_source(family: str) -> PromptSource:
    path = prompt_source_path_for_family(family)
    stat = path.stat()
    cache_key = str(path)
    cached = _PROMPT_SOURCE_CACHE.get(cache_key)
    if cached is not None and cached[0] == stat.st_mtime_ns:
        return cached[1]
    source = parse_prompt_file(path)
    _PROMPT_SOURCE_CACHE[cache_key] = (stat.st_mtime_ns, source)
    return source


def _render_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def render_prompt(
    family: str,
    *,
    values: dict[str, Any] | None = None,
    binding: PromptBindingMeta | None = None,
) -> RenderedPrompt:
    source = load_prompt_source(family)
    template = Template(source.body)
    render_values = {str(key): _render_value(value) for key, value in (values or {}).items()}
    text = template.substitute(render_values) if render_values else source.body
    version = lookup_prompt_version(family=source.family, body_sha256=source.body_sha256)
    return RenderedPrompt(text=text, source=source, binding=binding, version=version)


def load_bound_prompt(binding_callable: Any, *, values: dict[str, Any] | None = None) -> RenderedPrompt:
    binding = binding_meta_for_callable(binding_callable)
    return render_prompt(binding.family, values=values, binding=binding)


def prompt_observability_context(
    prompt: RenderedPrompt,
    *,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = dict(extra or {})
    payload.update(prompt.observability_fields())
    return payload
