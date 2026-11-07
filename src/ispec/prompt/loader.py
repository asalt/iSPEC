from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from string import Template
from typing import Any

from .bindings import binding_meta_for_callable
from .connect import lookup_prompt_version
from .models import PromptBindingMeta, PromptComponent, PromptSource, RenderedPrompt
from .parser import parse_prompt_file


_PROMPT_SOURCE_CACHE: dict[str, tuple[int, PromptSource]] = {}
PromptValue = (
    str | int | float | bool | None | list[Any] | dict[str, Any] | RenderedPrompt
)


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


def _render_value(value: PromptValue) -> str:
    if isinstance(value, RenderedPrompt):
        return value.text
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
    values: Mapping[str, PromptValue] | None = None,
    binding: PromptBindingMeta | None = None,
    variant_id: str | None = None,
) -> RenderedPrompt:
    source = load_prompt_source(family)
    template = Template(source.body)
    render_values = {
        str(key): _render_value(value) for key, value in (values or {}).items()
    }
    text = template.substitute(render_values) if render_values else source.body
    components: list[PromptComponent] = []
    value_slots: list[str] = []
    occurrences: dict[str, int] = {}
    for match in template.pattern.finditer(source.body):
        slot = match.group("named") or match.group("braced")
        if slot is None or slot not in (values or {}):
            continue
        value = values[slot]
        if isinstance(value, RenderedPrompt):
            occurrence = occurrences.get(slot, 0)
            components.append(
                PromptComponent(slot=slot, occurrence=occurrence, prompt=value)
            )
            occurrences[slot] = occurrence + 1
        elif slot not in value_slots:
            value_slots.append(slot)
    version = lookup_prompt_version(
        family=source.family, body_sha256=source.body_sha256
    )
    return RenderedPrompt(
        text=text,
        source=source,
        binding=binding,
        version=version,
        components=tuple(components),
        value_slots=tuple(value_slots),
        variant_id=variant_id,
    )


def load_bound_prompt(
    binding_callable: Any, *, values: Mapping[str, PromptValue] | None = None
) -> RenderedPrompt:
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
