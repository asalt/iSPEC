"""Bounded template provenance, not a full request transcript or execution graph."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .models import RenderedPrompt

MAX_COMPONENTS = 64


def prompt_composition(prompt: RenderedPrompt) -> dict[str, Any]:
    nodes = []
    pending = [(prompt, None, None, 0)]
    while pending and len(nodes) < MAX_COMPONENTS:
        current, parent, slot, occurrence = pending.pop()
        index = len(nodes)
        nodes.append(
            {
                "id": index,
                "parent": parent,
                "slot": slot,
                "occurrence": occurrence,
                "family": current.source.family,
                "source_sha256": current.source.body_sha256,
                "version_num": current.version.version_num,
                "variant_id": current.variant_id,
                "binding": (
                    current.binding.binding_ref if current.binding is not None else None
                ),
                "value_slots": list(current.value_slots),
            }
        )
        pending.extend(
            (part.prompt, index, part.slot, part.occurrence)
            for part in reversed(current.components)
        )
    return {
        "schema_version": 1,
        "coverage": "template_components_only",
        "nodes": nodes,
        "truncated": bool(pending),
    }


def clean_prompt_composition(value: Any) -> dict[str, Any] | None:
    """Whitelist metadata at log/export boundaries; never serialize prompt values."""
    if (
        not isinstance(value, dict)
        or type(value.get("schema_version")) is not int
        or value["schema_version"] != 1
    ):
        return None
    if value.get("coverage") != "template_components_only":
        return None
    rows = value.get("nodes")
    if not isinstance(rows, list) or not rows:
        return None
    nodes = []
    for index, row in enumerate(rows[:MAX_COMPONENTS]):
        if (
            not isinstance(row, dict)
            or type(row.get("id")) is not int
            or row["id"] != index
        ):
            return None
        parent, slot, occurrence = (
            row.get("parent"),
            row.get("slot"),
            row.get("occurrence"),
        )
        if index == 0:
            if parent is not None or slot is not None or occurrence != 0:
                return None
        elif type(parent) is not int or not 0 <= parent < index or not _text(slot, 128):
            return None
        if type(occurrence) is not int or occurrence < 0:
            return None
        family, digest = row.get("family"), row.get("source_sha256")
        if not _text(family, 256) or not isinstance(digest, str) or len(digest) != 64:
            return None
        if any(char not in "0123456789abcdef" for char in digest):
            return None
        version, variant = row.get("version_num"), row.get("variant_id")
        if version is not None and (type(version) is not int or version < 1):
            return None
        if variant is not None and not _text(variant, 128):
            return None
        binding = row.get("binding")
        if binding is not None and not _text(binding, 512):
            return None
        slots = row.get("value_slots", [])
        if (
            not isinstance(slots, list)
            or len(slots) > 64
            or not all(_text(s, 128) for s in slots)
        ):
            return None
        nodes.append(
            {
                "id": index,
                "parent": parent,
                "slot": slot,
                "occurrence": occurrence,
                "family": family,
                "source_sha256": digest,
                "version_num": version,
                "variant_id": variant,
                "binding": binding,
                "value_slots": slots,
            }
        )
    return {
        "schema_version": 1,
        "coverage": "template_components_only",
        "nodes": nodes,
        "truncated": value.get("truncated") is True or len(rows) > MAX_COMPONENTS,
    }


def _text(value: Any, limit: int) -> bool:
    return isinstance(value, str) and 0 < len(value) <= limit
