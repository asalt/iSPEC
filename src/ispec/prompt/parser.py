from __future__ import annotations

import hashlib
import tomllib
from pathlib import Path

from .models import PromptSource


_ALLOWED_FRONTMATTER_KEYS = frozenset({"title", "notes"})


def _strip_bom(text: str) -> str:
    return text[1:] if text.startswith("\ufeff") else text


def _parse_frontmatter(raw_text: str) -> tuple[dict[str, str], str]:
    text = _strip_bom(raw_text)
    lines = text.splitlines(keepends=True)
    if not lines:
        return {}, ""
    if lines[0].strip() != "+++":
        return {}, text

    closing_index: int | None = None
    for index, line in enumerate(lines[1:], start=1):
        if line.strip() == "+++":
            closing_index = index
            break
    if closing_index is None:
        raise ValueError("Prompt frontmatter is missing a closing +++ fence.")

    frontmatter_text = "".join(lines[1:closing_index])
    parsed = tomllib.loads(frontmatter_text) if frontmatter_text.strip() else {}
    if not isinstance(parsed, dict):
        raise ValueError("Prompt frontmatter must parse to a TOML table.")

    unknown_keys = sorted(set(parsed) - _ALLOWED_FRONTMATTER_KEYS)
    if unknown_keys:
        raise ValueError(f"Unknown prompt frontmatter key(s): {', '.join(unknown_keys)}")

    metadata: dict[str, str] = {}
    for key in _ALLOWED_FRONTMATTER_KEYS:
        value = parsed.get(key)
        if value is None:
            continue
        if not isinstance(value, str):
            raise ValueError(f"Prompt frontmatter field {key!r} must be a string.")
        cleaned = value.strip()
        if cleaned:
            metadata[key] = cleaned

    body = "".join(lines[closing_index + 1 :])
    return metadata, body


def parse_prompt_file(path: str | Path) -> PromptSource:
    source_path = Path(path).expanduser().resolve()
    raw_text = source_path.read_text(encoding="utf-8")
    metadata, body = _parse_frontmatter(raw_text)
    family = source_path.name[:-3] if source_path.name.endswith(".md") else source_path.stem
    if not family.strip():
        raise ValueError(f"Prompt file {source_path} does not produce a valid family name.")
    body_sha256 = hashlib.sha256(body.encode("utf-8")).hexdigest()
    return PromptSource(
        family=family.strip(),
        source_path=str(source_path),
        title=metadata.get("title"),
        notes=metadata.get("notes"),
        body=body,
        body_sha256=body_sha256,
    )
