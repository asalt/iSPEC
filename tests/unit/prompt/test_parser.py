from __future__ import annotations

import hashlib

import pytest

from ispec.prompt.parser import parse_prompt_file


def test_parse_prompt_file_with_toml_frontmatter(tmp_path):
    path = tmp_path / "assistant.example.classifier.md"
    body = "Return only JSON.\n$payload\n"
    path.write_text(
        "+++\n"
        'title = "Example Classifier"\n'
        'notes = "Used for unit testing."\n'
        "+++\n"
        + body,
        encoding="utf-8",
    )

    source = parse_prompt_file(path)

    assert source.family == "assistant.example.classifier"
    assert source.title == "Example Classifier"
    assert source.notes == "Used for unit testing."
    assert source.body == body
    assert source.body_sha256 == hashlib.sha256(body.encode("utf-8")).hexdigest()


def test_parse_prompt_file_without_frontmatter_uses_full_body(tmp_path):
    path = tmp_path / "assistant.no_frontmatter.md"
    body = "Just the body.\nNo metadata here.\n"
    path.write_text(body, encoding="utf-8")

    source = parse_prompt_file(path)

    assert source.family == "assistant.no_frontmatter"
    assert source.title is None
    assert source.notes is None
    assert source.body == body


def test_parse_prompt_file_rejects_unknown_frontmatter_keys(tmp_path):
    path = tmp_path / "assistant.invalid.md"
    path.write_text(
        "+++\n"
        'title = "Example"\n'
        'family = "assistant.invalid"\n'
        "+++\n"
        "Hello\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Unknown prompt frontmatter key"):
        parse_prompt_file(path)


def test_parse_prompt_file_requires_closing_frontmatter_fence(tmp_path):
    path = tmp_path / "assistant.malformed.md"
    path.write_text(
        "+++\n"
        'title = "Broken"\n'
        "body = \"no closing fence\"\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"closing \+\+\+ fence"):
        parse_prompt_file(path)
