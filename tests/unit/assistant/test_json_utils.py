from __future__ import annotations

from ispec.assistant.json_utils import parse_json_object


def test_parse_json_object_accepts_plain_object() -> None:
    assert parse_json_object('{"key":"value"}') == {"key": "value"}


def test_parse_json_object_extracts_object_from_wrapped_text() -> None:
    assert parse_json_object('Result: {"intent":"save_now","confidence":0.9}') == {
        "intent": "save_now",
        "confidence": 0.9,
    }


def test_parse_json_object_rejects_non_object_json() -> None:
    assert parse_json_object('["not","an","object"]') is None


def test_parse_json_object_returns_none_for_empty_or_invalid() -> None:
    assert parse_json_object("") is None
    assert parse_json_object("no-json-here") is None
