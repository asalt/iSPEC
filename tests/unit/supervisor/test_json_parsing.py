from __future__ import annotations

from ispec.supervisor.loop import _parse_json_object


def test_parse_json_object_parses_plain_object():
    assert _parse_json_object('{"a": 1}') == {"a": 1}


def test_parse_json_object_parses_object_wrapped_in_json_string():
    assert _parse_json_object('"{\\"a\\": 1}"') == {"a": 1}


def test_parse_json_object_parses_object_inside_code_fence():
    assert _parse_json_object("```json\n{\"a\": 1}\n```") == {"a": 1}


def test_parse_json_object_repairs_missing_closing_brace():
    assert _parse_json_object('{\"a\": {\"b\": 1}') == {"a": {"b": 1}}
