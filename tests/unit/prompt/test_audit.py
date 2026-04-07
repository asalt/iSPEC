from __future__ import annotations

import ast
from textwrap import dedent

from ispec.prompt.audit import _literal_string_value, audit_inline_prompt_literals


def test_literal_string_value_reconstructs_prompt_like_text():
    tree = ast.parse(
        dedent(
            '''
            PROMPT = """You are the assistant.\nResponse format:\n- Output only:\n  FINAL:\n  <answer>\nTool calling protocol:\n- Do not call tools.\n"""
            '''
        ).strip()
    )
    node = next(
        item for item in ast.walk(tree) if isinstance(item, ast.Constant) and isinstance(item.value, str)
    )

    text = _literal_string_value(node)
    assert text is not None
    assert "You are the assistant." in text
    assert "Response format:" in text
    assert "Tool calling protocol:" in text


def test_audit_inline_prompt_literals_ignores_docstrings_and_small_glue(tmp_path):
    source_root = tmp_path / "src"
    source_root.mkdir()
    (source_root / "module.py").write_text(
        dedent(
            '''
            """You are a module docstring.\nResponse format:\n- Output only:\n  FINAL:\n  <answer>\nTool calling protocol:\n- Do not call tools.\n"""

            def helper():
                return "Output only: FINAL"
            '''
        ),
        encoding="utf-8",
    )

    findings = audit_inline_prompt_literals(source_root=source_root, min_chars=80, min_newlines=4)
    assert findings == []
