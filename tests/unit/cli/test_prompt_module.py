from __future__ import annotations

import argparse
import json
import types

from ispec.cli import prompt as prompt_cli


def test_register_subcommands_parses_audit_inline():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="subcommand", required=True)
    prompt_cli.register_subcommands(subparsers)
    args = parser.parse_args(["audit-inline", "--check", "--min-chars", "120"])
    assert args.subcommand == "audit-inline"
    assert args.check is True
    assert args.min_chars == 120


def test_dispatch_audit_inline_prints_json(monkeypatch, capsys):
    finding = types.SimpleNamespace(as_dict=lambda: {"source_file": "/tmp/x.py", "line": 3})
    monkeypatch.setattr(prompt_cli, "audit_inline_prompt_literals", lambda **_: [finding])
    args = types.SimpleNamespace(
        subcommand="audit-inline",
        source_root="",
        min_chars=160,
        min_newlines=4,
        check=False,
    )

    prompt_cli.dispatch(args)
    payload = json.loads(capsys.readouterr().out)
    assert payload["count"] == 1
    assert payload["findings"][0]["source_file"] == "/tmp/x.py"
