from __future__ import annotations

from pathlib import Path

from ispec.assistant.tools import (
    _tmux_allowlist_entries,
    _tmux_send_text,
    _tmux_tools_status,
)


def test_tmux_tools_status_requires_nonempty_allowlist_file(monkeypatch, tmp_path):
    allowlist_path = tmp_path / "tmux-pane-allowlist.txt"
    allowlist_path.write_text("", encoding="utf-8")

    monkeypatch.setenv("ISPEC_ASSISTANT_TMUX_TOOLS_ENABLED", "1")
    monkeypatch.setenv("ISPEC_ASSISTANT_TMUX_TARGET_ALLOWLIST_PATH", str(allowlist_path))
    monkeypatch.setattr("ispec.assistant.tools.shutil.which", lambda name: "/usr/bin/tmux")

    enabled, reason = _tmux_tools_status()
    assert enabled is False
    assert str(allowlist_path) in str(reason)


def test_tmux_allowlist_entries_merge_env_and_file(monkeypatch, tmp_path):
    allowlist_path = tmp_path / "tmux-pane-allowlist.txt"
    allowlist_path.write_text(
        "\n".join(
            [
                "# comment",
                "ispecfull:backend",
                "ispecfull:codex.1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("ISPEC_ASSISTANT_TMUX_TARGET_ALLOWLIST", "ispecfull:supervisor ispecfull:backend")
    monkeypatch.setenv("ISPEC_ASSISTANT_TMUX_TARGET_ALLOWLIST_PATH", str(allowlist_path))

    entries = _tmux_allowlist_entries()
    assert entries == [
        "ispecfull:supervisor",
        "ispecfull:backend",
        "ispecfull:codex.1",
    ]


def test_tmux_send_text_uses_literal_send_then_enter(monkeypatch):
    calls: list[tuple[str, ...]] = []

    class _Proc:
        def __init__(self):
            self.returncode = 0
            self.stderr = ""

    monkeypatch.setattr(
        "ispec.assistant.tools._tmux_find_allowed_pane",
        lambda target: {
            "target": "ispecfull:codex.1",
            "pane_id": "%12",
        }
        if target == "ispecfull:codex.1"
        else None,
    )
    monkeypatch.setattr(
        "ispec.assistant.tools._tmux_raw",
        lambda *args: calls.append(tuple(args)) or _Proc(),
    )

    result = _tmux_send_text(target="ispecfull:codex.1", text="printf 'hello_test_123\\n'")
    assert result["target"] == "ispecfull:codex.1"
    assert calls == [
        ("send-keys", "-l", "-t", "ispecfull:codex.1", "printf 'hello_test_123\\n'"),
        ("send-keys", "-t", "ispecfull:codex.1", "Enter"),
    ]
