from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import time
import uuid

import pytest

from ispec.assistant.tools import _tmux_capture_snapshot, _tmux_find_allowed_pane, _tmux_send_text


pytestmark = pytest.mark.skipif(shutil.which("tmux") is None, reason="tmux is not installed")


def test_tmux_send_text_round_trip(monkeypatch, tmp_path):
    session_name = f"ispec-tmux-test-{uuid.uuid4().hex[:8]}"
    shell = os.environ.get("SHELL") or "/bin/bash"
    allowlist_path = tmp_path / "tmux-pane-allowlist.txt"

    subprocess.run(
        ["tmux", "new-session", "-d", "-s", session_name, "-n", "work", shell],
        check=True,
        capture_output=True,
        text=True,
    )

    try:
        target_proc = subprocess.run(
            ["tmux", "list-panes", "-t", f"{session_name}:work", "-F", "#S:#W.#P"],
            check=True,
            capture_output=True,
            text=True,
        )
        target = str(target_proc.stdout or "").strip().splitlines()[0]
        allowlist_path.write_text(f"{target}\n", encoding="utf-8")

        monkeypatch.setenv("ISPEC_ASSISTANT_TMUX_TARGET_ALLOWLIST_PATH", str(allowlist_path))
        monkeypatch.delenv("ISPEC_ASSISTANT_TMUX_TARGET_ALLOWLIST", raising=False)

        pane = _tmux_find_allowed_pane(target)
        assert pane is not None

        marker = "hello_test_123"
        _tmux_send_text(target=target, text=f"printf '{marker}\\n'")
        time.sleep(0.4)

        snapshot = _tmux_capture_snapshot(
            pane=pane,
            lines=60,
            include_history=True,
            history_lines=200,
        )
        assert marker in snapshot["content"]
    finally:
        subprocess.run(
            ["tmux", "kill-session", "-t", session_name],
            check=False,
            capture_output=True,
            text=True,
        )
