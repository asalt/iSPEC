from __future__ import annotations

import os
import shutil
import subprocess
import time
import uuid
from pathlib import Path

import pytest

from ispec.assistant.support_policies import select_support_tool_policy
from ispec.assistant.tools import _tmux_capture_snapshot, _tmux_find_allowed_pane, _tmux_send_text


pytestmark = pytest.mark.skipif(shutil.which("tmux") is None, reason="tmux is not installed")


def _run_tmux(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["tmux", *args],
        check=True,
        capture_output=True,
        text=True,
    )


def _create_test_pane(*, session_name: str, pane_title: str) -> str:
    shell = os.environ.get("SHELL") or "/bin/bash"
    _run_tmux("new-session", "-d", "-s", session_name, "-n", "work", shell)
    target_proc = _run_tmux("list-panes", "-t", f"{session_name}:work", "-F", "#S:#W.#P")
    target = str(target_proc.stdout or "").strip().splitlines()[0]
    _run_tmux("select-pane", "-t", target, "-T", pane_title)
    time.sleep(0.4)
    return target


def _seed_pane_output(*, target: str, marker: str) -> None:
    _tmux_send_text(target=target, text=f"echo {marker}")
    time.sleep(0.4)


def _configure_allowlist(monkeypatch: pytest.MonkeyPatch, *, allowlist_path: Path, target: str) -> None:
    allowlist_path.write_text(f"{target}\n", encoding="utf-8")
    monkeypatch.setenv("ISPEC_ASSISTANT_TMUX_TOOLS_ENABLED", "1")
    monkeypatch.setenv("ISPEC_ASSISTANT_TMUX_TARGET_ALLOWLIST_PATH", str(allowlist_path))
    monkeypatch.delenv("ISPEC_ASSISTANT_TMUX_TARGET_ALLOWLIST", raising=False)


def test_tmux_send_text_round_trip(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    session_name = f"ispec-tmux-test-{uuid.uuid4().hex[:8]}"
    allowlist_path = tmp_path / "tmux-pane-allowlist.txt"
    target = _create_test_pane(session_name=session_name, pane_title="roundtrip send pane")

    try:
        _configure_allowlist(monkeypatch, allowlist_path=allowlist_path, target=target)

        pane = _tmux_find_allowed_pane(target)
        assert pane is not None

        marker = "hello_test_123"
        _tmux_send_text(target=target, text=f"echo {marker}")
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


def test_tmux_policy_can_find_examine_and_push_text_for_described_pane(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    session_name = f"ispec-tmux-find-{uuid.uuid4().hex[:8]}"
    allowlist_path = tmp_path / "tmux-pane-allowlist.txt"
    descriptor = f"spectra-bench-{uuid.uuid4().hex[:6]}"
    initial_marker = f"{descriptor}-ready"
    roundtrip_marker = f"{descriptor}-roundtrip"
    target = _create_test_pane(session_name=session_name, pane_title=f"{descriptor} pane")

    try:
        _configure_allowlist(monkeypatch, allowlist_path=allowlist_path, target=target)
        _seed_pane_output(target=target, marker=initial_marker)

        selection = select_support_tool_policy(
            message=f"what is going on in the {descriptor} tmux pane?",
        )
        assert selection is not None
        assert selection.rule_name == "tmux_capture_unique_pane"
        assert selection.tool_name == "assistant_capture_tmux_pane"

        resolved_target = str(selection.args["target"])
        pane = _tmux_find_allowed_pane(resolved_target)
        assert pane is not None

        snapshot = _tmux_capture_snapshot(
            pane=pane,
            lines=int(selection.args.get("lines", 40) or 40),
            include_history=True,
            history_lines=200,
        )
        assert descriptor in str(snapshot.get("pane_title") or "")
        assert initial_marker in str(snapshot.get("content") or "")

        _tmux_send_text(target=resolved_target, text=f"echo {roundtrip_marker}")
        time.sleep(0.4)

        snapshot_after = _tmux_capture_snapshot(
            pane=pane,
            lines=80,
            include_history=True,
            history_lines=200,
        )
        assert roundtrip_marker in str(snapshot_after.get("content") or "")
    finally:
        subprocess.run(
            ["tmux", "kill-session", "-t", session_name],
            check=False,
            capture_output=True,
            text=True,
        )
