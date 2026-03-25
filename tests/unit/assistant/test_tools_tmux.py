from __future__ import annotations

from ispec.assistant.tools import _tmux_find_allowed_pane, _tmux_is_allowed_pane, _tmux_list_candidate_panes, openai_tools_for_user, run_tool
from ispec.db.models import AuthUser, UserRole


def _internal_user() -> AuthUser:
    return AuthUser(
        username="viewer",
        password_hash="hash",
        password_salt="salt",
        password_iterations=1,
        role=UserRole.viewer,
        is_active=True,
    )


def _sample_pane(*, target: str = "ispecfull:codex.1", pane_id: str = "%12") -> dict:
    return {
        "target": target,
        "window_target": "ispecfull:codex",
        "pane_id": pane_id,
        "session": "ispecfull",
        "window_name": "codex",
        "window_index": 5,
        "pane_index": 1,
        "current_command": "codex",
        "pane_dead": False,
        "pane_active": True,
    }


def test_assistant_list_tools_shows_tmux_tools_unavailable_when_disabled(db_session, monkeypatch):
    monkeypatch.delenv("ISPEC_ASSISTANT_TMUX_TOOLS_ENABLED", raising=False)

    payload = run_tool(
        name="assistant_list_tools",
        args={"include_unavailable": True, "query": "tmux"},
        core_db=db_session,
        schedule_db=None,
        user=_internal_user(),
        api_schema=None,
        user_message="show tmux tools",
    )
    assert payload["ok"] is True
    unavailable = payload["result"].get("unavailable_tools")
    assert isinstance(unavailable, list) and unavailable
    names = {item.get("name") for item in unavailable if isinstance(item, dict)}
    assert "assistant_list_tmux_panes" in names
    assert "assistant_capture_tmux_pane" in names
    assert "assistant_compare_tmux_pane" in names

    tool_names = {
        tool["function"]["name"]
        for tool in openai_tools_for_user(_internal_user())
        if isinstance(tool, dict) and isinstance(tool.get("function"), dict)
    }
    assert "assistant_list_tmux_panes" not in tool_names


def test_assistant_list_tools_includes_tmux_tools_when_enabled(db_session, monkeypatch):
    monkeypatch.setenv("ISPEC_ASSISTANT_TMUX_TOOLS_ENABLED", "1")
    monkeypatch.setattr("ispec.assistant.tools._tmux_tools_status", lambda: (True, None))
    monkeypatch.setattr("ispec.assistant.tools._code_tool_access_status", lambda user: (True, None))

    payload = run_tool(
        name="assistant_list_tools",
        args={"query": "tmux", "limit": 20},
        core_db=db_session,
        schedule_db=None,
        user=_internal_user(),
        api_schema=None,
        user_message="show tmux tools",
    )
    assert payload["ok"] is True
    tool_names = {item["name"] for item in payload["result"]["available_tools"]}
    assert "assistant_list_tmux_panes" in tool_names
    assert "assistant_capture_tmux_pane" in tool_names
    assert "assistant_compare_tmux_pane" in tool_names


def test_assistant_list_tmux_panes_returns_allowed_items(db_session, monkeypatch):
    monkeypatch.setenv("ISPEC_ASSISTANT_TMUX_TOOLS_ENABLED", "1")
    monkeypatch.setattr("ispec.assistant.tools._tmux_tools_status", lambda: (True, None))
    monkeypatch.setattr("ispec.assistant.tools._code_tool_access_status", lambda user: (True, None))
    monkeypatch.setattr(
        "ispec.assistant.tools._tmux_list_allowed_panes",
        lambda: [
            _sample_pane(target="ispecfull:codex.1", pane_id="%12"),
            {
                **_sample_pane(target="ispecfull:supervisor.1", pane_id="%13"),
                "window_target": "ispecfull:supervisor",
                "window_name": "supervisor",
                "window_index": 2,
                "current_command": "fish",
            },
        ],
    )
    monkeypatch.setattr("ispec.assistant.tools._tmux_allowlist_entries", lambda: ["ispecfull"])
    monkeypatch.setattr("ispec.assistant.tools._tmux_default_session_name", lambda: "ispecfull")

    payload = run_tool(
        name="assistant_list_tmux_panes",
        args={"query": "codex"},
        core_db=db_session,
        schedule_db=None,
        user=_internal_user(),
        api_schema=None,
        user_message="list codex panes",
    )
    assert payload["ok"] is True
    result = payload["result"]
    assert result["count"] == 1
    assert result["items"][0]["target"] == "ispecfull:codex.1"


def test_assistant_capture_tmux_pane_returns_snapshot(db_session, monkeypatch):
    pane = _sample_pane()
    monkeypatch.setenv("ISPEC_ASSISTANT_TMUX_TOOLS_ENABLED", "1")
    monkeypatch.setattr("ispec.assistant.tools._tmux_tools_status", lambda: (True, None))
    monkeypatch.setattr("ispec.assistant.tools._code_tool_access_status", lambda user: (True, None))
    monkeypatch.setattr("ispec.assistant.tools._tmux_find_allowed_pane", lambda target: pane if target == pane["target"] else None)
    monkeypatch.setattr(
        "ispec.assistant.tools._tmux_capture_snapshot",
        lambda **kwargs: {
            **pane,
            "include_history": False,
            "history_lines": None,
            "captured_total_lines": 2,
            "visible_line_count": 2,
            "last_nonempty_line": "Codex ready",
            "content": "thinking...\nCodex ready",
        },
    )

    payload = run_tool(
        name="assistant_capture_tmux_pane",
        args={"target": "ispecfull:codex.1", "lines": 50},
        core_db=db_session,
        schedule_db=None,
        user=_internal_user(),
        api_schema=None,
        user_message="capture the codex pane",
    )
    assert payload["ok"] is True
    assert payload["result"]["target"] == "ispecfull:codex.1"
    assert payload["result"]["last_nonempty_line"] == "Codex ready"


def test_assistant_compare_tmux_pane_reports_change(db_session, monkeypatch):
    pane = _sample_pane()
    snapshots = [
        {
            **pane,
            "include_history": False,
            "history_lines": None,
            "captured_total_lines": 1,
            "visible_line_count": 1,
            "last_nonempty_line": "thinking",
            "content": "thinking",
        },
        {
            **pane,
            "include_history": False,
            "history_lines": None,
            "captured_total_lines": 1,
            "visible_line_count": 1,
            "last_nonempty_line": "ready",
            "content": "ready",
        },
    ]
    monkeypatch.setenv("ISPEC_ASSISTANT_TMUX_TOOLS_ENABLED", "1")
    monkeypatch.setattr("ispec.assistant.tools._tmux_tools_status", lambda: (True, None))
    monkeypatch.setattr("ispec.assistant.tools._code_tool_access_status", lambda user: (True, None))
    monkeypatch.setattr("ispec.assistant.tools._tmux_find_allowed_pane", lambda target: pane if target == pane["target"] else None)
    monkeypatch.setattr("ispec.assistant.tools.time_module", type("T", (), {"sleep": staticmethod(lambda seconds: None)})())
    monkeypatch.setattr("ispec.assistant.tools._tmux_capture_snapshot", lambda **kwargs: snapshots.pop(0))

    payload = run_tool(
        name="assistant_compare_tmux_pane",
        args={"target": "ispecfull:codex.1", "interval_seconds": 2},
        core_db=db_session,
        schedule_db=None,
        user=_internal_user(),
        api_schema=None,
        user_message="did the codex pane change",
    )
    assert payload["ok"] is True
    assert payload["result"]["changed"] is True
    assert payload["result"]["before"]["content"] == "thinking"
    assert payload["result"]["after"]["content"] == "ready"

def test_tmux_is_allowed_pane_accepts_session_group_allowlist(monkeypatch):
    monkeypatch.setattr("ispec.assistant.tools._tmux_allowlist_entries", lambda: ["ispec"])
    monkeypatch.setattr("ispec.assistant.tools._tmux_allowlist_path", lambda: (None, False))

    assert _tmux_is_allowed_pane(
        {
            "session": "ispec-0",
            "session_group": "ispec",
            "window_target": "ispec-0:node",
            "group_window_target": "ispec:node",
            "target": "ispec-0:node.1",
            "group_target": "ispec:node.1",
            "pane_id": "%1",
        }
    ) is True


def test_assistant_list_tmux_panes_filters_by_session_group(db_session, monkeypatch):
    monkeypatch.setenv("ISPEC_ASSISTANT_TMUX_TOOLS_ENABLED", "1")
    monkeypatch.setattr("ispec.assistant.tools._tmux_tools_status", lambda: (True, None))
    monkeypatch.setattr("ispec.assistant.tools._code_tool_access_status", lambda user: (True, None))
    monkeypatch.setattr("ispec.assistant.tools._tmux_allowlist_entries", lambda: ["ispecfull", "ispec"])
    monkeypatch.setattr(
        "ispec.assistant.tools._tmux_list_allowed_panes",
        lambda: [
            {
                "target": "ispecfull:backend.1",
                "window_target": "ispecfull:backend",
                "group_window_target": None,
                "pane_id": "%2",
                "session": "ispecfull",
                "session_group": None,
                "window_name": "backend",
                "window_index": 1,
                "pane_index": 1,
                "pane_title": "backend logs",
                "current_command": "make",
                "pane_dead": False,
                "pane_active": True,
                "summary": "ispecfull 1.1 window=\"backend\" title=\"backend logs\" cmd=\"make\"",
            },
            {
                "target": "ispec-0:node.1",
                "window_target": "ispec-0:node",
                "group_window_target": "ispec:node",
                "pane_id": "%3",
                "session": "ispec-0",
                "session_group": "ispec",
                "window_name": "node",
                "window_index": 1,
                "pane_index": 1,
                "pane_title": "codex resume --all /home/alex/tools/ispec-full",
                "current_command": "node",
                "pane_dead": False,
                "pane_active": True,
                "summary": "ispec-0 group=ispec 1.1 window=\"node\" title=\"codex resume --all /home/alex/tools/ispec-full\" cmd=\"node\"",
            },
        ],
    )

    payload = run_tool(
        name="assistant_list_tmux_panes",
        args={"session_name": "ispec", "query": "codex"},
        core_db=db_session,
        schedule_db=None,
        user=_internal_user(),
        api_schema=None,
        user_message="show the codex panes in ispec",
    )

    assert payload["ok"] is True
    result = payload["result"]
    assert result["count"] == 1
    assert result["session_name"] == "ispec"
    assert result["items"][0]["session"] == "ispec-0"
    assert result["items"][0]["session_group"] == "ispec"
    assert "codex resume" in str(result["items"][0]["pane_title"])



def test_openai_tools_for_user_hides_tmux_tools_for_non_allowlisted_user(monkeypatch):
    monkeypatch.setenv("ISPEC_ASSISTANT_TMUX_TOOLS_ENABLED", "1")
    monkeypatch.setattr("ispec.assistant.tools._tmux_tools_status", lambda: (True, None))
    monkeypatch.setattr(
        "ispec.assistant.tools._code_tool_access_status",
        lambda user: (False, "Code tools are unavailable for viewer; add that username to /tmp/assistant-code-tool-users.local.txt."),
    )

    tool_names = {
        tool["function"]["name"]
        for tool in openai_tools_for_user(_internal_user())
        if isinstance(tool, dict) and isinstance(tool.get("function"), dict)
    }
    assert "assistant_list_tmux_panes" not in tool_names
    assert "assistant_capture_tmux_pane" not in tool_names
    assert "assistant_compare_tmux_pane" not in tool_names



def test_tmux_list_candidate_panes_dedupes_grouped_sessions(monkeypatch):
    monkeypatch.setattr("ispec.assistant.tools._tmux_allowlist_entries", lambda: ["ispec"])
    monkeypatch.setattr("ispec.assistant.tools._tmux_allowlist_path", lambda: (None, False))
    monkeypatch.setattr("ispec.assistant.tools._tmux_default_session_name", lambda: "ispecfull")
    monkeypatch.setattr(
        "ispec.assistant.tools._tmux_raw",
        lambda *args: type(
            "Proc",
            (),
            {
                "returncode": 0,
                "stdout": (
                    "ispec-0\tispec\t1\tnode\t1\t%0\tcodex resume --all /home/alex/tools/ispec-full\tnode\t0\t1\n"
                    "ispec-1\tispec\t1\tnode\t1\t%0\tcodex resume --all /home/alex/tools/ispec-full\tnode\t0\t1\n"
                ),
                "stderr": "",
            },
        )(),
    )

    items = _tmux_list_candidate_panes()
    assert len(items) == 1
    item = items[0]
    assert item["pane_id"] == "%0"
    assert item["pane_number"] == 0
    assert item["capture_target"] == "%0"
    assert item["preferred_alias"] == "ispec:node.1"
    assert item["session_names"] == ["ispec-0", "ispec-1"]
    assert "ispec-0:node.1" in item["target_aliases"]
    assert "ispec-1:node.1" in item["target_aliases"]



def test_tmux_find_allowed_pane_accepts_group_alias_and_numeric_id(monkeypatch):
    pane = {
        "target": "ispec:node.1",
        "preferred_alias": "ispec:node.1",
        "capture_target": "%0",
        "target_aliases": ["ispec:node.1", "ispec-0:node.1", "ispec-1:node.1"],
        "window_aliases": ["ispec:node", "ispec-0:node", "ispec-1:node"],
        "group_target": "ispec:node.1",
        "group_window_target": "ispec:node",
        "pane_id": "%0",
        "pane_number": 0,
        "session": "ispec-0",
        "session_names": ["ispec-0", "ispec-1"],
        "session_group": "ispec",
        "window_name": "node",
        "window_index": 1,
        "pane_index": 1,
        "pane_title": "codex resume --all /home/alex/tools/ispec-full",
        "current_command": "node",
        "pane_dead": False,
        "pane_active": True,
    }
    monkeypatch.setattr("ispec.assistant.tools._tmux_list_allowed_panes", lambda: [pane])

    assert _tmux_find_allowed_pane("ispec-1:node.1") == pane
    assert _tmux_find_allowed_pane("%0") == pane
    assert _tmux_find_allowed_pane("0") == pane
