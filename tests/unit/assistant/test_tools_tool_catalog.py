from __future__ import annotations

from ispec.assistant.tools import run_tool
from ispec.db.models import AuthUser, UserRole


def test_assistant_list_tools_filters_unavailable_by_default(db_session):
    user = AuthUser(
        username="viewer",
        password_hash="hash",
        password_salt="salt",
        password_iterations=1,
        role=UserRole.viewer,
        is_active=True,
    )
    payload = run_tool(
        name="assistant_list_tools",
        args={},
        core_db=db_session,
        schedule_db=None,
        user=user,
        api_schema=None,
        user_message="what tools do you have?",
    )
    assert payload["ok"] is True
    assert payload["result"]["available_tools"]
    tool_names = {item["name"] for item in payload["result"]["available_tools"]}
    assert "assistant_stats" in tool_names
    assert "assistant_recent_agent_commands" in tool_names


def test_assistant_list_tools_can_include_unavailable_for_internal(db_session, monkeypatch):
    monkeypatch.delenv("ISPEC_ASSISTANT_ENABLE_REPO_TOOLS", raising=False)
    # Keep this test stable even if the runner exports ISPEC_STATE_DIR=.pids.
    monkeypatch.setenv("ISPEC_STATE_DIR", "/tmp/ispec-state")
    user = AuthUser(
        username="viewer",
        password_hash="hash",
        password_salt="salt",
        password_iterations=1,
        role=UserRole.viewer,
        is_active=True,
    )
    payload = run_tool(
        name="assistant_list_tools",
        args={"include_unavailable": True},
        core_db=db_session,
        schedule_db=None,
        user=user,
        api_schema=None,
        user_message="list tools including unavailable",
    )
    assert payload["ok"] is True
    result = payload["result"]
    assert result["include_unavailable"] is True
    unavailable = result.get("unavailable_tools")
    assert isinstance(unavailable, list) and unavailable
    unavailable_names = {item["name"] for item in unavailable if isinstance(item, dict) and item.get("name")}
    assert "repo_search" in unavailable_names


def test_assistant_list_tools_includes_repo_tools_when_enabled(db_session, monkeypatch):
    monkeypatch.setenv("ISPEC_ASSISTANT_ENABLE_REPO_TOOLS", "1")
    monkeypatch.setattr("ispec.assistant.tools._code_tool_access_status", lambda user: (True, None))
    user = AuthUser(
        username="viewer",
        password_hash="hash",
        password_salt="salt",
        password_iterations=1,
        role=UserRole.viewer,
        is_active=True,
    )
    payload = run_tool(
        name="assistant_list_tools",
        args={"query": "repo", "limit": 50},
        core_db=db_session,
        schedule_db=None,
        user=user,
        api_schema=None,
        user_message="what tools do you have?",
    )
    assert payload["ok"] is True
    tool_names = {item["name"] for item in payload["result"]["available_tools"]}
    assert "repo_search" in tool_names


def test_assistant_list_tools_query_reports_unavailable_matches(db_session):
    user = AuthUser(
        username="viewer",
        password_hash="hash",
        password_salt="salt",
        password_iterations=1,
        role=UserRole.viewer,
        is_active=True,
    )
    payload = run_tool(
        name="assistant_list_tools",
        args={"query": "agent commands"},
        core_db=db_session,
        schedule_db=None,
        user=user,
        api_schema=None,
        user_message="show agent commands tools",
    )
    assert payload["ok"] is True
    result = payload["result"]
    assert result["counts"]["matched_total"] >= 1
    tool_names = {item.get("name") for item in result.get("available_tools") or []}
    assert "assistant_recent_agent_commands" in tool_names
    assert result["counts"]["matched_unavailable_total"] == 0


def test_assistant_list_tools_reports_code_tools_unavailable_for_non_allowlisted_user(db_session, monkeypatch):
    monkeypatch.setenv("ISPEC_ASSISTANT_ENABLE_REPO_TOOLS", "1")
    monkeypatch.setattr(
        "ispec.assistant.tools._code_tool_access_status",
        lambda user: (False, "Code tools are unavailable for viewer; add that username to /tmp/assistant-code-tool-users.local.txt."),
    )
    user = AuthUser(
        username="viewer",
        password_hash="hash",
        password_salt="salt",
        password_iterations=1,
        role=UserRole.viewer,
        is_active=True,
    )
    payload = run_tool(
        name="assistant_list_tools",
        args={"include_unavailable": True, "query": "repo"},
        core_db=db_session,
        schedule_db=None,
        user=user,
        api_schema=None,
        user_message="show repo tools",
    )
    assert payload["ok"] is True
    unavailable = payload["result"].get("unavailable_tools") or []
    repo_search = next(item for item in unavailable if item.get("name") == "repo_search")
    assert "Code tools are unavailable" in str(repo_search.get("unavailable_reason") or "")
