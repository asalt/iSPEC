from __future__ import annotations

from ispec.assistant.tools import openai_tools_for_user, run_tool


def test_repo_tools_disabled_by_default(db_session, monkeypatch):
    monkeypatch.delenv("ISPEC_ASSISTANT_ENABLE_REPO_TOOLS", raising=False)
    monkeypatch.delenv("ISPEC_ASSISTANT_REPO_ROOT", raising=False)

    payload = run_tool(
        name="repo_search",
        args={"query": "hello", "path": "iSPEC/src", "limit": 5},
        core_db=db_session,
        schedule_db=None,
        user=None,
        api_schema=None,
    )
    assert payload["ok"] is False
    assert "disabled" in (payload.get("error") or "").lower()

    tool_names = {tool["function"]["name"] for tool in openai_tools_for_user(user=None)}
    assert "repo_search" not in tool_names


def test_repo_tools_can_search_and_read_files_when_enabled(tmp_path, db_session, monkeypatch):
    monkeypatch.setenv("ISPEC_ASSISTANT_ENABLE_REPO_TOOLS", "1")
    monkeypatch.setenv("ISPEC_ASSISTANT_REPO_ROOT", str(tmp_path))

    target = tmp_path / "iSPEC" / "src" / "ispec" / "example.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("def hello():\n    return 'MAGICWORD'\n", encoding="utf-8")

    search_payload = run_tool(
        name="repo_search",
        args={"query": "MAGICWORD", "path": "iSPEC/src", "limit": 10, "regex": False, "ignore_case": False},
        core_db=db_session,
        schedule_db=None,
        user=None,
        api_schema=None,
    )
    assert search_payload["ok"] is True
    matches = search_payload["result"]["matches"]
    assert matches
    assert any(match["path"].endswith("iSPEC/src/ispec/example.py") for match in matches)

    read_payload = run_tool(
        name="repo_read_file",
        args={"path": "iSPEC/src/ispec/example.py", "start_line": 1, "max_lines": 5},
        core_db=db_session,
        schedule_db=None,
        user=None,
        api_schema=None,
    )
    assert read_payload["ok"] is True
    assert "MAGICWORD" in read_payload["result"]["content"]

    list_payload = run_tool(
        name="repo_list_files",
        args={"query": "example.py", "path": "iSPEC/src", "limit": 20},
        core_db=db_session,
        schedule_db=None,
        user=None,
        api_schema=None,
    )
    assert list_payload["ok"] is True
    assert any(path.endswith("iSPEC/src/ispec/example.py") for path in list_payload["result"]["files"])

    tool_names = {tool["function"]["name"] for tool in openai_tools_for_user(user=None)}
    assert "repo_search" in tool_names
    assert "repo_read_file" in tool_names


def test_repo_tools_refuse_env_files(tmp_path, db_session, monkeypatch):
    monkeypatch.setenv("ISPEC_ASSISTANT_ENABLE_REPO_TOOLS", "1")
    monkeypatch.setenv("ISPEC_ASSISTANT_REPO_ROOT", str(tmp_path))

    env_path = tmp_path / ".env"
    env_path.write_text("SECRET=1\n", encoding="utf-8")

    payload = run_tool(
        name="repo_read_file",
        args={"path": ".env", "start_line": 1, "max_lines": 5},
        core_db=db_session,
        schedule_db=None,
        user=None,
        api_schema=None,
    )
    assert payload["ok"] is False
    assert "not allowed" in (payload.get("error") or "").lower()
