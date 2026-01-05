from __future__ import annotations

from ispec.assistant.tools import run_tool


def test_search_projects_rejects_wildcard_query(db_session):
    payload = run_tool(
        name="search_projects",
        args={"query": "*", "limit": 5},
        core_db=db_session,
        schedule_db=None,
        user=None,
        api_schema=None,
    )
    assert payload["ok"] is False
    assert "count_projects" in payload.get("error", "")

