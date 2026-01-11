from __future__ import annotations

from ispec.db.models import Project

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


def test_count_projects_ignores_invalid_status_filter(db_session):
    db_session.add_all(
        [
            Project(prj_AddedBy="test", prj_ProjectTitle="One"),
            Project(prj_AddedBy="test", prj_ProjectTitle="Two"),
            Project(prj_AddedBy="test", prj_ProjectTitle="Three"),
        ]
    )
    db_session.commit()

    payload = run_tool(
        name="count_projects",
        args={"status": "success"},
        core_db=db_session,
        schedule_db=None,
        user=None,
        api_schema=None,
    )
    assert payload["ok"] is True
    assert payload["result"]["count"] == 3
    assert payload["result"]["status"] is None
