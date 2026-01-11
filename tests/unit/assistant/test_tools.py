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
    assert "count_all_projects" in payload.get("error", "")


def test_count_all_projects_counts_everything(db_session):
    db_session.add_all(
        [
            Project(prj_AddedBy="test", prj_ProjectTitle="One"),
            Project(prj_AddedBy="test", prj_ProjectTitle="Two", prj_Current_FLAG=True),
            Project(prj_AddedBy="test", prj_ProjectTitle="Three", prj_Current_FLAG=True),
        ]
    )
    db_session.commit()

    payload = run_tool(
        name="count_all_projects",
        args={},
        core_db=db_session,
        schedule_db=None,
        user=None,
        api_schema=None,
    )
    assert payload["ok"] is True
    assert payload["result"]["count"] == 3
    assert payload["result"]["scope"] == "all"


def test_count_current_projects_counts_only_current(db_session):
    db_session.add_all(
        [
            Project(prj_AddedBy="test", prj_ProjectTitle="One"),
            Project(prj_AddedBy="test", prj_ProjectTitle="Two", prj_Current_FLAG=True),
            Project(prj_AddedBy="test", prj_ProjectTitle="Three", prj_Current_FLAG=True),
        ]
    )
    db_session.commit()

    payload = run_tool(
        name="count_current_projects",
        args={},
        core_db=db_session,
        schedule_db=None,
        user=None,
        api_schema=None,
    )
    assert payload["ok"] is True
    assert payload["result"]["count"] == 2
    assert payload["result"]["scope"] == "current"
