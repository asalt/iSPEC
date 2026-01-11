from __future__ import annotations

from ispec.assistant.tools import parse_tool_call, run_tool
from ispec.db.models import Project


def test_parse_tool_call_accepts_tool_calls_fence_function_syntax():
    text = "```tool_calls\nprojects(project_id=1498)\n```"
    assert parse_tool_call(text) == ("projects", {"project_id": 1498})


def test_parse_tool_call_accepts_tool_calls_fence_json_object():
    text = '```tool_calls\n{"name":"search_projects","arguments":{"query":"example"}}\n```'
    assert parse_tool_call(text) == ("search_projects", {"query": "example"})


def test_run_tool_projects_alias_resolves_to_get_project(db_session):
    project = Project(prj_AddedBy="test", prj_ProjectTitle="Demo project")
    db_session.add(project)
    db_session.commit()

    payload = run_tool(
        name="projects",
        args={"project_id": int(project.id)},
        core_db=db_session,
        schedule_db=None,
        user=None,
        api_schema=None,
    )
    assert payload["ok"] is True
    assert payload["tool"] == "get_project"
    assert payload["result"]["id"] == int(project.id)
