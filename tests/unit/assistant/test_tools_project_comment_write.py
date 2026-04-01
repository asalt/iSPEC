from __future__ import annotations

import types

import pytest

from ispec.agent.commands import COMMAND_LEGACY_PUSH_PROJECT_COMMENTS
from ispec.agent.connect import get_agent_session
from ispec.agent.models import AgentCommand
from ispec.assistant.tools import openai_tools_for_user, run_tool
from ispec.db.models import AuthUser, AuthUserProject, Person, Project, ProjectComment, UserRole


def _make_user(username: str, *, role: UserRole) -> AuthUser:
    return AuthUser(
        username=username,
        password_hash="hash",
        password_salt="salt",
        password_iterations=1,
        role=role,
        is_active=True,
    )


def test_create_project_comment_tool_visible_for_viewer():
    user = _make_user("viewer", role=UserRole.viewer)
    tool_names = {tool["function"]["name"] for tool in openai_tools_for_user(user)}
    assert "create_project_comment" in tool_names


def test_create_project_comment_tool_visible_for_client():
    user = _make_user("client", role=UserRole.client)
    tool_names = {tool["function"]["name"] for tool in openai_tools_for_user(user)}
    assert "create_project_comment" in tool_names


def test_create_project_comment_tool_visible_for_service_viewer():
    user = types.SimpleNamespace(
        username="api_key",
        role=UserRole.viewer,
        can_write_project_comments=True,
    )
    tool_names = {tool["function"]["name"] for tool in openai_tools_for_user(user)}
    assert "create_project_comment" in tool_names


def test_create_project_comment_allows_service_viewer_write(db_session):
    project = Project(id=12, prj_AddedBy="test", prj_ProjectTitle="Project 12")
    db_session.add(project)
    db_session.commit()

    user = types.SimpleNamespace(
        username="api_key",
        role=UserRole.viewer,
        can_write_project_comments=True,
    )
    payload = run_tool(
        name="create_project_comment",
        args={"project_id": 12, "comment": "Note", "confirm": True},
        core_db=db_session,
        schedule_db=None,
        user=user,
        api_schema=None,
        user_message="make a note for project 12",
    )
    assert payload["ok"] is True


def test_create_project_comment_requires_client_project_access(db_session):
    allowed = Project(id=10, prj_AddedBy="test", prj_ProjectTitle="Allowed Project")
    denied = Project(id=11, prj_AddedBy="test", prj_ProjectTitle="Denied Project")
    user = _make_user("client", role=UserRole.client)
    db_session.add_all([allowed, denied, user])
    db_session.commit()
    db_session.refresh(user)

    db_session.add(AuthUserProject(user_id=int(user.id), project_id=int(allowed.id)))
    db_session.commit()

    payload = run_tool(
        name="create_project_comment",
        args={"project_id": int(denied.id), "comment": "My note", "confirm": True},
        core_db=db_session,
        schedule_db=None,
        user=user,
        api_schema=None,
        user_message="please save this note to project history",
    )
    assert payload["ok"] is False
    assert "accessible" in (payload.get("error") or "").lower()

    payload = run_tool(
        name="create_project_comment",
        args={
            "project_id": int(allowed.id),
            "comment": "Client note: please focus on PC1 vs PC2 biplot.",
            "comment_type": "meeting_note",
            "confirm": True,
        },
        core_db=db_session,
        schedule_db=None,
        user=user,
        api_schema=None,
        user_message="please save this note to project history",
    )
    assert payload["ok"] is True
    result = payload["result"]
    comment_id = int(result["comment_id"])

    comment = db_session.get(ProjectComment, comment_id)
    assert comment is not None
    assert comment.project_id == int(allowed.id)
    assert comment.com_AddedBy == "client"
    assert comment.com_CommentType == "client_note"
    assert "Client note" in (comment.com_Comment or "")


def test_create_project_comment_requires_confirm_and_explicit_user_request(db_session):
    project = Project(id=1, prj_AddedBy="test", prj_ProjectTitle="Project 1")
    user = _make_user("editor", role=UserRole.editor)
    db_session.add_all([project, user])
    db_session.commit()

    payload = run_tool(
        name="create_project_comment",
        args={"project_id": 1, "comment": "Note", "confirm": True},
        core_db=db_session,
        schedule_db=None,
        user=user,
        api_schema=None,
        user_message="Can you summarize project 1?",
    )
    assert payload["ok"] is False
    assert "explicit" in (payload.get("error") or "").lower()

    payload = run_tool(
        name="create_project_comment",
        args={"project_id": 1, "comment": "Note"},
        core_db=db_session,
        schedule_db=None,
        user=user,
        api_schema=None,
        user_message="save this to project history",
    )
    assert payload["ok"] is False
    assert "confirm" in (payload.get("error") or "").lower()

    payload = run_tool(
        name="create_project_comment",
        args={"project_id": 1, "comment": "Note", "confirm": True},
        core_db=db_session,
        schedule_db=None,
        user=user,
        api_schema=None,
        user_message="make a note for project 1",
    )
    assert payload["ok"] is True

    payload = run_tool(
        name="create_project_comment",
        args={"project_id": 1, "comment": "Note", "confirm": True},
        core_db=db_session,
        schedule_db=None,
        user=user,
        api_schema=None,
        user_message="Confirm yes commit it",
    )
    assert payload["ok"] is True

    payload = run_tool(
        name="create_project_comment",
        args={"project_id": 1, "comment": "Note", "confirm": True},
        core_db=db_session,
        schedule_db=None,
        user=user,
        api_schema=None,
        user_message="please commit the project note",
    )
    assert payload["ok"] is True


@pytest.mark.parametrize(
    "user_message",
    [
        "make a note for project 1",
        "add a comment to project 1",
        "log this in project history for project 1",
        "please save this note on project 1",
        "write this note into the project history for project 1",
    ],
)
def test_create_project_comment_accepts_common_write_variations(db_session, user_message):
    project = Project(id=1, prj_AddedBy="test", prj_ProjectTitle="Project 1")
    user = _make_user("editor", role=UserRole.editor)
    db_session.add_all([project, user])
    db_session.commit()

    payload = run_tool(
        name="create_project_comment",
        args={"project_id": 1, "comment": "Variation note", "confirm": True},
        core_db=db_session,
        schedule_db=None,
        user=user,
        api_schema=None,
        user_message=user_message,
    )
    assert payload["ok"] is True


@pytest.mark.parametrize(
    "user_message",
    [
        "help me write a comment about project 1",
        "draft a comment for project 1",
        "help me word a project note for project 1",
        "rewrite this note for project 1 before we save it",
    ],
)
def test_create_project_comment_rejects_draft_only_requests(db_session, user_message):
    project = Project(id=1, prj_AddedBy="test", prj_ProjectTitle="Project 1")
    user = _make_user("editor", role=UserRole.editor)
    db_session.add_all([project, user])
    db_session.commit()

    payload = run_tool(
        name="create_project_comment",
        args={"project_id": 1, "comment": "Draft note", "confirm": True},
        core_db=db_session,
        schedule_db=None,
        user=user,
        api_schema=None,
        user_message=user_message,
    )
    assert payload["ok"] is False
    assert "explicit" in (payload.get("error") or "").lower()


def test_create_project_comment_creates_comment_and_assistant_person(db_session):
    project = Project(id=2, prj_AddedBy="test", prj_ProjectTitle="Project 2")
    user = _make_user("editor", role=UserRole.editor)
    db_session.add_all([project, user])
    db_session.commit()

    payload = run_tool(
        name="create_project_comment",
        args={
            "project_id": 2,
            "comment": "Meeting notes: agreed on next steps.",
            "comment_type": "meeting_note",
            "confirm": True,
        },
        core_db=db_session,
        schedule_db=None,
        user=user,
        api_schema=None,
        user_message="please save this meeting note to the project history",
    )
    assert payload["ok"] is True
    result = payload["result"]
    comment_id = int(result["comment_id"])

    comment = db_session.get(ProjectComment, comment_id)
    assert comment is not None
    assert comment.project_id == 2
    assert comment.com_AddedBy == "editor"
    assert comment.com_CommentType == "meeting_note"
    assert "Meeting notes" in (comment.com_Comment or "")

    person = db_session.get(Person, int(comment.person_id))
    assert person is not None
    assert person.ppl_Name_First == "iSPEC"
    assert person.ppl_Name_Last == "Assistant"


def test_create_project_comment_enqueues_legacy_push_when_scheduler_enabled(db_session, tmp_path, monkeypatch):
    project = Project(id=3, prj_AddedBy="test", prj_ProjectTitle="Project 3")
    user = _make_user("editor", role=UserRole.editor)
    db_session.add_all([project, user])
    db_session.commit()

    agent_db_path = tmp_path / "agent.db"
    monkeypatch.setenv("ISPEC_AGENT_DB_PATH", str(agent_db_path))
    monkeypatch.setenv("ISPEC_LEGACY_PUSH_PROJECT_COMMENTS_ENABLED", "1")
    monkeypatch.setenv("ISPEC_LEGACY_PUSH_PROJECT_COMMENTS_RECENT_DAYS", "14")
    monkeypatch.setenv("ISPEC_LEGACY_PUSH_PROJECT_COMMENTS_LIMIT", "77")
    monkeypatch.delenv("ISPEC_LEGACY_PUSH_PROJECT_COMMENTS_DRY_RUN", raising=False)

    payload = run_tool(
        name="create_project_comment",
        args={"project_id": 3, "comment": "Please sync this note.", "confirm": True},
        core_db=db_session,
        schedule_db=None,
        user=user,
        api_schema=None,
        user_message="please save this note to the project history",
    )
    assert payload["ok"] is True
    enqueue = payload["result"].get("legacy_push_enqueue")
    assert isinstance(enqueue, dict)
    assert enqueue["ok"] is True
    assert enqueue["enqueued"] is True

    with get_agent_session(agent_db_path) as agent_db:
        cmd = (
            agent_db.query(AgentCommand)
            .filter(AgentCommand.command_type == COMMAND_LEGACY_PUSH_PROJECT_COMMENTS)
            .one()
        )
        assert cmd.status == "queued"
        assert int(cmd.payload_json["project_id"]) == 3
        assert int(cmd.payload_json["recent_days"]) == 14
        assert int(cmd.payload_json["limit"]) == 77
        assert cmd.payload_json["trigger"] == "project_comment_created"


def test_create_project_comment_reuses_existing_legacy_push_command(db_session, tmp_path, monkeypatch):
    project = Project(id=4, prj_AddedBy="test", prj_ProjectTitle="Project 4")
    user = _make_user("editor", role=UserRole.editor)
    db_session.add_all([project, user])
    db_session.commit()

    agent_db_path = tmp_path / "agent.db"
    monkeypatch.setenv("ISPEC_AGENT_DB_PATH", str(agent_db_path))
    monkeypatch.setenv("ISPEC_LEGACY_PUSH_PROJECT_COMMENTS_ENABLED", "1")

    with get_agent_session(agent_db_path) as agent_db:
        existing = AgentCommand(
            command_type=COMMAND_LEGACY_PUSH_PROJECT_COMMENTS,
            status="queued",
            payload_json={"project_id": 999, "trigger": "test"},
            result_json={},
        )
        agent_db.add(existing)
        agent_db.flush()
        existing_id = int(existing.id)

    payload = run_tool(
        name="create_project_comment",
        args={"project_id": 4, "comment": "Second note.", "confirm": True},
        core_db=db_session,
        schedule_db=None,
        user=user,
        api_schema=None,
        user_message="please save this note to the project history",
    )
    assert payload["ok"] is True
    enqueue = payload["result"].get("legacy_push_enqueue")
    assert isinstance(enqueue, dict)
    assert enqueue["ok"] is True
    assert enqueue["enqueued"] is False
    assert enqueue["reason"] == "already_enqueued"
    assert int(enqueue["command_id"]) == existing_id

    with get_agent_session(agent_db_path) as agent_db:
        count = (
            agent_db.query(AgentCommand)
            .filter(AgentCommand.command_type == COMMAND_LEGACY_PUSH_PROJECT_COMMENTS)
            .count()
        )
        assert count == 1
