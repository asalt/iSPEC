from __future__ import annotations

from ispec.assistant.tools import openai_tools_for_user, run_tool
from ispec.db.models import AuthUser, Person, Project, ProjectComment, UserRole


def _make_user(username: str, *, role: UserRole) -> AuthUser:
    return AuthUser(
        username=username,
        password_hash="hash",
        password_salt="salt",
        password_iterations=1,
        role=role,
        is_active=True,
    )


def test_create_project_comment_tool_hidden_for_viewer():
    user = _make_user("viewer", role=UserRole.viewer)
    tool_names = {tool["function"]["name"] for tool in openai_tools_for_user(user)}
    assert "create_project_comment" not in tool_names


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

