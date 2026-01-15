from __future__ import annotations

from ispec.assistant.context import build_ispec_context
from ispec.assistant.tools import run_tool
from ispec.db.models import AuthUser, AuthUserProject, Person, Project, ProjectComment, UserRole


def test_build_ispec_context_includes_client_notes_for_current_project(db_session):
    project = Project(id=20, prj_AddedBy="test", prj_ProjectTitle="Client Notes Project")
    user = AuthUser(
        username="client",
        password_hash="hash",
        password_salt="salt",
        password_iterations=1,
        role=UserRole.client,
        is_active=True,
    )
    db_session.add_all([project, user])
    db_session.commit()
    db_session.refresh(user)

    db_session.add(AuthUserProject(user_id=int(user.id), project_id=int(project.id)))
    db_session.commit()

    payload = run_tool(
        name="create_project_comment",
        args={"project_id": int(project.id), "comment": "Client note: check PCA plots.", "confirm": True},
        core_db=db_session,
        schedule_db=None,
        user=user,
        api_schema=None,
        user_message="please save this note to project history",
    )
    assert payload["ok"] is True

    assistant_person = (
        db_session.query(Person)
        .filter(Person.ppl_Name_First == "iSPEC")
        .filter(Person.ppl_Name_Last == "Assistant")
        .first()
    )
    assert assistant_person is not None
    db_session.add(
        ProjectComment(
            project_id=int(project.id),
            person_id=int(assistant_person.id),
            com_Comment="Staff note: internal details.",
            com_CommentType="assistant_note",
            com_AddedBy="editor",
        )
    )
    db_session.commit()

    context = build_ispec_context(
        db_session,
        message="hello",
        state={"current_project_id": int(project.id)},
        user=user,
    )
    notes = context.get("current_project_notes")
    assert isinstance(notes, dict)
    assert notes["project_id"] == int(project.id)
    assert notes["total"] == 1
    items = notes.get("items")
    assert isinstance(items, list)
    assert any("Client note" in (item.get("comment") or "") for item in items)
