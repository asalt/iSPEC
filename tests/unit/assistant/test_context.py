from __future__ import annotations

from ispec.assistant.context import build_ispec_context
from ispec.db.models import Project


def test_build_ispec_context_does_not_inject_current_project_for_plural_project_questions(db_session):
    project = Project(prj_AddedBy="test", prj_ProjectTitle="Focused Project")
    db_session.add(project)
    db_session.commit()
    db_session.refresh(project)

    state = {"current_project_id": int(project.id)}
    context = build_ispec_context(db_session, message="How many projects are there?", state=state)

    assert "projects" not in context
    assert "current_project" not in context


def test_build_ispec_context_includes_current_project_for_singular_project_questions(db_session):
    project = Project(prj_AddedBy="test", prj_ProjectTitle="Focused Project")
    db_session.add(project)
    db_session.commit()
    db_session.refresh(project)

    state = {"current_project_id": int(project.id)}
    context = build_ispec_context(db_session, message="Tell me about the project", state=state)

    assert "projects" not in context
    assert context["current_project"]["id"] == int(project.id)
    assert context["current_project"]["title"] == "Focused Project"


def test_build_ispec_context_includes_explicitly_mentioned_projects(db_session):
    project = Project(prj_AddedBy="test", prj_ProjectTitle="Mentioned Project")
    db_session.add(project)
    db_session.commit()
    db_session.refresh(project)

    context = build_ispec_context(db_session, message=f"Tell me about project {project.id}", state={})

    assert isinstance(context.get("projects"), list)
    assert context["projects"][0]["id"] == int(project.id)
    assert context["projects"][0]["title"] == "Mentioned Project"
