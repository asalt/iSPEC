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


def test_build_ispec_context_includes_project_details_for_explicit_project_lookup(db_session):
    project = Project(
        prj_AddedBy="test",
        prj_ProjectTitle="Detailed Project",
        prj_ProjectQuestions="Which proteins matter most?",
        prj_ProjectBackground="Investigating plasma proteins linked to inflammation.",
        prj_ProjectCoreTasks="QC, PCA, clustering, and targeted follow-up.",
        prj_ProjectSuggestions2Customer="Bring a target list and note any samples removed during QC.",
    )
    db_session.add(project)
    db_session.commit()
    db_session.refresh(project)

    context = build_ispec_context(db_session, message=f"What should I know about project {project.id}?", state={})

    assert isinstance(context.get("projects"), list)
    payload = context["projects"][0]
    assert payload["id"] == int(project.id)
    assert payload["question"] == "Which proteins matter most?"
    assert "Investigating plasma proteins" in payload["background"]
    assert "QC, PCA" in payload["core_tasks"]
    assert "target list" in payload["suggestions"]


def test_build_ispec_context_includes_current_project_for_results_followup(db_session):
    project = Project(
        prj_AddedBy="test",
        prj_ProjectTitle="Focused Followup Project",
        prj_ProjectQuestions="Are the filtered samples still informative?",
    )
    db_session.add(project)
    db_session.commit()
    db_session.refresh(project)

    context = build_ispec_context(
        db_session,
        message="I found bad samples in the results directory and a removed folder.",
        state={"current_project_id": int(project.id)},
    )

    assert context["current_project"]["id"] == int(project.id)
    assert context["current_project"]["question"] == "Are the filtered samples still informative?"
