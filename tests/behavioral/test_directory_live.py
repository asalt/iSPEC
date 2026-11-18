"""Eight synthetic, read-only OLMo directory conversations."""
import json
import os
from types import SimpleNamespace

import pytest

from ispec.api.routes import support
from ispec.assistant.connect import get_assistant_session
from ispec.db.connect import get_session
from ispec.db.models import AuthUser, AuthUserProject, ExperimentRun, Project, ProjectFile
from ispec.db.models.auth import ProjectAccessMode, UserRole
from ispec.schedule.connect import get_schedule_session
from scripts.jeff_router_catalog import ToolCatalog
from tests.behavioral.test_olmo_application_live import application_store, seed_records  # noqa: F401 (pytest fixture)

pytestmark = [pytest.mark.behavioral, pytest.mark.skipif(os.getenv("ISPEC_RUN_LLM_TESTS") != "1", reason="local GPU only")]
schema_request = "Which API operations accept an experiment run identifier? Inspect the schema."
CASES = [
    (True, ("experiments","misc"), schema_request, "search_api"),
    (True, ("experiments",), schema_request, "search_api"),
    (True, ("misc",), "Get experiment run 764 and show its run number and search number.", "get_experiment_run"),
    (False, ("projects",), "List the attachments on project 88.", "project_files_for_project"),
    (False, ("misc",), "Look up person 73 and tell me their institution.", None),
    (False, ("projects","misc"), "Get experiment run 764 and show its run number and search number.", None),
    (False, ("files",), "Show the files attached to project 89.", "denied"),
    (False, (), "Hi iSPEC, good morning!", None),
]


def turn_directory(user):
    staff, groups, _, _ = CASES[int(user.username[1:])-1]
    schemas = support.comment_workflow_tools(tools=support.openai_tools_for_user(user))
    return ToolCatalog.from_authorized_schemas(schemas=schemas, staff=staff), groups


@pytest.mark.parametrize("case", list(range(8)), ids=[f"D{i}" for i in range(1,9)])
def test_directory(case, application_store, monkeypatch, tmp_path):  # noqa: F811 (pytest fixture)
    store = application_store
    seed_records(store=store)
    staff, _, message, expected = CASES[case]
    calls = []
    execute = support.run_tool

    def observe(**kwargs):
        name = kwargs["name"]
        assert name in {"get_experiment_run","project_files_for_project","search_api","my_projects","get_project","experiments_for_project","latest_experiment_runs"}
        result = execute(**kwargs)
        calls.append((name, kwargs["args"], result))
        return result

    monkeypatch.setattr(support, "run_tool", observe)
    schema = {"paths": {"/fixture/experiment-runs/{run_id}": {"get": {"summary":"Retrieve experiment run by identifier","operationId":"read_experiment_run","parameters":[{"name":"run_id","in":"path"}]}}}}
    with get_session(store.core_db_path) as db, get_assistant_session(store.assistant_db_path) as adb, get_schedule_session(store.schedule_db_path) as schedule:
        actor = AuthUser(username=f"D{case+1}", role=UserRole.editor if staff else UserRole.client,
                         project_access_mode=ProjectAccessMode.all_projects if staff else ProjectAccessMode.explicit_projects,
                         password_hash="fixture", password_salt="fixture")
        db.add(actor)
        for project in (88,89):
            db.add(Project(id=project, prj_AddedBy="fixture", prj_ProjectTitle=f"Synthetic {project}"))
        db.flush()
        db.add(AuthUserProject(user_id=actor.id, project_id=88))
        db.add(ExperimentRun(id=764,experiment_id=58101,run_no=4,search_no=7))
        for project in (88,89):
            db.add(ProjectFile(project_id=project, prjfile_FileName=f"fixture-{project}.txt",prjfile_SizeBytes=1,prjfile_Data=b"x"))
        db.flush()
        response = support.chat(support.ChatRequest(sessionId=actor.username,message=message,history=[]),
            request=SimpleNamespace(app=SimpleNamespace(openapi=lambda:schema)), assistant_db=adb,core_db=db,schedule_db=schedule,user=actor)
        (tmp_path/"directory.json").write_text(json.dumps({"message":message,"reply":response.message,"calls":calls},default=str,indent=2))
        if expected == "denied":
            assert any(n=="project_files_for_project" and not r["ok"] for n,a,r in calls)
        elif expected:
            assert any(n==expected and r["ok"] for n,a,r in calls)
        else:
            assert not calls
