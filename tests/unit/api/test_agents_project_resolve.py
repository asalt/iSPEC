import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from ispec.api.routes.agents import router as agents_router
from ispec.db.connect import get_session_dep, make_session_factory
from ispec.db.models import Project, initialize_db, sqlite_engine

pytestmark = pytest.mark.testclient


def test_agents_resolve_projects_by_display_id_and_numeric_id(tmp_path):
    db_url = f"sqlite:///{tmp_path}/ispec.db"
    engine = sqlite_engine(db_url)
    initialize_db(engine)
    session_factory = make_session_factory(engine)

    with session_factory() as session:
        session.add_all(
            [
                Project(
                    id=1498,
                    prj_AddedBy="tester",
                    prj_ProjectTitle="Project 1498",
                    prj_PRJ_DisplayID="MSPC001498",
                ),
                # Leave display id NULL; endpoint should still resolve via numeric id parsing.
                Project(
                    id=1,
                    prj_AddedBy="tester",
                    prj_ProjectTitle="Project 1",
                    prj_PRJ_DisplayID=None,
                ),
            ]
        )

    app = FastAPI()
    app.include_router(agents_router, prefix="/api")

    def override_session():
        with session_factory() as session:
            yield session

    app.dependency_overrides[get_session_dep] = override_session

    with TestClient(app) as client:
        resp = client.post(
            "/api/agents/projects/resolve",
            json={"tokens": ["MSPC001498", "/mnt/e/MSPC001498/", "MSPC000001", "999999"]},
        )
        assert resp.status_code == 200
        body = resp.json()
        resolved = {row["token"]: row for row in body["projects"]}

        assert resolved["MSPC001498"]["project_id"] == 1498
        assert resolved["MSPC001498"]["display_id"] == "MSPC001498"

        assert resolved["MSPC000001"]["project_id"] == 1
        assert resolved["MSPC000001"]["display_id"] == "MSPC000001"

        assert "999999" in set(body["unknown_tokens"])


def test_agents_project_index_is_lightweight(tmp_path):
    db_url = f"sqlite:///{tmp_path}/ispec.db"
    engine = sqlite_engine(db_url)
    initialize_db(engine)
    session_factory = make_session_factory(engine)

    with session_factory() as session:
        session.add_all(
            [
                Project(
                    id=10,
                    prj_AddedBy="tester",
                    prj_ProjectTitle="Project 10",
                    prj_PRJ_DisplayID="MSPC000010",
                ),
                Project(
                    id=11,
                    prj_AddedBy="tester",
                    prj_ProjectTitle="Project 11",
                    prj_PRJ_DisplayID="MSPC000011",
                ),
            ]
        )

    app = FastAPI()
    app.include_router(agents_router, prefix="/api")

    def override_session():
        with session_factory() as session:
            yield session

    app.dependency_overrides[get_session_dep] = override_session

    with TestClient(app) as client:
        resp = client.get("/api/agents/projects/index?min_id=11&limit=10")
        assert resp.status_code == 200
        rows = resp.json()["projects"]
        assert [row["project_id"] for row in rows] == [11]
