import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from ispec.api.routes.routes import generate_crud_router, ROUTE_PREFIX_BY_TABLE
from ispec.db.crud import ProjectCRUD
from ispec.db.models import Project
from ispec.db.connect import get_session, make_session_factory, sqlite_engine, initialize_db


@pytest.fixture
def client(tmp_path):
    db_url = f"sqlite:///{tmp_path}/test.db"
    engine = sqlite_engine(db_url)
    initialize_db(engine)
    test_session = make_session_factory(engine)

    # ensure clean ROUTE_PREFIX_BY_TABLE for test isolation
    ROUTE_PREFIX_BY_TABLE.clear()

    app = FastAPI()
    project_router = generate_crud_router(
        model=Project,
        crud_class=ProjectCRUD,
        prefix="/projects",
        tag="Project",
        exclude_fields=set(),
        create_exclude_fields={"id", "prj_CreationTS", "prj_ModificationTS"},
    )
    app.include_router(project_router)

    def override_get_session():
        with test_session() as session:
            yield session

    app.dependency_overrides[get_session] = override_get_session

    with TestClient(app) as client:
        client.session_factory = test_session  # type: ignore[attr-defined]
        yield client


def test_project_router_crud_and_route_prefix(client):
    # ROUTE_PREFIX_BY_TABLE should register the project table for FK resolution
    assert ROUTE_PREFIX_BY_TABLE["project"] == "/projects"

    # schema endpoint should expose model schema
    resp = client.get("/projects/schema")
    assert resp.status_code == 200
    schema = resp.json()
    assert "prj_ProjectTitle" in schema["properties"]

    payload = {
        "prj_AddedBy": "tester",
        "prj_ProjectTitle": "Test Project",
        "prj_ProjectBackground": "Background info",
    }

    # create project
    resp = client.post("/projects/", json=payload)
    assert resp.status_code == 201

    with client.session_factory() as db:  # type: ignore[attr-defined]
        project_id = db.query(Project).filter_by(prj_ProjectTitle="Test Project").first().id

    # retrieve project
    resp = client.get(f"/projects/{project_id}")
    assert resp.status_code == 200
    assert resp.json()["prj_ProjectTitle"] == payload["prj_ProjectTitle"]

    # delete project
    resp = client.delete(f"/projects/{project_id}")
    assert resp.status_code == 200
    assert resp.json()["status"] == "deleted"

    # ensure project is gone
    resp = client.get(f"/projects/{project_id}")
    assert resp.status_code == 404
