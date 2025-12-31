import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from ispec.api.routes.routes import generate_crud_router
from ispec.db.crud import ExperimentCRUD
from ispec.db.models import Project, Experiment
from ispec.db.connect import get_session_dep, make_session_factory, sqlite_engine, initialize_db


@pytest.fixture
def client(tmp_path):
    db_url = f"sqlite:///{tmp_path}/exp.db"
    engine = sqlite_engine(db_url)
    initialize_db(engine)
    test_session = make_session_factory(engine)

    app = FastAPI()
    route_prefix_map: dict[str, str] = {}
    router = generate_crud_router(
        model=Experiment,
        crud_class=ExperimentCRUD,
        prefix="/experiments",
        tag="Experiment",
        exclude_fields=set(),
        create_exclude_fields={"id", "Experiment_CreationTS", "Experiment_ModificationTS"},
        route_prefix_by_table=route_prefix_map,
    )
    app.include_router(router)

    def override_get_session():
        with test_session() as session:
            yield session

    app.dependency_overrides[get_session_dep] = override_get_session

    with TestClient(app) as client:
        client.session_factory = test_session  # type: ignore[attr-defined]
        client.route_prefix_map = route_prefix_map  # type: ignore[attr-defined]
        yield client


def test_experiment_crud(client):
    # seed project
    with client.session_factory() as db:  # type: ignore[attr-defined]
        project = Project(prj_AddedBy="tester", prj_ProjectTitle="P1")
        db.add(project)
        db.commit()
        db.refresh(project)
        project_id = project.id

    payload = {"project_id": project_id, "record_no": f"{project_id:05d}-01"}
    resp = client.post("/experiments/", json=payload)
    assert resp.status_code == 201
    exp_id = resp.json()["id"]

    resp = client.get(f"/experiments/{exp_id}")
    assert resp.status_code == 200
    assert resp.json()["project_id"] == project_id

    resp = client.delete(f"/experiments/{exp_id}")
    assert resp.status_code == 200
    assert resp.json()["status"] == "deleted"

    resp = client.get(f"/experiments/{exp_id}")
    assert resp.status_code == 404
