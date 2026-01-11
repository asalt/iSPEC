import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from ispec.api.routes.routes import generate_crud_router
from ispec.db.crud import ExperimentRunCRUD, ExperimentCRUD
from ispec.db.models import Project, Experiment, ExperimentRun
from ispec.db.connect import get_session_dep, make_session_factory, sqlite_engine, initialize_db

pytestmark = pytest.mark.testclient


@pytest.fixture
def client(tmp_path):
    db_url = f"sqlite:///{tmp_path}/run.db"
    engine = sqlite_engine(db_url)
    initialize_db(engine)
    test_session = make_session_factory(engine)

    app = FastAPI()
    route_prefix_map: dict[str, str] = {}
    router = generate_crud_router(
        model=ExperimentRun,
        crud_class=ExperimentRunCRUD,
        prefix="/experiment_runs",
        tag="ExperimentRun",
        exclude_fields=set(),
        create_exclude_fields={"id", "ExperimentRun_CreationTS", "ExperimentRun_ModificationTS"},
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


def test_experiment_run_crud(client):
    # seed project + experiment
    with client.session_factory() as db:  # type: ignore[attr-defined]
        project = Project(prj_AddedBy="tester", prj_ProjectTitle="P1")
        db.add(project)
        db.flush()

        experiment = Experiment(project_id=project.id, record_no=f"{project.id:05d}-01")
        db.add(experiment)
        db.commit()
        db.refresh(experiment)
        exp_id = experiment.id

    payload = {"experiment_id": exp_id, "run_no": 1, "search_no": 1}
    resp = client.post("/experiment_runs/", json=payload)
    assert resp.status_code == 201
    run_id = resp.json()["id"]

    resp = client.get(f"/experiment_runs/{run_id}")
    assert resp.status_code == 200
    assert resp.json()["experiment_id"] == exp_id

    resp = client.delete(f"/experiment_runs/{run_id}")
    assert resp.status_code == 200
    assert resp.json()["status"] == "deleted"

    resp = client.get(f"/experiment_runs/{run_id}")
    assert resp.status_code == 404
