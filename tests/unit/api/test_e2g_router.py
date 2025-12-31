import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from ispec.api.routes.routes import generate_crud_router
from ispec.db.crud import E2GCRUD
from ispec.db.models import Project, Experiment, ExperimentRun, E2G
from ispec.db.connect import get_session_dep, make_session_factory, sqlite_engine, initialize_db


@pytest.fixture
def client(tmp_path):
    db_url = f"sqlite:///{tmp_path}/e2g.db"
    engine = sqlite_engine(db_url)
    initialize_db(engine)
    test_session = make_session_factory(engine)

    app = FastAPI()
    route_prefix_map: dict[str, str] = {}
    router = generate_crud_router(
        model=E2G,
        crud_class=E2GCRUD,
        prefix="/experiment_to_gene",
        tag="E2G",
        exclude_fields=set(),
        create_exclude_fields={"id", "E2G_CreationTS", "E2G_ModificationTS"},
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


def test_e2g_router_exposes_by_run_endpoint(client):
    # Endpoint should exist even with no rows present
    resp = client.get("/experiment_to_gene/by_run/1")
    assert resp.status_code == 200
    assert resp.json() == []


def test_e2g_crud(client):
    # seed project -> experiment -> run
    with client.session_factory() as db:  # type: ignore[attr-defined]
        project = Project(prj_AddedBy="tester", prj_ProjectTitle="X")
        db.add(project)
        db.flush()

        experiment = Experiment(project_id=project.id, record_no=f"{project.id:05d}-01")
        db.add(experiment)
        db.flush()

        run = ExperimentRun(experiment_id=experiment.id, run_no=1, search_no=1)
        db.add(run)
        db.commit()
        db.refresh(run)

        run_id = run.id

    payload = {"experiment_run_id": run_id, "gene": "TP53", "geneidtype": "symbol"}
    resp = client.post("/experiment_to_gene/", json=payload)
    assert resp.status_code == 201
    e2g_id = resp.json()["id"]

    resp = client.get(f"/experiment_to_gene/{e2g_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["experiment_run_id"] == run_id
    assert data["gene"] == "TP53"

    # convenience listing by run
    resp = client.get(f"/experiment_to_gene/by_run/{run_id}")
    assert resp.status_code == 200
    items = resp.json()
    assert any(row["id"] == e2g_id for row in items)

    resp = client.delete(f"/experiment_to_gene/{e2g_id}")
    assert resp.status_code == 200
    assert resp.json()["status"] == "deleted"

    resp = client.get(f"/experiment_to_gene/{e2g_id}")
    assert resp.status_code == 404
