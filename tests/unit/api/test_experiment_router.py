import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from ispec.api.routes.routes import generate_crud_router
from ispec.db.crud import ExperimentCRUD
from ispec.db.models import Project, Experiment
from ispec.db.connect import get_session_dep, make_session_factory, sqlite_engine, initialize_db

pytestmark = pytest.mark.testclient


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
    assert resp.json()["is_qc"] is False
    assert resp.json().get("qc_instrument") is None

    resp = client.delete(f"/experiments/{exp_id}")
    assert resp.status_code == 200
    assert resp.json()["status"] == "deleted"

    resp = client.get(f"/experiments/{exp_id}")
    assert resp.status_code == 404


def test_experiment_qc_classification_from_mapping(client, monkeypatch, tmp_path):
    mapping_path = tmp_path / "qc-experiments.json"
    mapping_path.write_text(
        json.dumps(
            {
                "experiments": {
                    "99990": {"qc_instrument": "Orbitrap Exploris"},
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("ISPEC_QC_EXPERIMENTS_JSON", str(mapping_path))

    from ispec.api import qc as qc_module

    qc_module.clear_qc_map_cache()

    with client.session_factory() as db:  # type: ignore[attr-defined]
        project = Project(prj_AddedBy="tester", prj_ProjectTitle="P1")
        db.add(project)
        db.flush()
        db.add(
            Experiment(
                id=99990,
                project_id=project.id,
                record_no="99990",
                exp_Name="QC Exploris",
            )
        )
        db.commit()

    resp = client.get("/experiments/99990")
    assert resp.status_code == 200
    assert resp.json()["is_qc"] is True
    assert resp.json()["qc_instrument"] == "Orbitrap Exploris"

    resp = client.get("/experiments/", params={"ids": [99990]})
    assert resp.status_code == 200
    payload = resp.json()
    assert len(payload) == 1
    assert payload[0]["id"] == 99990
    assert payload[0]["is_qc"] is True
    assert payload[0]["qc_instrument"] == "Orbitrap Exploris"
