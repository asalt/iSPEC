import importlib

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


@pytest.fixture
def full_client(tmp_path, monkeypatch):
    db_url = f"sqlite:///{tmp_path}/run-full.db"
    engine = sqlite_engine(db_url)
    initialize_db(engine)
    test_session = make_session_factory(engine)

    monkeypatch.setenv("ISPEC_API_RESOURCES", "all")
    import ispec.api.routes.routes as routes_mod

    routes_mod = importlib.reload(routes_mod)
    app = FastAPI()
    app.include_router(routes_mod.router)

    def override_get_session():
        with test_session() as session:
            yield session

    app.dependency_overrides[get_session_dep] = override_get_session

    with TestClient(app) as client:
        client.session_factory = test_session  # type: ignore[attr-defined]
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

    payload = {"experiment_id": exp_id, "run_no": 1, "search_no": 1, "label": "none"}
    resp = client.post("/experiment_runs/", json=payload)
    assert resp.status_code == 201
    run_id = resp.json()["id"]
    assert resp.json()["label"] == "0"
    assert resp.json()["sample_name"] == f"{exp_id}_1_1_0"

    resp = client.get(f"/experiment_runs/{run_id}")
    assert resp.status_code == 200
    assert resp.json()["experiment_id"] == exp_id
    assert resp.json()["label"] == "0"

    resp = client.delete(f"/experiment_runs/{run_id}")
    assert resp.status_code == 200
    assert resp.json()["status"] == "deleted"

    resp = client.get(f"/experiment_runs/{run_id}")
    assert resp.status_code == 404


def test_experiment_runs_by_project_includes_parent_experiment_metadata(full_client):
    with full_client.session_factory() as db:  # type: ignore[attr-defined]
        project = Project(id=1489, prj_AddedBy="tester", prj_ProjectTitle="P1")
        db.add(project)
        experiment = Experiment(
            id=58150,
            project_id=1489,
            record_no="58150",
            exp_Name="peripheral blood 051526-370-mono mi16 m",
            exp_CellTissue="peripheral blood",
            exp_Genotype="051526-370-mono mi16 m",
            exp_Treatment="miRNA mimic",
        )
        db.add(experiment)
        db.add(
            ExperimentRun(
                experiment_id=58150,
                run_no=1,
                search_no=4,
                label="0",
                label_type="none",
                sample_name="58150_1_4_0",
                ms_instrument="BCM-MSPC-Ultra2",
                ref_database="GENCODE_hs",
                taxon_id=9606,
                legacy_grouper_started=True,
                legacy_grouper_completed=True,
                legacy_e2g_import_started=True,
                legacy_e2g_import_fix_requested=True,
            )
        )
        db.commit()

    resp = full_client.get("/experiment_runs/by_project/1489")
    assert resp.status_code == 200
    rows = resp.json()
    assert len(rows) == 1
    assert rows[0]["sample_name"] == "58150_1_4_0"
    assert rows[0]["ms_instrument"] == "BCM-MSPC-Ultra2"
    assert rows[0]["ref_database"] == "GENCODE_hs"
    assert rows[0]["taxon_id"] == 9606
    assert rows[0]["experiment_record_no"] == "58150"
    assert rows[0]["experiment_cell_tissue"] == "peripheral blood"
    assert rows[0]["experiment_genotype"] == "051526-370-mono mi16 m"
    assert rows[0]["experiment_treatment"] == "miRNA mimic"
    assert rows[0]["legacy_grouping_status"] == "complete"
    assert rows[0]["legacy_import_status"] == "needs_fix"


def test_experiment_run_detail_includes_derived_legacy_status(client):
    with client.session_factory() as db:  # type: ignore[attr-defined]
        project = Project(prj_AddedBy="tester", prj_ProjectTitle="P1")
        db.add(project)
        db.flush()
        experiment = Experiment(project_id=project.id, record_no="58150")
        db.add(experiment)
        db.flush()
        run = ExperimentRun(
            experiment_id=experiment.id,
            run_no=1,
            search_no=4,
            label="0",
            legacy_grouper_started=True,
            legacy_e2g_import_error="legacy import failed",
        )
        db.add(run)
        db.commit()
        run_id = run.id

    resp = client.get(f"/experiment_runs/{run_id}")
    assert resp.status_code == 200
    assert resp.json()["legacy_grouping_status"] == "running"
    assert resp.json()["legacy_import_status"] == "failed"


def test_run_source_snapshots_cannot_be_supplied_or_changed_through_api(full_client):
    with full_client.session_factory() as db:
        db.add(Project(id=1494, prj_AddedBy="test", prj_ProjectTitle="P"))
        db.add(Experiment(id=57454, project_id=1494, record_no="57454"))
        db.commit()

    snapshots = {
        "legacy_source_baseline_json": '{"forged": "baseline"}',
        "legacy_source_conflicts_json": '{"forged": "conflict"}',
    }
    payload = {"experiment_id": 57454, "run_no": 1, "search_no": 2, "label": "0"}
    response = full_client.post("/experiment_runs/", json={**payload, **snapshots})
    assert response.status_code == 201
    run_id = response.json()["id"]
    with full_client.session_factory() as db:
        run = db.get(ExperimentRun, run_id)
        assert run.legacy_source_baseline_json is None
        assert run.legacy_source_conflicts_json is None
        run.legacy_source_baseline_json = '{"saved": "baseline"}'
        run.legacy_source_conflicts_json = '{"saved": "conflict"}'
        db.commit()

    response = full_client.put(
        f"/experiment_runs/{run_id}",
        json={**payload, **snapshots, "ms_instrument": "local correction"},
    )
    assert response.status_code == 200
    with full_client.session_factory() as db:
        run = db.get(ExperimentRun, run_id)
        assert run.ms_instrument == "local correction"
        assert run.legacy_source_baseline_json == '{"saved": "baseline"}'
        assert run.legacy_source_conflicts_json == '{"saved": "conflict"}'
