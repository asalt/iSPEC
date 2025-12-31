from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from ispec.api.main import app
from ispec.db import operations
from ispec.db.connect import get_session, get_session_dep
from ispec.db.models import E2G, Experiment, ExperimentRun, Project
from tests.integration.sandbox_builder import build_sandbox_db


def test_experiment_run_gene_relationships(tmp_path: Path):
    db_path = tmp_path / "sandbox.db"
    summary = build_sandbox_db(
        db_path,
        experiments_per_project=2,
        runs_per_experiment=2,
        genes_per_run=3,
    )

    assert summary["experiments"] == summary["projects"] * 2
    assert summary["runs"] == summary["experiments"] * 2
    assert summary["genes"] == summary["runs"] * 3

    with get_session(file_path=str(db_path)) as session:
        project = session.query(Project).first()
        assert project is not None
        assert project.experiments, "Expected experiments linked to project"

        experiment = project.experiments[0]
        assert len(experiment.runs) == 2
        assert len(experiment.runs[0].gene_mappings) == 3

        experiment_id = experiment.id
        session.delete(experiment)
        session.flush()

        remaining_runs = session.query(ExperimentRun).filter_by(
            experiment_id=experiment_id
        )
        remaining_genes = (
            session.query(E2G)
            .join(ExperimentRun)
            .filter(ExperimentRun.experiment_id == experiment_id)
        )

        assert remaining_runs.count() == 0
        assert remaining_genes.count() == 0


def test_sandbox_db_environment_accessible_via_api(tmp_path: Path, monkeypatch):
    db_path = tmp_path / "sandbox.db"
    summary = build_sandbox_db(db_path)

    assert db_path.exists()
    assert db_path.stat().st_size > 0

    tables = operations.show_tables(file_path=str(db_path))
    assert "experiment_to_gene" in tables
    assert "experiment_run" in tables

    monkeypatch.setenv("ISPEC_DB_PATH", str(db_path))

    def override_get_session():
        with get_session() as session:
            yield session

    app.dependency_overrides[get_session_dep] = override_get_session
    client = TestClient(app)

    try:
        resp = client.get("/people/options")
        assert resp.status_code == 200
        assert len(resp.json()) == summary["people"]

        resp = client.get("/projects/options")
        assert resp.status_code == 200
        assert len(resp.json()) == summary["projects"]
    finally:
        app.dependency_overrides.clear()
