from __future__ import annotations

import json
from datetime import datetime

from ispec.db.connect import get_session
from ispec.db.models import Experiment, ExperimentRun, LegacySyncState, Project


def test_sync_legacy_experiments_single_id_inserts_experiment(tmp_path, monkeypatch):
    db_path = tmp_path / "sync.db"

    with get_session(file_path=str(db_path)) as session:
        session.add(Project(id=1494, prj_AddedBy="user", prj_ProjectTitle="P"))

    payload = {
        "ok": True,
        "table": "iSPEC_Experiments",
        "items": [
            {
                "exp_EXPRecNo": 57454.0,
                "exp_Exp_ProjectNo": 1494.0,
                "exp_IDENTIFIER": "TBK1 profiling",
                "exp_ExpType": "Affinity-XL",
                "exp_EXPLabelFLAG": "1",
                "exp_Extract_LysisBuffer": "ABC",
                "exp_Extract_DTT": "No",
                "exp_Extract_IAA": "Yes",
                "exp_Exp_Description": "Some description",
                "exp_CreationTS": "2025-09-29 11:14:33",
                "exp_ModificationTS": "2025-12-05 12:16:37",
            }
        ],
        "has_more": False,
    }

    class DummyResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return payload

    captured: dict[str, object] = {}

    def fake_get(url, params=None, headers=None, auth=None, timeout=None):
        captured["params"] = dict(params or {})
        return DummyResponse()

    from ispec.db import legacy_sync

    monkeypatch.setattr(legacy_sync.requests, "get", fake_get)

    summary = legacy_sync.sync_legacy_experiments(
        legacy_url="http://legacy.example",
        db_file_path=str(db_path),
        experiment_id=57454,
        dry_run=False,
    )

    assert summary["inserted"] == 1
    assert summary["updated"] == 0
    assert summary["conflicted"] == 0

    assert captured["params"]["pk_field"] == "exp_EXPRecNo"
    assert captured["params"]["id"] == 57454

    with get_session(file_path=str(db_path)) as session:
        exp = session.get(Experiment, 57454)
        assert exp is not None
        assert exp.project_id == 1494
        assert exp.record_no == "57454"
        assert exp.exp_Name == "TBK1 profiling"
        assert exp.exp_Type == "Affinity-XL"
        assert exp.exp_LabelFLAG == 1
        assert exp.exp_Lysis == "ABC"
        assert exp.exp_DTT is False
        assert exp.exp_IAA is True
        assert exp.exp_Description == "Some description"
        assert isinstance(exp.Experiment_CreationTS, datetime)
        assert isinstance(exp.Experiment_ModificationTS, datetime)

        # Single-id sync should not advance the incremental cursor.
        assert session.query(LegacySyncState).count() == 0


def test_sync_legacy_experiments_creates_placeholder_project(tmp_path, monkeypatch):
    db_path = tmp_path / "sync.db"

    payload = {
        "ok": True,
        "table": "iSPEC_Experiments",
        "items": [
            {
                "exp_EXPRecNo": 1,
                "exp_Exp_ProjectNo": 42,
                "exp_IDENTIFIER": "X",
                "exp_ModificationTS": "2025-12-05 12:16:37",
            }
        ],
        "has_more": False,
    }

    class DummyResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return payload

    def fake_get(url, params=None, headers=None, auth=None, timeout=None):
        return DummyResponse()

    from ispec.db import legacy_sync

    monkeypatch.setattr(legacy_sync.requests, "get", fake_get)

    summary = legacy_sync.sync_legacy_experiments(
        legacy_url="http://legacy.example",
        db_file_path=str(db_path),
        experiment_id=1,
        dry_run=False,
    )

    assert summary["inserted"] == 1

    with get_session(file_path=str(db_path)) as session:
        project = session.get(Project, 42)
        assert project is not None
        assert project.prj_AddedBy == "legacy_import"
        assert project.prj_PRJ_DisplayID == "MSPC000042"
        exp = session.get(Experiment, 1)
        assert exp is not None
        assert exp.project_id == 42


def test_sync_legacy_experiments_backfills_missing_fields_on_conflict(tmp_path, monkeypatch):
    db_path = tmp_path / "sync.db"
    mapping_path = tmp_path / "mapping.json"

    mapping_path.write_text(
        json.dumps(
            {
                "tables": {
                    "iSPEC_Experiments": {
                        "pk": {"legacy": "exp_EXPRecNo", "local": "id"},
                        "created_ts": "exp_CreationTS",
                        "modified_ts": "exp_ModificationTS",
                        "field_map": {
                            "exp_Exp_ProjectNo": "project_id",
                            "exp_IDENTIFIER": "exp_Name",
                            "exp_Exp_Description": "exp_Description",
                        },
                    }
                }
            }
        )
    )

    with get_session(file_path=str(db_path)) as session:
        session.add(Project(id=1494, prj_AddedBy="user", prj_ProjectTitle="P"))
        session.add(
            Experiment(
                id=57454,
                project_id=1494,
                record_no="57454",
                exp_Name="Local Name",
                exp_Description=None,
                Experiment_LegacyImportTS=datetime(2026, 1, 1, 0, 0, 0),
                Experiment_ModificationTS=datetime(2026, 1, 2, 0, 0, 0),
            )
        )

    payload = {
        "ok": True,
        "table": "iSPEC_Experiments",
        "items": [
            {
                "exp_EXPRecNo": 57454,
                "exp_Exp_ProjectNo": 1494,
                "exp_IDENTIFIER": "Legacy Name",
                "exp_Exp_Description": "Legacy description",
                "exp_CreationTS": "2025-10-01 01:02:03",
                "exp_ModificationTS": "2025-12-05 12:16:37",
            }
        ],
        "has_more": False,
    }

    class DummyResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return payload

    def fake_get(url, params=None, headers=None, auth=None, timeout=None):
        return DummyResponse()

    from ispec.db import legacy_sync

    monkeypatch.setattr(legacy_sync.requests, "get", fake_get)

    summary = legacy_sync.sync_legacy_experiments(
        legacy_url="http://legacy.example",
        mapping_path=str(mapping_path),
        db_file_path=str(db_path),
        experiment_id=57454,
        backfill_missing=True,
        dry_run=False,
    )

    assert summary["backfilled"] == 1
    assert summary["updated"] == 0
    assert summary["conflicted"] == 0

    with get_session(file_path=str(db_path)) as session:
        exp = session.get(Experiment, 57454)
        assert exp is not None
        assert exp.exp_Name == "Local Name"
        assert exp.exp_Description == "Legacy description"


def test_sync_legacy_experiment_runs_single_id_inserts_run(tmp_path, monkeypatch):
    db_path = tmp_path / "sync.db"

    with get_session(file_path=str(db_path)) as session:
        session.add(Project(id=1494, prj_AddedBy="user", prj_ProjectTitle="P"))
        session.add(Experiment(id=57454, project_id=1494, record_no="57454"))

    payload = {
        "ok": True,
        "table": "iSPEC_ExperimentRuns",
        "items": [
            {
                "exprun_EXPRecNo": 57454.0,
                "exprun_EXPRunNo": 1,
                "exprun_EXPSearchNo": 2,
                "exprun_CreationTS": "2025-10-01 01:02:03",
                "exprun_ModificationTS": "2025-12-06 04:05:06",
            }
        ],
        "has_more": False,
    }

    class DummyResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return payload

    def fake_get(url, params=None, headers=None, auth=None, timeout=None):
        return DummyResponse()

    from ispec.db import legacy_sync

    monkeypatch.setattr(legacy_sync.requests, "get", fake_get)

    summary = legacy_sync.sync_legacy_experiment_runs(
        legacy_url="http://legacy.example",
        db_file_path=str(db_path),
        experiment_id=57454,
        dry_run=False,
    )

    assert summary["inserted"] == 1
    assert summary["updated"] == 0
    assert summary["conflicted"] == 0

    with get_session(file_path=str(db_path)) as session:
        run = (
            session.query(ExperimentRun)
            .filter(
                ExperimentRun.experiment_id == 57454,
                ExperimentRun.run_no == 1,
                ExperimentRun.search_no == 2,
            )
            .one_or_none()
        )
        assert run is not None
        assert isinstance(run.ExperimentRun_CreationTS, datetime)
        assert isinstance(run.ExperimentRun_ModificationTS, datetime)

        # Single-id sync should not advance the incremental cursor.
        assert session.query(LegacySyncState).count() == 0
