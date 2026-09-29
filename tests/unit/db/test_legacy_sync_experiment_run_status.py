from __future__ import annotations

import json
from datetime import datetime

import pytest

from ispec.db.connect import get_session
from ispec.db.models import Experiment, ExperimentRun, LegacySyncState, Project


def _seed_experiments(db_path, *experiment_ids: int) -> None:
    with get_session(file_path=str(db_path)) as session:
        session.add(Project(id=1494, prj_AddedBy="user", prj_ProjectTitle="P"))
        session.add_all(
            Experiment(id=exp_id, project_id=1494, record_no=str(exp_id))
            for exp_id in experiment_ids
        )


def test_targeted_run_sync_maps_legacy_processing_metadata(tmp_path, monkeypatch):
    db_path = tmp_path / "sync.db"
    _seed_experiments(db_path, 57454)
    payload = {
        "items": [
            {
                "exprun_EXPRecNo": 57454,
                "exprun_EXPRunNo": 1,
                "exprun_EXPSearchNo": 2,
                "exprun_ModificationTS": "2025-12-06 04:05:06",
                "exprun_Grouper_StartFLAG": 1,
                "exprun_Grouper_EndFLAG": 1,
                "exprun_Grouper_FailedFLAG": 0,
                "exprun_Grouper_Version": "1.0.3",
                "exprun_Grouper_FilterStamp": "SRA-30-20-10",
                "exprun_Grouper_RefDatabase": "RefSeq_9823",
                "exprun_InputFileName": "57454_1_2_psms_all.txt",
                "exprun_PSMCount_matched": 1234,
                "exprun_PSMCount_unmatched": 56,
                "exprun_Import_StartFLAG": 1,
                "exprun_Import_EndFLAG": 1,
                "exprun_Import_FixFLAG": 0,
                "exprun_ImportTS": "2025-12-07 08:09:10",
                "exprun_ImportErrorLog": "",
            }
        ],
        "has_more": False,
    }
    monkeypatch.setattr(
        "ispec.db.legacy_sync._fetch_legacy_rows_best_effort",
        lambda **kwargs: (payload, "csv"),
    )

    from ispec.db.legacy_sync import sync_legacy_experiment_runs

    summary = sync_legacy_experiment_runs(
        legacy_url="http://legacy.example",
        db_file_path=str(db_path),
        experiment_id=57454,
    )

    assert summary["inserted"] == 1
    with get_session(file_path=str(db_path)) as session:
        run = session.query(ExperimentRun).one()
        assert run.legacy_grouper_started is True
        assert run.legacy_grouper_completed is True
        assert run.legacy_grouper_failed is False
        assert run.legacy_grouper_version == "1.0.3"
        assert run.legacy_grouper_filter_stamp == "SRA-30-20-10"
        assert run.legacy_grouper_ref_database == "RefSeq_9823"
        assert run.legacy_grouper_input_file == "57454_1_2_psms_all.txt"
        assert run.legacy_grouper_matched_psm_count == 1234
        assert run.legacy_grouper_unmatched_psm_count == 56
        assert run.legacy_e2g_import_started is True
        assert run.legacy_e2g_import_completed is True
        assert run.legacy_e2g_import_fix_requested is False
        assert run.legacy_e2g_imported_at == datetime(2025, 12, 7, 8, 9, 10)
        assert run.legacy_e2g_import_error is None
        assert run.ExperimentRun_LegacyImportTS != run.legacy_e2g_imported_at
        assert session.query(LegacySyncState).count() == 0


def test_incremental_run_sync_uses_and_persists_composite_cursor(tmp_path, monkeypatch):
    db_path = tmp_path / "sync.db"
    _seed_experiments(db_path, 57454, 57455)
    pages = [
        {
            "items": [
                {
                    "exprun_EXPRecNo": 57454,
                    "exprun_EXPRunNo": 1,
                    "exprun_EXPSearchNo": 2,
                    "exprun_ModificationTS": "2025-12-06 04:05:06",
                }
            ],
            "has_more": True,
            "next_since": "2025-12-06T04:05:06Z",
            "next_since_pk": [57454, 1, 2],
        },
        {
            "items": [
                {
                    "exprun_EXPRecNo": 57455,
                    "exprun_EXPRunNo": 2,
                    "exprun_EXPSearchNo": 3,
                    "exprun_ModificationTS": "2025-12-06 04:05:06",
                }
            ],
            "has_more": False,
            "next_since": "2025-12-06T04:05:06Z",
            "next_since_pk": [57455, 2, 3],
        },
    ]
    captured_params: list[dict[str, object]] = []

    def fake_fetch(**kwargs):
        captured_params.append(dict(kwargs["params"]))
        return pages.pop(0), "csv"

    monkeypatch.setattr(
        "ispec.db.legacy_sync._fetch_legacy_rows_best_effort", fake_fetch
    )
    from ispec.db.legacy_sync import sync_legacy_experiment_runs_incremental

    summary = sync_legacy_experiment_runs_incremental(
        legacy_url="http://legacy.example",
        db_file_path=str(db_path),
        limit=1,
    )

    assert summary["pages"] == 2
    assert summary["inserted"] == 2
    assert captured_params[0]["pk_fields"] == (
        "exprun_EXPRecNo,exprun_EXPRunNo,exprun_EXPSearchNo"
    )
    assert captured_params[0]["since"] == "1900-01-01T00:00:00Z"
    assert captured_params[0]["since_pk"] == "0,0,0"
    assert captured_params[1]["since_pk"] == "57454,1,2"

    with get_session(file_path=str(db_path)) as session:
        state = session.get(LegacySyncState, "iSPEC_ExperimentRuns")
        assert state is not None
        assert state.since == datetime(2025, 12, 6, 4, 5, 6)
        assert json.loads(state.since_pk_json) == [57455, 2, 3]
        assert state.since_pk is None


def test_incremental_run_sync_dry_run_does_not_advance_cursor(tmp_path, monkeypatch):
    db_path = tmp_path / "sync.db"
    payload = {
        "items": [
            {
                "exprun_EXPRecNo": 57454,
                "exprun_EXPRunNo": 1,
                "exprun_EXPSearchNo": 2,
                "exprun_ModificationTS": "2025-12-06 04:05:06",
            }
        ],
        "has_more": False,
        "next_since": "2025-12-06T04:05:06Z",
        "next_since_pk": [57454, 1, 2],
    }
    monkeypatch.setattr(
        "ispec.db.legacy_sync._fetch_legacy_rows_best_effort",
        lambda **kwargs: (payload, "csv"),
    )
    from ispec.db.legacy_sync import sync_legacy_experiment_runs_incremental

    summary = sync_legacy_experiment_runs_incremental(
        legacy_url="http://legacy.example",
        db_file_path=str(db_path),
        dry_run=True,
    )

    assert summary["inserted"] == 1
    with get_session(file_path=str(db_path)) as session:
        assert session.query(ExperimentRun).count() == 0
        assert session.query(LegacySyncState).count() == 0


def test_status_refresh_does_not_overwrite_locally_edited_metadata(
    tmp_path, monkeypatch
):
    db_path = tmp_path / "sync.db"
    _seed_experiments(db_path, 57454)
    with get_session(file_path=str(db_path)) as session:
        session.add(
            ExperimentRun(
                experiment_id=57454,
                run_no=1,
                search_no=2,
                label="0",
                ms_instrument="locally corrected instrument",
                ExperimentRun_LegacyImportTS=datetime(2026, 1, 1),
                ExperimentRun_ModificationTS=datetime(2026, 2, 1),
            )
        )
    payload = {
        "items": [
            {
                "exprun_EXPRecNo": 57454,
                "exprun_EXPRunNo": 1,
                "exprun_EXPSearchNo": 2,
                "exprun_ModificationTS": "2026-03-01 00:00:00",
                "exprun_MS_Instrument": "legacy instrument",
                "exprun_Grouper_EndFLAG": 1,
                "exprun_Import_EndFLAG": 1,
            }
        ],
        "has_more": False,
    }
    monkeypatch.setattr(
        "ispec.db.legacy_sync._fetch_legacy_rows_best_effort",
        lambda **kwargs: (payload, "csv"),
    )
    from ispec.db.legacy_sync import sync_legacy_experiment_runs

    summary = sync_legacy_experiment_runs(
        legacy_url="http://legacy.example",
        db_file_path=str(db_path),
        experiment_id=57454,
        backfill_missing=True,
    )

    assert summary["conflicted"] == 1
    with get_session(file_path=str(db_path)) as session:
        run = session.query(ExperimentRun).one()
        assert run.ms_instrument == "locally corrected instrument"
        assert run.legacy_grouper_completed is True
        assert run.legacy_e2g_import_completed is True
        assert run.ExperimentRun_LegacyModificationTS == datetime(2026, 3, 1)
        conflicts = json.loads(run.legacy_source_conflicts_json)["values"]
        assert conflicts["ms_instrument"]["reason"] == "unknown_baseline"


@pytest.mark.parametrize("incremental", [False, True])
def test_repeated_sync_three_way_reconciliation(tmp_path, monkeypatch, incremental):
    from ispec.db.legacy_sync import (
        sync_legacy_experiment_runs,
        sync_legacy_experiment_runs_incremental,
    )

    db_path = tmp_path / "reconcile.db"
    _seed_experiments(db_path, 57454)
    row = {
        "exprun_EXPRecNo": 57454,
        "exprun_EXPRunNo": 1,
        "exprun_EXPSearchNo": 2,
        "exprun_ModificationTS": "2026-01-01 00:00:00",
        "exprun_MS_Instrument": "initial instrument",
        "exprun_Search_RefDatabase": "initial database",
        "exprun_TaxonID": 9606,
        "exprun_Grouper_EndFLAG": 0,
    }
    monkeypatch.setattr(
        "ispec.db.legacy_sync._fetch_legacy_rows_best_effort",
        lambda **_: ({"items": [dict(row)], "has_more": False}, "csv"),
    )

    def sync(**kwargs):
        options = {
            "legacy_url": "http://legacy.example",
            "db_file_path": str(db_path),
            **kwargs,
        }
        if incremental:
            return sync_legacy_experiment_runs_incremental(**options)
        return sync_legacy_experiment_runs(experiment_id=57454, **options)

    assert sync()["inserted"] == 1
    with get_session(file_path=str(db_path)) as db:
        run = db.query(ExperimentRun).one()
        run.ms_instrument = "local correction"

    row["exprun_Search_RefDatabase"] = "new source database"
    row["exprun_Grouper_EndFLAG"] = 1
    row["exprun_ModificationTS"] = "2026-02-01 00:00:00"
    for _ in range(3):
        assert sync()["conflicted"] == 0
        with get_session(file_path=str(db_path)) as db:
            run = db.query(ExperimentRun).one()
            assert run.ms_instrument == "local correction"
            assert run.ref_database == "new source database"
            assert run.legacy_grouper_completed is True
            assert (
                json.loads(run.legacy_source_baseline_json)["values"]["ms_instrument"]
                == "initial instrument"
            )

    row["exprun_MS_Instrument"] = "different source correction"
    row["exprun_TaxonID"] = 9823
    row["exprun_ModificationTS"] = "2026-03-01 00:00:00"
    for _ in range(2):
        assert sync()["conflicted"] == 1
        with get_session(file_path=str(db_path)) as db:
            run = db.query(ExperimentRun).one()
            assert run.ms_instrument == "local correction"
            assert run.taxon_id == 9823
            conflict = json.loads(run.legacy_source_conflicts_json)["values"][
                "ms_instrument"
            ]
            assert conflict["reason"] == "divergent_changes"
            assert conflict["previous_source"] == "initial instrument"
            assert conflict["local"] == "local correction"
            assert conflict["incoming_source"] == "different source correction"

    row["exprun_MS_Instrument"] = "local correction"
    row["exprun_ModificationTS"] = "2026-04-01 00:00:00"
    assert sync()["conflicted"] == 0
    with get_session(file_path=str(db_path)) as db:
        run = db.query(ExperimentRun).one()
        assert not json.loads(run.legacy_source_conflicts_json)["values"]
        assert (
            json.loads(run.legacy_source_baseline_json)["values"]["ms_instrument"]
            == "local correction"
        )

    row["exprun_MS_Instrument"] = "next source revision"
    row["exprun_ModificationTS"] = "2026-05-01 00:00:00"
    assert sync()["conflicted"] == 0
    with get_session(file_path=str(db_path)) as db:
        run = db.query(ExperimentRun).one()
        assert run.ms_instrument == "next source revision"
        before = (
            run.ms_instrument,
            run.legacy_source_baseline_json,
            run.legacy_source_conflicts_json,
            run.ExperimentRun_LegacyImportTS,
        )

    row["exprun_MS_Instrument"] = "dry run only"
    assert sync(dry_run=True)["conflicted"] == 0
    with get_session(file_path=str(db_path)) as db:
        run = db.query(ExperimentRun).one()
        assert (
            run.ms_instrument,
            run.legacy_source_baseline_json,
            run.legacy_source_conflicts_json,
            run.ExperimentRun_LegacyImportTS,
        ) == before


def test_unbaselined_rows_stay_conservative_across_syncs(tmp_path, monkeypatch):
    from ispec.db.legacy_sync import sync_legacy_experiment_runs

    db_path = tmp_path / "old.db"
    _seed_experiments(db_path, 57454)
    with get_session(file_path=str(db_path)) as db:
        db.add(
            ExperimentRun(
                experiment_id=57454,
                run_no=1,
                search_no=2,
                label="0",
                ms_instrument="local correction",
                ref_database="agreed database",
                ExperimentRun_LegacyImportTS=datetime(2026, 9, 1),
                ExperimentRun_ModificationTS=datetime(2026, 1, 1),
            )
        )
    row = {
        "exprun_EXPRecNo": 57454,
        "exprun_EXPRunNo": 1,
        "exprun_EXPSearchNo": 2,
        "exprun_MS_Instrument": "source instrument",
        "exprun_Search_RefDatabase": "agreed database",
    }
    monkeypatch.setattr(
        "ispec.db.legacy_sync._fetch_legacy_rows_best_effort",
        lambda **_: ({"items": [dict(row)]}, "csv"),
    )
    for index in range(3):
        if index:
            row["exprun_Search_RefDatabase"] = "new database"
        summary = sync_legacy_experiment_runs(
            legacy_url="http://legacy.example",
            db_file_path=str(db_path),
            experiment_id=57454,
        )
        assert summary["conflicted"] == 1
        with get_session(file_path=str(db_path)) as db:
            run = db.query(ExperimentRun).one()
            assert run.ms_instrument == "local correction"
            assert run.ref_database == ("new database" if index else "agreed database")
            assert (
                json.loads(run.legacy_source_conflicts_json)["values"]["ms_instrument"][
                    "reason"
                ]
                == "unknown_baseline"
            )


def test_partial_payload_retains_fields_but_explicit_null_clears_them(
    tmp_path, monkeypatch
):
    from ispec.db.legacy_sync import sync_legacy_experiment_runs

    db_path = tmp_path / "partial.db"
    _seed_experiments(db_path, 57454)
    row = {
        "exprun_EXPRecNo": 57454,
        "exprun_EXPRunNo": 1,
        "exprun_EXPSearchNo": 2,
        "exprun_LabelType": "none",
        "exprun_MS_Instrument": "instrument",
    }
    monkeypatch.setattr(
        "ispec.db.legacy_sync._fetch_legacy_rows_best_effort",
        lambda **_: ({"items": [dict(row)]}, "csv"),
    )

    def sync():
        return sync_legacy_experiment_runs(
            legacy_url="http://legacy.example",
            db_file_path=str(db_path),
            experiment_id=57454,
        )

    assert sync()["inserted"] == 1
    del row["exprun_LabelType"]
    del row["exprun_MS_Instrument"]
    assert sync()["conflicted"] == 0
    with get_session(file_path=str(db_path)) as db:
        run = db.query(ExperimentRun).one()
        assert run.label_type == "none"
        assert run.ms_instrument == "instrument"
        assert (
            json.loads(run.legacy_source_baseline_json)["values"]["label_type"]
            == "none"
        )

    row["exprun_LabelType"] = None
    row["exprun_MS_Instrument"] = None
    assert sync()["conflicted"] == 0
    with get_session(file_path=str(db_path)) as db:
        run = db.query(ExperimentRun).one()
        assert run.label_type is None
        assert run.ms_instrument is None
        assert (
            json.loads(run.legacy_source_baseline_json)["values"]["label_type"] is None
        )
