"""Exercise real SQLite connections while the legacy HTTP boundary is paused."""

from datetime import datetime
import json
import sqlite3
from types import SimpleNamespace

import pytest

from ispec.db import legacy_sync
from ispec.db.connect import get_session
from ispec.db.models import Experiment, ExperimentRun, LegacySyncState, Person, Project


@pytest.fixture(params=["projects", "people", "experiments", "runs"])
def lane(request):
    definitions = {
        "projects": (
            "iSPEC_Projects",
            Project,
            "prj_PRJRecNo",
            "prj_ModificationTS",
            "_apply_project_record",
        ),
        "people": (
            "iSPEC_People",
            Person,
            "ppl_PPLRecNo",
            "ppl_ModificationTS",
            "_apply_person_record",
        ),
        "experiments": (
            "iSPEC_Experiments",
            Experiment,
            "exp_EXPRecNo",
            "exp_ModificationTS",
            "_apply_experiment_record",
        ),
        "runs": (
            "iSPEC_ExperimentRuns",
            ExperimentRun,
            "exprun_EXPRecNo",
            "exprun_ModificationTS",
            "_apply_experiment_run_record",
        ),
    }
    table, model, pk, modified, apply_name = definitions[request.param]
    function = (
        legacy_sync.sync_legacy_experiment_runs_incremental
        if request.param == "runs"
        else getattr(legacy_sync, f"sync_legacy_{request.param}")
    )
    return SimpleNamespace(
        name=request.param,
        table=table,
        model=model,
        pk=pk,
        modified=modified,
        function=function,
        apply_name=apply_name,
    )


def seed(path, lane, *, existing_cursor=False):
    with get_session(path) as db:
        db.add(Project(id=1, prj_AddedBy="test", prj_ProjectTitle="Local"))
        if lane.name == "runs":
            db.add_all(
                Experiment(id=n, project_id=1, record_no=str(n)) for n in (1001, 1002)
            )
        if existing_cursor:
            db.add(
                LegacySyncState(
                    legacy_table=lane.table,
                    since=datetime(1900, 1, 1),
                    since_pk=None if lane.name == "runs" else 0,
                    since_pk_json="[0,0,0]" if lane.name == "runs" else None,
                )
            )


def page(lane, number, *, has_more):
    row = {lane.pk: 1000 + number, lane.modified: f"2026-01-0{number}T00:00:00Z"}
    extras = {
        "projects": {"prj_ProjectTitle": "Remote project"},
        "people": {"ppl_Name_First": "Remote", "ppl_Name_Last": "Person"},
        "experiments": {"exp_Exp_ProjectNo": 1, "exp_IDENTIFIER": "Remote experiment"},
        "runs": {
            "exprun_EXPRunNo": 1,
            "exprun_EXPSearchNo": 1,
            "exprun_MS_Instrument": "Source instrument",
        },
    }
    return {
        "items": [{**row, **extras[lane.name]}],
        "has_more": has_more,
        "next_since": row[lane.modified],
        "next_since_pk": (
            [1000 + number, 1, 1] if lane.name == "runs" else 1000 + number
        ),
    }, "csv"


def sync(path, lane, **options):
    return lane.function(
        legacy_url="http://legacy.example",
        db_file_path=str(path),
        limit=1,
        max_pages=3,
        **options,
    )


def snapshot(path, lane):
    with sqlite3.connect(path) as db:
        count = db.execute(
            f'SELECT count(*) FROM "{lane.model.__tablename__}"'
        ).fetchone()[0]
        cursor = db.execute(
            "SELECT since, since_pk, since_pk_json FROM legacy_sync_state WHERE legacy_table=?",
            (lane.table,),
        ).fetchone()
    return count, cursor


def test_forward_cursor_respects_server_ordering_without_skips_or_repeats(
    tmp_path, monkeypatch, lane
):
    path = tmp_path / "sync.db"
    seed(path, lane)
    rows = [page(lane, number, has_more=False)[0]["items"][0] for number in (1, 2)]
    # Equal timestamps require the PK tie-breaker to advance, not skip a row.
    rows[1][lane.modified] = rows[0][lane.modified]
    requests = []

    def fetch(**kwargs):
        params = kwargs["params"]
        requests.append(dict(params))
        pk_fields = (params.get("pk_fields") or lane.pk).split(",")
        ordering = params.get("order_by", ",".join([lane.modified, *pk_fields]))
        remaining = list(rows)
        if params.get("since"):
            since_pk = tuple(int(part) for part in str(params["since_pk"]).split(","))
            after = (params["since"], *since_pk)
            remaining = [
                row
                for row in rows
                if (row[lane.modified], *(row[key] for key in pk_fields)) > after
            ]
        for field in reversed(ordering.split(",")):
            remaining.sort(
                key=lambda row: row[field.lstrip("-")], reverse=field.startswith("-")
            )
        selected = remaining[: params["limit"]]
        result = {"items": selected, "has_more": len(remaining) > len(selected)}
        if selected:
            last = selected[-1]
            result["next_since"] = last[lane.modified]
            result["next_since_pk"] = (
                [last[key] for key in pk_fields]
                if lane.name == "runs"
                else last[lane.pk]
            )
        return result, "csv"

    monkeypatch.setattr(legacy_sync, "_fetch_legacy_rows_best_effort", fetch)
    summary = sync(path, lane)
    assert summary["inserted"] == 2
    assert summary["items"] == 2
    assert summary["duplicates_skipped"] == 0
    assert len(requests) == 2
    assert requests[1]["since_pk"] == ("1001,1,1" if lane.name == "runs" else 1001)
    summary = sync(path, lane)
    assert summary["items"] == 0
    assert summary["inserted"] == 0


@pytest.mark.parametrize("existing_cursor", [False, True])
@pytest.mark.parametrize("dry_run", [False, True])
def test_fetch_never_holds_writer_lock(
    tmp_path, monkeypatch, lane, existing_cursor, dry_run
):
    path = tmp_path / "sync.db"
    seed(path, lane, existing_cursor=existing_cursor)
    before = snapshot(path, lane)
    requests = []

    def fetch(**kwargs):
        requests.append(dict(kwargs["params"]))
        # A separate connection must be able to commit while HTTP is in flight.
        with sqlite3.connect(path, timeout=0.05) as other:
            other.execute(
                "UPDATE project SET prj_ProjectTitle=? WHERE id=1",
                (f"edit {len(requests)}",),
            )
        if len(requests) == 2 and not dry_run:
            assert snapshot(path, lane)[0] == before[0] + 1
        return page(lane, len(requests), has_more=len(requests) < 2)

    monkeypatch.setattr(legacy_sync, "_fetch_legacy_rows_best_effort", fetch)
    assert sync(path, lane, dry_run=dry_run)["inserted"] == 2
    assert len(requests) == 2
    assert requests[1]["since_pk"] == ("1001,1,1" if lane.name == "runs" else 1001)
    if dry_run:
        assert snapshot(path, lane) == before
    else:
        assert snapshot(path, lane)[0] == before[0] + 2


@pytest.mark.parametrize("failure", ["http", "apply"])
def test_failed_second_page_keeps_first_page_and_resume_cursor(
    tmp_path, monkeypatch, lane, failure
):
    path = tmp_path / "sync.db"
    seed(path, lane)
    before_count = snapshot(path, lane)[0]
    calls = []

    def fetch(**kwargs):
        calls.append(kwargs["params"])
        if len(calls) == 2 and failure == "http":
            raise RuntimeError("synthetic HTTP failure")
        return page(lane, len(calls), has_more=len(calls) < 2)

    real_apply = getattr(legacy_sync, lane.apply_name)

    def apply(session, **kwargs):
        result = real_apply(session, **kwargs)
        if len(calls) == 2 and failure == "apply":
            session.flush()
            raise RuntimeError("synthetic apply failure")
        return result

    monkeypatch.setattr(legacy_sync, "_fetch_legacy_rows_best_effort", fetch)
    monkeypatch.setattr(legacy_sync, lane.apply_name, apply)
    with pytest.raises(RuntimeError, match="synthetic"):
        sync(path, lane)
    count, cursor = snapshot(path, lane)
    assert count == before_count + 1
    assert cursor[1:] == ((None, "[1001,1,1]") if lane.name == "runs" else (1001, None))

    def resume(**kwargs):
        assert kwargs["params"]["since_pk"] == (
            "1001,1,1" if lane.name == "runs" else 1001
        )
        return page(lane, 2, has_more=False)

    monkeypatch.setattr(legacy_sync, "_fetch_legacy_rows_best_effort", resume)
    monkeypatch.setattr(legacy_sync, lane.apply_name, real_apply)
    assert sync(path, lane)["inserted"] == 1
    assert snapshot(path, lane)[0] == before_count + 2


@pytest.mark.parametrize("existing_cursor", [False, True])
def test_competing_cursor_change_rejects_stale_payload(
    tmp_path, monkeypatch, lane, existing_cursor
):
    path = tmp_path / "sync.db"
    seed(path, lane, existing_cursor=existing_cursor)
    before_count = snapshot(path, lane)[0]

    def fetch(**kwargs):
        with get_session(path) as other:
            cursor = other.get(LegacySyncState, lane.table)
            if cursor is None:
                cursor = LegacySyncState(legacy_table=lane.table)
                other.add(cursor)
            cursor.since = datetime(2026, 2, 1)
            cursor.since_pk = 2000 if lane.name != "runs" else None
            cursor.since_pk_json = "[2000,1,1]" if lane.name == "runs" else None
        return page(lane, 1, has_more=False)

    monkeypatch.setattr(legacy_sync, "_fetch_legacy_rows_best_effort", fetch)
    with pytest.raises(RuntimeError, match="cursor.*changed"):
        sync(path, lane)
    count, cursor = snapshot(path, lane)
    assert count == before_count
    assert cursor[0].startswith("2026-02-01")


@pytest.mark.parametrize("existing_cursor", [False, True])
@pytest.mark.parametrize("empty_response", [False, True])
def test_fetch_without_payload_preserves_stored_cursor(
    tmp_path, monkeypatch, lane, existing_cursor, empty_response
):
    path = tmp_path / "sync.db"
    seed(path, lane, existing_cursor=existing_cursor)
    before = snapshot(path, lane)

    def fetch(**kwargs):
        assert kwargs["params"]["since"].startswith("2026-01-01")
        assert kwargs["params"]["since_pk"] == (
            "500,1,1" if lane.name == "runs" else 500
        )
        if empty_response:
            return {"items": [], "has_more": False}, "csv"
        raise RuntimeError("synthetic fetch failure")

    monkeypatch.setattr(legacy_sync, "_fetch_legacy_rows_best_effort", fetch)
    options = {
        "reset_cursor": True,
        "since": "2026-01-01T00:00:00Z",
        "since_pk": [500, 1, 1] if lane.name == "runs" else 500,
    }
    if empty_response:
        assert sync(path, lane, **options)["inserted"] == 0
    else:
        with pytest.raises(RuntimeError, match="synthetic fetch failure"):
            sync(path, lane, **options)
    assert snapshot(path, lane) == before


def test_run_reconciliation_sees_local_edit_made_during_fetch(tmp_path, monkeypatch):
    path = tmp_path / "sync.db"
    with get_session(path) as db:
        db.add(Experiment(id=1001, record_no="1001"))
    row = {
        "exprun_EXPRecNo": 1001,
        "exprun_EXPRunNo": 1,
        "exprun_EXPSearchNo": 1,
        "exprun_ModificationTS": "2026-01-01T00:00:00Z",
        "exprun_MS_Instrument": "Original",
    }
    monkeypatch.setattr(
        legacy_sync,
        "_fetch_legacy_rows_best_effort",
        lambda **_: ({"items": [dict(row)], "has_more": False}, "csv"),
    )
    legacy_sync.sync_legacy_experiment_runs_incremental(
        legacy_url="http://legacy.example", db_file_path=str(path)
    )

    def fetch(**kwargs):
        with sqlite3.connect(path, timeout=0.05) as other:
            other.execute(
                "UPDATE experiment_run SET ms_instrument='Local correction' WHERE experiment_id=1001"
            )
        return {
            "items": [{**row, "exprun_MS_Instrument": "Remote correction"}],
            "has_more": False,
        }, "csv"

    monkeypatch.setattr(legacy_sync, "_fetch_legacy_rows_best_effort", fetch)
    result = legacy_sync.sync_legacy_experiment_runs_incremental(
        legacy_url="http://legacy.example", db_file_path=str(path)
    )
    assert result["conflicted"] == 1
    with get_session(path) as db:
        run = db.query(ExperimentRun).one()
        assert run.ms_instrument == "Local correction"
        conflict = json.loads(run.legacy_source_conflicts_json)["values"][
            "ms_instrument"
        ]
        assert conflict["incoming_source"] == "Remote correction"
