"""Synthetic staff API, manual edit and Phase 1 archive compatibility checks."""

from contextlib import closing
from types import SimpleNamespace
import sqlite3

import pytest

from ispec.api.security import get_current_user
from ispec.db.connect import get_session
from ispec.db.models import UserRole
from ispec.instruments import dashboard
from scripts.instrument_registry import apply
from tests.unit.api.test_instruments import setup  # noqa: F401

pytestmark = pytest.mark.testclient
URL = "/api/instruments"


@pytest.fixture
def staff(setup, tmp_path, monkeypatch):  # noqa: F811
    client, app, _ = setup
    monkeypatch.setenv("ISPEC_INSTRUMENT_ARCHIVES", str(tmp_path))
    monkeypatch.setenv("ISPEC_INSTRUMENT_MOUNT_ROOT", str(tmp_path / "mounts"))
    app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(
        role=UserRole.editor, must_change_password=False
    )
    root = str(tmp_path / "mounts/lab/data")
    entry = dict(
        id="scope",
        plugin="thermo",
        mode="pull",
        display_name="Scope",
        roots={"acquisition": root},
        connections={"acquisition": dict(host="lab", share="data", mount_path=root)},
    )
    return client, app, entry


def test_registry_access_validation_and_empty(staff, monkeypatch):
    c, app, entry = staff
    assert (
        c.post(URL, json={**entry, "password": "not-a-real-password"}).status_code
        == 422
    )
    assert c.post(URL, json={**entry, "id": "../bad"}).status_code == 422
    assert (
        c.post(URL, json={**entry, "roots": {"acquisition": "/etc"}}).status_code == 422
    )
    assert c.post(URL, json=entry).status_code == 201
    assert c.post(URL, json=entry).status_code == 409
    assert c.get(URL + "/absent/status").status_code == 404
    monkeypatch.setattr(dashboard, "mounted", lambda *_: False)
    status = c.get(URL + "/scope/status")
    assert status.status_code == 200 and status.json()["summary"] is None
    assert status.json()["summary_error"]["code"] == "source_not_mounted"
    assert c.get(URL + "/scope/runs").json()["items"] == []
    assert (
        c.patch(
            URL + "/scope", json={"display_name": "Renamed", "revision": 1}
        ).status_code
        == 200
    )
    assert (
        c.patch(
            URL + "/scope", json={"display_name": "Stale", "revision": 1}
        ).status_code
        == 409
    )
    assert c.patch(URL + "/scope", json={"arbitrary": "input"}).status_code == 422
    assert (
        c.patch(
            URL + "/scope",
            json={"enabled": False},
            headers={"Origin": "https://elsewhere.invalid"},
        ).status_code
        == 403
    )
    for actor, code in [(None, 401), (UserRole.client, 403), (UserRole.viewer, 403)]:
        app.dependency_overrides[get_current_user] = (
            lambda: SimpleNamespace(role=actor, must_change_password=False)
            if actor
            else None
        )
        assert c.get(URL).status_code == c.get(URL + "/scope/runs").status_code == code
        assert (
            c.patch(URL + "/scope", json={}).status_code
            == c.post(URL, json=entry).status_code
            == code
        )


def test_manual_edit_and_retirement(staff, tmp_path):
    c, _, entry = staff
    saved = c.post(URL, json=entry).json()
    saved["connections"]["acquisition"]["host"] = "new-host"
    with get_session(tmp_path / "core.sqlite") as db:
        apply(db, [saved])
    assert (
        c.get(URL).json()["instruments"][0]["connections"]["acquisition"]["host"]
        == "new-host"
    )
    with pytest.raises(ValueError, match="stale"):
        with get_session(tmp_path / "core.sqlite") as db:
            apply(db, [saved])


def test_live_archive_projection_and_keyset_snapshot(staff, tmp_path, monkeypatch):
    pytest.importorskip("ispec_observer.pilot_store", reason="Requires the pinned Phase 1 writer fixture")
    from ispec_observer.pilot_store import Archive, Fact, Scan

    c, _, entry = staff
    c.post(URL, json=entry).raise_for_status()
    path = tmp_path / "scope.1.sqlite"
    t = "2026-01-02T00:00:00.000000Z"
    with closing(sqlite3.connect(path)) as db:
        a = Archive(db, "scope", "thermo")
        a.apply(
            Scan(
                coverage={"qc": "readable"},
                facts=[
                    Fact("qc_raw", "qc", name, t, {"bytes": 10})
                    for name in ("first.raw", "second.raw")
                ]
                + [
                    Fact(
                        "tng_record", "tng", "DDI", None, {"source_time_local": "12:30"}
                    )
                ],
            ),
            generation=0,
            recorded_at=t,
            heartbeat_before=t,
        )
        monkeypatch.setattr(dashboard, "mounted", lambda *_: False)
        before = path.read_bytes()
        status = c.get(URL + "/scope/status").json()
        assert status["summary_error"]["code"] == "source_not_mounted"
        s = status["summary"]
        assert (
            s["reported_state"] is None
            and s["latest_source_event"]["observed_at"] is None
        )
        assert s["latest_source_event"]["source_time_text"] == "12:30"
        page = c.get(URL + "/scope/runs?limit=1").json()
        assert page["items"][0]["name"] == "second.raw" and page["next_cursor"]
        assert path.read_bytes() == before
        a.apply(
            Scan(
                coverage={"qc": "readable"},
                facts=[Fact("qc_raw", "qc", "third.raw", t, {"bytes": 5})],
            ),
            generation=1,
            recorded_at=t,
            heartbeat_before=t,
        )
        page = c.get(
            URL + "/scope/runs", params=dict(limit=1, cursor=page["next_cursor"])
        ).json()
        assert page["items"][0]["name"] == "first.raw" and not page["next_cursor"]
        assert c.get(URL + "/scope/runs?cursor=bad").status_code == 422
        db.execute("update stream set version=99")
        db.commit()
        assert (
            c.get(URL).json()["instruments"][0]["summary_error"]["code"]
            == "archive_unavailable"
        )
        assert c.get(URL + "/scope/runs").status_code == 503
    c.patch(URL + "/scope", json={"collector_id": "other"}).raise_for_status()
    assert path.exists() and c.get(URL + "/scope/status").json()["summary"] is None
