from __future__ import annotations

import json
from datetime import datetime

from ispec.db.connect import get_session
from ispec.db.models import LegacySyncState, Person


def test_sync_legacy_people_single_id_inserts_person(tmp_path, monkeypatch):
    db_path = tmp_path / "sync.db"
    mapping_path = tmp_path / "mapping.json"

    mapping_path.write_text(
        json.dumps(
            {
                "tables": {
                    "iSPEC_People": {
                        "pk": {"legacy": "ppl_PPLRecNo", "local": "id"},
                        "created_ts": "ppl_CreationTS",
                        "modified_ts": "ppl_ModificationTS",
                        "field_map": {
                            "ppl_AddedBy": "ppl_AddedBy",
                            "ppl_Name_First": "ppl_Name_First",
                            "ppl_Name_Last": "ppl_Name_Last",
                            "ppl_Email": "ppl_Email",
                        },
                    }
                }
            }
        )
    )

    payload = {
        "ok": True,
        "table": "iSPEC_People",
        "items": [
            {
                "ppl_PPLRecNo": 12,
                "ppl_AddedBy": None,
                "ppl_Name_First": "Alice",
                "ppl_Name_Last": "Smith",
                "ppl_Email": "alice@example.com",
                "ppl_CreationTS": "2025-09-29 11:14:33",
                "ppl_ModificationTS": "2025-12-05 12:16:37",
            }
        ],
        "has_more": False,
    }

    captured: dict[str, object] = {}

    class DummyResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return payload

    def fake_get(url, params=None, headers=None, auth=None, timeout=None):
        captured["url"] = url
        captured["params"] = dict(params or {})
        return DummyResponse()

    from ispec.db import legacy_sync

    monkeypatch.setattr(legacy_sync.requests, "get", fake_get)

    summary = legacy_sync.sync_legacy_people(
        legacy_url="http://legacy.example",
        mapping_path=str(mapping_path),
        db_file_path=str(db_path),
        person_id=12,
        dry_run=False,
    )

    assert summary["inserted"] == 1
    assert summary["updated"] == 0
    assert summary["conflicted"] == 0

    assert captured["url"] == "http://legacy.example/api/v2/legacy/tables/iSPEC_People/rows"
    assert captured["params"]["id"] == 12

    with get_session(file_path=str(db_path)) as session:
        person = session.get(Person, 12)
        assert person is not None
        assert person.ppl_Name_First == "Alice"
        assert person.ppl_Name_Last == "Smith"
        assert person.ppl_Email == "alice@example.com"
        assert person.ppl_AddedBy == "legacy_import"
        assert isinstance(person.ppl_CreationTS, datetime)
        assert isinstance(person.ppl_ModificationTS, datetime)

        # Single-id sync should not advance the incremental cursor.
        assert session.query(LegacySyncState).count() == 0


def test_sync_legacy_people_backfills_missing_fields_on_conflict(tmp_path, monkeypatch):
    db_path = tmp_path / "sync.db"
    mapping_path = tmp_path / "mapping.json"

    mapping_path.write_text(
        json.dumps(
            {
                "tables": {
                    "iSPEC_People": {
                        "pk": {"legacy": "ppl_PPLRecNo", "local": "id"},
                        "created_ts": "ppl_CreationTS",
                        "modified_ts": "ppl_ModificationTS",
                        "field_map": {
                            "ppl_AddedBy": "ppl_AddedBy",
                            "ppl_Name_First": "ppl_Name_First",
                            "ppl_Name_Last": "ppl_Name_Last",
                            "ppl_Email": "ppl_Email",
                        },
                    }
                }
            }
        )
    )

    with get_session(file_path=str(db_path)) as session:
        session.add(
            Person(
                id=12,
                ppl_AddedBy="user",
                ppl_LegacyImportTS=datetime(2026, 1, 1, 0, 0, 0),
                ppl_ModificationTS=datetime(2026, 1, 2, 0, 0, 0),
                ppl_Name_First="Local",
                ppl_Name_Last="User",
                ppl_Email=None,
            )
        )

    payload = {
        "ok": True,
        "table": "iSPEC_People",
        "items": [
            {
                "ppl_PPLRecNo": 12,
                "ppl_AddedBy": "legacy-user",
                "ppl_Name_First": "Legacy",
                "ppl_Name_Last": "Name",
                "ppl_Email": "legacy@example.com",
                "ppl_CreationTS": "2025-10-01 01:02:03",
                "ppl_ModificationTS": "2025-12-05 12:16:37",
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

    summary = legacy_sync.sync_legacy_people(
        legacy_url="http://legacy.example",
        mapping_path=str(mapping_path),
        db_file_path=str(db_path),
        person_id=12,
        backfill_missing=True,
        dry_run=False,
    )

    assert summary["backfilled"] == 1
    assert summary["updated"] == 0
    assert summary["conflicted"] == 0

    with get_session(file_path=str(db_path)) as session:
        person = session.get(Person, 12)
        assert person is not None
        assert person.ppl_Name_First == "Local"
        assert person.ppl_Name_Last == "User"
        assert person.ppl_Email == "legacy@example.com"


def test_sync_legacy_people_persists_incremental_cursor(tmp_path, monkeypatch):
    db_path = tmp_path / "sync.db"
    mapping_path = tmp_path / "mapping.json"

    mapping_path.write_text(
        json.dumps(
            {
                "tables": {
                    "iSPEC_People": {
                        "pk": {"legacy": "ppl_PPLRecNo", "local": "id"},
                        "created_ts": "ppl_CreationTS",
                        "modified_ts": "ppl_ModificationTS",
                        "field_map": {
                            "ppl_AddedBy": "ppl_AddedBy",
                            "ppl_Name_First": "ppl_Name_First",
                            "ppl_Name_Last": "ppl_Name_Last",
                        },
                    }
                }
            }
        )
    )

    pages = [
        {
            "ok": True,
            "table": "iSPEC_People",
            "items": [
                {
                    "ppl_PPLRecNo": 1,
                    "ppl_AddedBy": "legacy",
                    "ppl_Name_First": "A",
                    "ppl_Name_Last": "B",
                    "ppl_CreationTS": "2025-01-01 00:00:00",
                    "ppl_ModificationTS": "2025-01-01 00:00:00",
                }
            ],
            "has_more": True,
            "next_since": "2025-01-01 00:00:00",
            "next_since_pk": 1,
        },
        {
            "ok": True,
            "table": "iSPEC_People",
            "items": [
                {
                    "ppl_PPLRecNo": 2,
                    "ppl_AddedBy": "legacy",
                    "ppl_Name_First": "C",
                    "ppl_Name_Last": "D",
                    "ppl_CreationTS": "2025-01-02 00:00:00",
                    "ppl_ModificationTS": "2025-01-02 00:00:00",
                }
            ],
            "has_more": False,
            "next_since": "2025-01-03 00:00:00",
            "next_since_pk": 3,
        },
    ]

    captured_params: list[dict[str, object]] = []

    class DummyResponse:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    def fake_get(url, params=None, headers=None, auth=None, timeout=None):
        captured_params.append(dict(params or {}))
        return DummyResponse(pages[len(captured_params) - 1])

    from ispec.db import legacy_sync

    monkeypatch.setattr(legacy_sync.requests, "get", fake_get)

    summary = legacy_sync.sync_legacy_people(
        legacy_url="http://legacy.example",
        mapping_path=str(mapping_path),
        db_file_path=str(db_path),
        limit=1,
        dry_run=False,
    )

    assert summary["pages"] == 2
    assert summary["items"] == 2
    assert summary["inserted"] == 2

    assert len(captured_params) == 2
    assert "since" not in captured_params[0]
    assert captured_params[1]["since"] == "2025-01-01T00:00:00Z"
    assert captured_params[1]["since_pk"] == 1

    with get_session(file_path=str(db_path)) as session:
        state = session.get(LegacySyncState, "iSPEC_People")
        assert state is not None
        assert state.since == datetime(2025, 1, 3, 0, 0, 0)
        assert state.since_pk == 3
