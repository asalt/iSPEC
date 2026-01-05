from __future__ import annotations

import json
from datetime import datetime

from ispec.db.connect import get_session
from ispec.db.models import LegacySyncState, Project


def test_sync_legacy_projects_single_id_inserts_project(tmp_path, monkeypatch):
    db_path = tmp_path / "sync.db"

    payload = {
        "ok": True,
        "table": "iSPEC_Projects",
        "items": [
            {
                "prj_PRJRecNo": 1494.0,
                "prj_AddedBy": None,
                "prj_ProjectTitle": "Tbk1/2 phosphoproteomics",
                "prj_CreationTS": "2025-09-29 11:14:33",
                "prj_ModificationTS": "2025-12-05 12:16:37",
                "prj_Status": "closed",
                "prj_RnD": "No",
                "prj_Current_FLAG": None,
                "prj_ProjectCostMinimum": "",
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
        captured["headers"] = dict(headers or {})
        captured["auth"] = auth
        captured["timeout"] = timeout
        return DummyResponse()

    from ispec.db import legacy_sync

    monkeypatch.setattr(legacy_sync.requests, "get", fake_get)

    summary = legacy_sync.sync_legacy_projects(
        legacy_url="http://legacy.example",
        db_file_path=str(db_path),
        project_id=1494,
        dry_run=False,
    )

    assert summary["inserted"] == 1
    assert summary["updated"] == 0
    assert summary["conflicted"] == 0

    assert captured["url"] == "http://legacy.example/api/v2/legacy/tables/iSPEC_Projects/rows"
    assert captured["params"]["id"] == 1494

    with get_session(file_path=str(db_path)) as session:
        project = session.get(Project, 1494)
        assert project is not None
        assert project.prj_ProjectTitle == "Tbk1/2 phosphoproteomics"
        assert project.prj_AddedBy == "legacy_import"
        assert project.prj_PRJ_DisplayID == "MSPC001494"
        assert project.prj_PRJ_DisplayTitle.startswith("MSPC001494")
        assert project.prj_Status == "closed"
        assert project.prj_RnD is False
        assert project.prj_Current_FLAG is False
        assert project.prj_ProjectCostMinimum == 0
        assert isinstance(project.prj_CreationTS, datetime)
        assert isinstance(project.prj_ModificationTS, datetime)

        # Single-id sync should not advance the incremental cursor.
        assert session.query(LegacySyncState).count() == 0


def test_sync_legacy_projects_uses_basic_auth_from_env(tmp_path, monkeypatch):
    db_path = tmp_path / "sync.db"

    monkeypatch.setenv("ISPEC_LEGACY_USER", "legacy-user")
    monkeypatch.setenv("ISPEC_LEGACY_PASSWORD", "legacy-pass")

    payload = {"ok": True, "items": [{"prj_PRJRecNo": 1, "prj_ProjectTitle": "X", "prj_ModificationTS": None}]}
    captured: dict[str, object] = {}

    class DummyResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return payload

    def fake_get(url, params=None, headers=None, auth=None, timeout=None):
        captured["auth"] = auth
        return DummyResponse()

    from ispec.db import legacy_sync

    monkeypatch.setattr(legacy_sync.requests, "get", fake_get)

    legacy_sync.sync_legacy_projects(
        legacy_url="http://legacy.example",
        db_file_path=str(db_path),
        project_id=1,
    )

    auth = captured.get("auth")
    assert auth is not None
    assert getattr(auth, "username", None) == "legacy-user"
    assert getattr(auth, "password", None) == "legacy-pass"


def test_sync_legacy_projects_defaults_to_ispec_db_path_env(tmp_path, monkeypatch):
    db_path = tmp_path / "env.db"
    monkeypatch.setenv("ISPEC_DB_PATH", str(db_path))

    payload = {
        "ok": True,
        "table": "iSPEC_Projects",
        "items": [{"prj_PRJRecNo": 1, "prj_ProjectTitle": "X", "prj_ModificationTS": None}],
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

    legacy_sync.sync_legacy_projects(legacy_url="http://legacy.example", project_id=1)

    with get_session(file_path=str(db_path)) as session:
        project = session.get(Project, 1)
        assert project is not None
        assert project.prj_ProjectTitle == "X"


def test_sync_legacy_projects_backfills_missing_fields_on_conflict(tmp_path, monkeypatch):
    db_path = tmp_path / "sync.db"

    with get_session(file_path=str(db_path)) as session:
        session.add(
            Project(
                id=1,
                prj_AddedBy="user",
                prj_ProjectTitle="Local Title",
                prj_LegacyImportTS=datetime(2026, 1, 1, 0, 0, 0),
                prj_ModificationTS=datetime(2026, 1, 2, 0, 0, 0),
            )
        )

    payload = {
        "ok": True,
        "table": "iSPEC_Projects",
        "items": [
            {
                "prj_PRJRecNo": 1,
                "prj_AddedBy": "legacy-user",
                "prj_ProjectTitle": "Legacy Title",
                "prj_ProjectBackground": "Legacy background",
                "prj_ModificationTS": "2025-12-05 12:16:37",
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

    summary = legacy_sync.sync_legacy_projects(
        legacy_url="http://legacy.example",
        db_file_path=str(db_path),
        project_id=1,
        backfill_missing=True,
    )

    assert summary["backfilled"] == 1
    assert summary["conflicted"] == 0
    assert summary["updated"] == 0

    with get_session(file_path=str(db_path)) as session:
        project = session.get(Project, 1)
        assert project is not None
        assert project.prj_ProjectTitle == "Local Title"
        assert project.prj_ProjectBackground == "Legacy background"


def test_sync_legacy_projects_chunked_fields_recovers_missing_keys(tmp_path, monkeypatch):
    db_path = tmp_path / "sync.db"
    mapping_path = tmp_path / "mapping.json"

    mapping_path.write_text(
        json.dumps(
            {
                "tables": {
                    "iSPEC_Projects": {
                        "pk": {"legacy": "prj_PRJRecNo"},
                        "created_ts": "prj_CreationTS",
                        "modified_ts": "prj_ModificationTS",
                        "field_map": {
                            "prj_ProjectTitle": "prj_ProjectTitle",
                            "prj_ProjectDescription": "prj_ProjectDescription",
                            "prj_ProjectBackground": "prj_ProjectBackground",
                            "prj_Comments_Meeting": "prj_Comments_Meeting",
                            "prj_Comments_Billing": "prj_Comments_Billing",
                        },
                    }
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )

    # Force tiny chunks so the sync code issues multiple requests when it
    # detects missing keys from the legacy endpoint.
    monkeypatch.setenv("ISPEC_LEGACY_FIELDS_CHUNK_SIZE", "1")

    value_map = {
        "prj_PRJRecNo": 1478,
        "prj_CreationTS": "2025-01-01 00:00:00",
        "prj_ModificationTS": "2025-01-02 00:00:00",
        "prj_ProjectTitle": "Conditioned media profiling in Sgg species",
        "prj_ProjectDescription": "Please Input",
        "prj_ProjectBackground": "Please Input",
        "prj_Comments_Meeting": "Please Input",
        "prj_Comments_Billing": "Please Input",
    }

    captured_fields: list[list[str]] = []

    class DummyResponse:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    def fake_get(url, params=None, headers=None, auth=None, timeout=None):
        request_fields_raw = (params or {}).get("fields")
        if isinstance(request_fields_raw, str):
            request_fields = [f for f in request_fields_raw.split(",") if f]
        elif isinstance(request_fields_raw, list):
            request_fields = list(request_fields_raw)
        else:
            request_fields = []

        # Simulate a legacy endpoint that silently drops fields when too many
        # are requested at once.
        if len(request_fields) > 4:
            request_fields = request_fields[:4]

        captured_fields.append(request_fields)
        item = {k: value_map.get(k) for k in request_fields}
        return DummyResponse({"ok": True, "items": [item], "has_more": False})

    from ispec.db import legacy_sync

    monkeypatch.setattr(legacy_sync.requests, "get", fake_get)

    summary = legacy_sync.sync_legacy_projects(
        legacy_url="http://legacy.example",
        mapping_path=str(mapping_path),
        db_file_path=str(db_path),
        project_id=1478,
        limit=1,
        dry_run=False,
    )

    assert summary["inserted"] == 1
    assert summary["updated"] == 0
    assert len(captured_fields) >= 2

    with get_session(file_path=str(db_path)) as session:
        project = session.get(Project, 1478)
        assert project is not None
        assert project.prj_ProjectTitle == "Conditioned media profiling in Sgg species"
        assert project.prj_ProjectDescription == "Please Input"
        assert project.prj_ProjectBackground == "Please Input"
        assert project.prj_Comments_Meeting == "Please Input"
        assert project.prj_Comments_Billing == "Please Input"


def test_sync_legacy_projects_advances_cursor_and_paginates(tmp_path, monkeypatch):
    db_path = tmp_path / "sync.db"
    mapping_path = tmp_path / "mapping.json"

    mapping_path.write_text(
        json.dumps(
            {
                "tables": {
                    "iSPEC_Projects": {
                        "pk": {"legacy": "prj_PRJRecNo"},
                        "created_ts": "prj_CreationTS",
                        "modified_ts": "prj_ModificationTS",
                        "field_map": {"prj_ProjectTitle": "prj_ProjectTitle"},
                    }
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )

    pages = [
        {
            "ok": True,
            "table": "iSPEC_Projects",
            "items": [
                {
                    "prj_PRJRecNo": 1,
                    "prj_ProjectTitle": "One",
                    "prj_CreationTS": "2024-12-31 00:00:00",
                    "prj_ModificationTS": "2025-01-01 00:00:00",
                }
            ],
            "has_more": True,
            "next_since": "2025-01-01 00:00:00",
            "next_since_pk": 1,
        },
        {
            "ok": True,
            "table": "iSPEC_Projects",
            "items": [
                {
                    "prj_PRJRecNo": 2,
                    "prj_ProjectTitle": "Two",
                    "prj_CreationTS": "2025-01-01 00:00:00",
                    "prj_ModificationTS": "2025-01-02 00:00:00",
                }
            ],
            "has_more": False,
            "next_since": "2025-01-02 00:00:00",
            "next_since_pk": 2,
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

    summary = legacy_sync.sync_legacy_projects(
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
        state = session.get(LegacySyncState, "iSPEC_Projects")
        assert state is not None
        assert state.since == datetime(2025, 1, 2, 0, 0, 0)
        assert state.since_pk == 2
