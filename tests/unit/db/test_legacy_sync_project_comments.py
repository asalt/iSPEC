from __future__ import annotations

from datetime import datetime

from ispec.db.connect import get_session
from ispec.db.models import Person, Project, ProjectComment


def test_sync_legacy_project_comments_inserts_comments_and_system_person(tmp_path, monkeypatch):
    db_path = tmp_path / "sync.db"

    payload = {
        "ok": True,
        "table": "iSPEC_ProjectHistory",
        "items": [
            {
                "prh_PRJRecNo": 1498.0,
                "prh_ModificationTS": "2025-12-01 12:00:00",
                "prh_CreationTS": "2025-12-01 11:00:00",
                "prh_AddedBy": "legacy-user",
                "prh_CommentType": "meeting",
                "prh_Comment": "Met with customer.",
            },
            {
                "prh_PRJRecNo": 1498.0,
                "prh_ModificationTS": "2025-12-02 12:00:00",
                "prh_CreationTS": "2025-12-02 11:00:00",
                "prh_AddedBy": "legacy-user",
                "prh_CommentType": "billing",
                "prh_Comment": "",
            },
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
        captured["url"] = url
        captured["params"] = dict(params or {})
        return DummyResponse()

    from ispec.db import legacy_sync

    monkeypatch.setattr(legacy_sync.requests, "get", fake_get)

    summary = legacy_sync.sync_legacy_project_comments(
        legacy_url="http://legacy.example",
        db_file_path=str(db_path),
        project_id=1498,
    )

    assert summary["items"] == 2
    assert summary["inserted"] == 1
    assert summary["updated"] == 0
    assert summary["conflicted"] == 0
    assert summary["skipped_blank"] == 1

    assert captured["url"] == "http://legacy.example/api/v2/legacy/tables/iSPEC_ProjectHistory/rows"
    assert captured["params"]["id"] == 1498
    assert captured["params"]["pk_field"] == "prh_PRJRecNo"
    assert captured["params"]["modified_field"] == "prh_ModificationTS"

    with get_session(file_path=str(db_path)) as session:
        assert session.get(Project, 1498) is not None
        assert session.get(Person, 0) is not None
        comments = (
            session.query(ProjectComment)
            .filter(ProjectComment.project_id == 1498)
            .order_by(ProjectComment.id)
            .all()
        )
        assert len(comments) == 1
        assert comments[0].com_CommentType == "meeting"
        assert comments[0].com_AddedBy == "legacy-user"
        assert comments[0].com_Comment == "Met with customer."


def test_sync_legacy_project_comments_updates_existing_comment(tmp_path, monkeypatch):
    db_path = tmp_path / "sync.db"

    created = datetime(2025, 12, 1, 11, 0, 0)

    with get_session(file_path=str(db_path)) as session:
        session.add(
            Project(
                id=1498,
                prj_AddedBy="legacy_import",
                prj_ProjectTitle="X",
            )
        )
        session.add(
            Person(
                id=0,
                ppl_AddedBy="legacy_import",
                ppl_Name_First="System",
                ppl_Name_Last="System",
                ppl_ModificationTS=datetime(2026, 1, 1, 0, 0, 0),
            )
        )
        session.add(
            ProjectComment(
                project_id=1498,
                person_id=0,
                com_CreationTS=created,
                com_ModificationTS=datetime(2026, 1, 1, 0, 0, 0),
                com_LegacyImportTS=datetime(2026, 1, 1, 0, 0, 0),
                com_CommentType="meeting",
                com_AddedBy="legacy-user",
                com_Comment="Old",
            )
        )

    payload = {
        "ok": True,
        "table": "iSPEC_ProjectHistory",
        "items": [
            {
                "prh_PRJRecNo": 1498.0,
                "prh_ModificationTS": "2025-12-01 12:00:00",
                "prh_CreationTS": "2025-12-01 11:00:00",
                "prh_AddedBy": "legacy-user",
                "prh_CommentType": "meeting",
                "prh_Comment": "New",
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

    summary = legacy_sync.sync_legacy_project_comments(
        legacy_url="http://legacy.example",
        db_file_path=str(db_path),
        project_id=1498,
    )

    assert summary["inserted"] == 0
    assert summary["updated"] == 1
    assert summary["conflicted"] == 0

    with get_session(file_path=str(db_path)) as session:
        comment = (
            session.query(ProjectComment)
            .filter(ProjectComment.project_id == 1498)
            .filter(ProjectComment.person_id == 0)
            .one()
        )
        assert comment.com_Comment == "New"


def test_sync_legacy_project_comments_skips_conflicted_comment(tmp_path, monkeypatch):
    db_path = tmp_path / "sync.db"

    created = datetime(2025, 12, 1, 11, 0, 0)

    with get_session(file_path=str(db_path)) as session:
        session.add(
            Project(
                id=1498,
                prj_AddedBy="legacy_import",
                prj_ProjectTitle="X",
            )
        )
        session.add(
            Person(
                id=0,
                ppl_AddedBy="legacy_import",
                ppl_Name_First="System",
                ppl_Name_Last="System",
                ppl_ModificationTS=datetime(2026, 1, 1, 0, 0, 0),
            )
        )
        session.add(
            ProjectComment(
                project_id=1498,
                person_id=0,
                com_CreationTS=created,
                com_LegacyImportTS=datetime(2026, 1, 1, 0, 0, 0),
                com_ModificationTS=datetime(2026, 1, 2, 0, 0, 0),
                com_CommentType="meeting",
                com_AddedBy="legacy-user",
                com_Comment="Local edited",
            )
        )

    payload = {
        "ok": True,
        "table": "iSPEC_ProjectHistory",
        "items": [
            {
                "prh_PRJRecNo": 1498.0,
                "prh_ModificationTS": "2025-12-01 12:00:00",
                "prh_CreationTS": "2025-12-01 11:00:00",
                "prh_AddedBy": "legacy-user",
                "prh_CommentType": "meeting",
                "prh_Comment": "Legacy",
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

    summary = legacy_sync.sync_legacy_project_comments(
        legacy_url="http://legacy.example",
        db_file_path=str(db_path),
        project_id=1498,
    )

    assert summary["inserted"] == 0
    assert summary["updated"] == 0
    assert summary["conflicted"] == 1

    with get_session(file_path=str(db_path)) as session:
        comment = (
            session.query(ProjectComment)
            .filter(ProjectComment.project_id == 1498)
            .filter(ProjectComment.person_id == 0)
            .one()
        )
        assert comment.com_Comment == "Local edited"

