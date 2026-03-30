from __future__ import annotations

from datetime import datetime

from ispec.db import legacy_sync
from ispec.db.connect import get_session
from ispec.db.models import Person, Project, ProjectComment


class DummyResponse:
    def __init__(self, payload: dict):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


def _seed_project_comment_db(db_path):
    with get_session(file_path=str(db_path)) as session:
        session.add(Project(id=1498, prj_AddedBy="seed", prj_ProjectTitle="Example"))
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
            Person(
                id=1,
                ppl_AddedBy="seed",
                ppl_Name_First="iSPEC",
                ppl_Name_Last="Assistant",
                ppl_ModificationTS=datetime(2026, 1, 1, 0, 0, 0),
            )
        )


def test_sync_project_comments_to_legacy_dry_run_filters_system_and_skips_existing(
    tmp_path,
    monkeypatch,
):
    db_path = tmp_path / "push.db"
    _seed_project_comment_db(db_path)

    existing_dt = datetime(2026, 1, 2, 3, 4, 5, 123456)
    new_dt = datetime(2026, 1, 3, 4, 5, 6, 654321)

    with get_session(file_path=str(db_path)) as session:
        session.add(
            ProjectComment(
                project_id=1498,
                person_id=1,
                com_CreationTS=existing_dt,
                com_Comment="already upstream",
                com_CommentType="assistant_note",
                com_AddedBy="api_key",
            )
        )
        session.add(
            ProjectComment(
                project_id=1498,
                person_id=1,
                com_CreationTS=new_dt,
                com_Comment="needs push",
                com_CommentType="meeting_note",
                com_AddedBy="api_key",
            )
        )
        session.add(
            ProjectComment(
                project_id=1498,
                person_id=0,
                com_CreationTS=datetime(2026, 1, 1, 2, 0, 0),
                com_Comment="imported system note",
                com_CommentType="meeting",
                com_AddedBy="legacy-user",
            )
        )

    get_calls: list[tuple[str, dict | None]] = []

    def fake_get(url, params=None, headers=None, auth=None, timeout=None):
        get_calls.append((url, dict(params or {})))
        if params == {"limit": 1}:
            return DummyResponse(
                {
                    "ok": True,
                    "table": "iSPEC_ProjectHistory",
                    "fields": [
                        "prh_PRJRecNo",
                        "prh_Comment",
                        "prh_AddedBy",
                        "prh_CreationTS",
                        "prh_CommentType",
                    ],
                    "items": [],
                }
            )
        return DummyResponse(
            {
                "ok": True,
                "table": "iSPEC_ProjectHistory",
                "items": [
                    {
                        "prh_PRJRecNo": 1498,
                        "prh_Comment": "already upstream",
                        "prh_AddedBy": "legacy-user",
                        "prh_CreationTS": "2026-01-02 03:04:05",
                        "prh_CommentType": "assistant_note",
                    }
                ],
                "has_more": False,
            }
        )

    def fake_post(url, json=None, headers=None, auth=None, timeout=None):
        raise AssertionError("dry-run should not POST to legacy")

    monkeypatch.setattr(legacy_sync.requests, "get", fake_get)
    monkeypatch.setattr(legacy_sync.requests, "post", fake_post)

    summary = legacy_sync.sync_project_comments_to_legacy(
        legacy_url="http://legacy.example",
        db_file_path=str(db_path),
        dry_run=True,
    )

    assert summary["selected"] == 3
    assert summary["candidate_comments"] == 2
    assert summary["projects"] == 1
    assert summary["legacy_table"] == "iSPEC_ProjectHistory"
    assert summary["already_present"] == 1
    assert summary["would_insert"] == 1
    assert summary["inserted"] == 0
    assert summary["skipped_system"] == 1
    assert summary["dry_run"] is True
    assert len(get_calls) == 2


def test_sync_project_comments_to_legacy_posts_missing_comments_with_comment_type(
    tmp_path,
    monkeypatch,
):
    db_path = tmp_path / "push.db"
    _seed_project_comment_db(db_path)

    created_dt = datetime(2026, 2, 4, 5, 6, 7, 111222)

    with get_session(file_path=str(db_path)) as session:
        session.add(
            ProjectComment(
                project_id=1498,
                person_id=1,
                com_CreationTS=created_dt,
                com_Comment="fresh local note",
                com_CommentType="assistant_note",
                com_AddedBy="api_key",
            )
        )

    posted: list[dict] = []

    def fake_get(url, params=None, headers=None, auth=None, timeout=None):
        if params == {"limit": 1}:
            return DummyResponse(
                {
                    "ok": True,
                    "table": "iSPEC_ProjectHistory",
                    "fields": [
                        "prh_PRJRecNo",
                        "prh_Comment",
                        "prh_AddedBy",
                        "prh_CreationTS",
                        "prh_CommentType",
                    ],
                    "items": [],
                }
            )
        return DummyResponse(
            {
                "ok": True,
                "table": "iSPEC_ProjectHistory",
                "items": [],
                "has_more": False,
            }
        )

    def fake_post(url, json=None, headers=None, auth=None, timeout=None):
        posted.append({"url": url, "json": dict(json or {})})
        return DummyResponse({"ok": True, "table": "iSPEC_ProjectHistory"})

    monkeypatch.setattr(legacy_sync.requests, "get", fake_get)
    monkeypatch.setattr(legacy_sync.requests, "post", fake_post)

    summary = legacy_sync.sync_project_comments_to_legacy(
        legacy_url="http://legacy.example",
        db_file_path=str(db_path),
        dry_run=False,
    )

    assert summary["candidate_comments"] == 1
    assert summary["already_present"] == 0
    assert summary["would_insert"] == 1
    assert summary["inserted"] == 1
    assert len(posted) == 1
    assert posted[0]["url"] == "http://legacy.example/api/v2/legacy/project-comments"
    assert posted[0]["json"]["project_recno"] == 1498
    assert posted[0]["json"]["note"] == "fresh local note"
    assert posted[0]["json"]["author"] == "Assistant, iSPEC"
    assert posted[0]["json"]["comment_type"] == "assistant_note"
    assert posted[0]["json"]["created_ts"].startswith("2026-02-04 05:06:07")


def test_sync_project_comments_to_legacy_returns_early_when_no_candidates(tmp_path, monkeypatch):
    db_path = tmp_path / "push.db"
    _seed_project_comment_db(db_path)

    with get_session(file_path=str(db_path)) as session:
        session.add(
            ProjectComment(
                project_id=1498,
                person_id=0,
                com_CreationTS=datetime(2026, 1, 1, 2, 0, 0),
                com_Comment="imported system note",
                com_CommentType="meeting",
                com_AddedBy="legacy-user",
            )
        )

    def fail_get(*args, **kwargs):
        raise AssertionError("should not call legacy read path when no local candidates exist")

    def fail_post(*args, **kwargs):
        raise AssertionError("should not call legacy write path when no local candidates exist")

    monkeypatch.setattr(legacy_sync.requests, "get", fail_get)
    monkeypatch.setattr(legacy_sync.requests, "post", fail_post)

    summary = legacy_sync.sync_project_comments_to_legacy(
        legacy_url="http://legacy.example",
        db_file_path=str(db_path),
        dry_run=False,
    )

    assert summary["selected"] == 1
    assert summary["candidate_comments"] == 0
    assert summary["projects"] == 0
    assert summary["legacy_table"] is None
    assert summary["would_insert"] == 0
    assert summary["inserted"] == 0
    assert summary["skipped_system"] == 1


def test_sync_project_comments_to_legacy_recent_days_filters_old_rows(
    tmp_path,
    monkeypatch,
):
    db_path = tmp_path / "push.db"
    _seed_project_comment_db(db_path)

    recent_dt = datetime(2026, 3, 20, 9, 0, 0)
    old_dt = datetime(2026, 1, 10, 9, 0, 0)

    with get_session(file_path=str(db_path)) as session:
        session.add(
            ProjectComment(
                project_id=1498,
                person_id=1,
                com_CreationTS=old_dt,
                com_ModificationTS=old_dt,
                com_Comment="old local note",
                com_CommentType="assistant_note",
                com_AddedBy="api_key",
            )
        )
        session.add(
            ProjectComment(
                project_id=1498,
                person_id=1,
                com_CreationTS=recent_dt,
                com_ModificationTS=recent_dt,
                com_Comment="recent local note",
                com_CommentType="assistant_note",
                com_AddedBy="api_key",
            )
        )

    def fake_get(url, params=None, headers=None, auth=None, timeout=None):
        if params == {"limit": 1}:
            return DummyResponse(
                {
                    "ok": True,
                    "table": "iSPEC_ProjectHistory",
                    "fields": [
                        "prh_PRJRecNo",
                        "prh_Comment",
                        "prh_AddedBy",
                        "prh_CreationTS",
                        "prh_CommentType",
                    ],
                    "items": [],
                }
            )
        return DummyResponse({"ok": True, "table": "iSPEC_ProjectHistory", "items": [], "has_more": False})

    monkeypatch.setattr(legacy_sync.requests, "get", fake_get)

    summary = legacy_sync.sync_project_comments_to_legacy(
        legacy_url="http://legacy.example",
        db_file_path=str(db_path),
        dry_run=True,
        recent_days=14,
        since="2026-03-10T00:00:00Z",
    )

    assert summary["selected"] == 1
    assert summary["candidate_comments"] == 1
    assert summary["would_insert"] == 1
    assert summary["recent_days"] == 14
    assert summary["recent_window_start"] == "2026-03-10T00:00:00Z"
