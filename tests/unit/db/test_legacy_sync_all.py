from __future__ import annotations

from ispec.db import legacy_sync_all


def test_sync_legacy_all_combines_touched_and_recent_comment_projects(monkeypatch):
    comment_calls: list[int] = []
    run_calls: list[int] = []

    monkeypatch.setattr(
        legacy_sync_all,
        "sync_legacy_projects",
        lambda **kwargs: {"touched_ids": [1498], "items": 1},
    )
    monkeypatch.setattr(
        legacy_sync_all,
        "sync_legacy_people",
        lambda **kwargs: {"items": 0},
    )
    monkeypatch.setattr(
        legacy_sync_all,
        "sync_legacy_experiments",
        lambda **kwargs: {"touched_ids": [88], "items": 1},
    )
    monkeypatch.setattr(
        legacy_sync_all,
        "scan_recent_legacy_project_comment_projects",
        lambda **kwargs: {
            "items": 4,
            "projects": 2,
            "project_ids": [1498, 1501],
            "legacy_table": "iSPEC_ProjectHistory",
            "has_more": False,
            "recent_days": 30,
            "recent_window_start": "2026-02-26T00:00:00Z",
        },
    )

    def fake_sync_comments(**kwargs):
        comment_calls.append(int(kwargs["project_id"]))
        return {"items": 1, "inserted": 1, "updated": 0, "conflicted": 0}

    def fake_sync_runs(**kwargs):
        run_calls.append(int(kwargs["experiment_id"]))
        return {"inserted": 1, "updated": 0, "backfilled": 0, "conflicted": 0}

    monkeypatch.setattr(legacy_sync_all, "sync_legacy_project_comments", fake_sync_comments)
    monkeypatch.setattr(legacy_sync_all, "sync_legacy_experiment_runs", fake_sync_runs)

    summary = legacy_sync_all.sync_legacy_all(
        recent_project_comment_days=30,
        max_project_comments=5,
        max_experiment_runs=5,
    )

    assert comment_calls == [1498, 1501]
    assert run_calls == [88]
    assert summary["project_comments"]["requested"] == 2
    assert summary["project_comments"]["candidate_projects"] == [1498, 1501]
    assert summary["project_comments"]["source_counts"]["project_sync_touched"] == 1
    assert summary["project_comments"]["source_counts"]["recent_comment_scan"] == 2
    assert summary["project_comments"]["recent_scan"]["recent_days"] == 30
    assert summary["project_comments"]["totals"]["inserted"] == 2


def test_sync_legacy_all_skips_recent_comment_scan_when_disabled(monkeypatch):
    monkeypatch.setattr(
        legacy_sync_all,
        "sync_legacy_projects",
        lambda **kwargs: {"touched_ids": [1498], "items": 1},
    )
    monkeypatch.setattr(
        legacy_sync_all,
        "sync_legacy_people",
        lambda **kwargs: {"items": 0},
    )
    monkeypatch.setattr(
        legacy_sync_all,
        "sync_legacy_experiments",
        lambda **kwargs: {"touched_ids": [], "items": 0},
    )
    monkeypatch.setattr(
        legacy_sync_all,
        "scan_recent_legacy_project_comment_projects",
        lambda **kwargs: {
            "items": 0,
            "projects": 0,
            "project_ids": [],
            "legacy_table": None,
            "has_more": False,
            "recent_days": None,
            "recent_window_start": None,
            "disabled": True,
        },
    )

    comment_calls: list[int] = []
    monkeypatch.setattr(
        legacy_sync_all,
        "sync_legacy_project_comments",
        lambda **kwargs: comment_calls.append(int(kwargs["project_id"])) or {"items": 0, "inserted": 0, "updated": 0, "conflicted": 0},
    )
    monkeypatch.setattr(
        legacy_sync_all,
        "sync_legacy_experiment_runs",
        lambda **kwargs: {"inserted": 0, "updated": 0, "backfilled": 0, "conflicted": 0},
    )

    summary = legacy_sync_all.sync_legacy_all(
        recent_project_comment_days=None,
        max_project_comments=5,
        max_experiment_runs=5,
    )

    assert comment_calls == [1498]
    assert summary["project_comments"]["recent_scan"]["disabled"] is True
