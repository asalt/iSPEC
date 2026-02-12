from __future__ import annotations

from pathlib import Path
from typing import Any

from ispec.logging import get_logger

from .legacy_sync import (
    sync_legacy_experiment_runs,
    sync_legacy_experiments,
    sync_legacy_people,
    sync_legacy_project_comments,
    sync_legacy_projects,
)

logger = get_logger(__file__)


def _normalize_dump_target(dump_json: str | Path | None) -> str | Path | None:
    if dump_json is None:
        return None
    raw = str(dump_json).strip()
    if not raw:
        return None
    path = Path(raw).expanduser()
    if path.suffix.lower() == ".json":
        return str(path.with_suffix(""))
    return dump_json


def _coerce_int_list(value: Any) -> list[int]:
    if not isinstance(value, list):
        return []
    out: list[int] = []
    for item in value:
        if isinstance(item, int):
            out.append(int(item))
            continue
        if isinstance(item, str) and item.strip().isdigit():
            out.append(int(item.strip()))
    return out


def sync_legacy_all(
    *,
    legacy_url: str | None = None,
    mapping_path: str | Path | None = None,
    schema_path: str | Path | None = None,
    db_file_path: str | None = None,
    limit: int = 1000,
    max_pages: int | None = None,
    reset_cursor: bool = False,
    dry_run: bool = False,
    backfill_missing: bool = True,
    max_project_comments: int = 25,
    max_experiment_runs: int = 25,
    dump_json: str | Path | None = None,
) -> dict[str, Any]:
    """Convenience wrapper to sync the core legacy metadata needed by iSPEC.

    Intended for:
    - local dev refreshes
    - supervisor/orchestrator periodic sync

    Strategy:
    1) Incremental sync of projects/people/experiments (cursor-based)
    2) Then, for a small number of recently touched projects/experiments:
       - fetch project history comments (project_comment)
       - fetch experiment runs (experiment_run)
    """

    normalized_dump = _normalize_dump_target(dump_json)
    if dump_json is not None and normalized_dump != dump_json:
        logger.info("sync_legacy_all: normalized dump target to directory=%s", normalized_dump)

    projects = sync_legacy_projects(
        legacy_url=legacy_url,
        mapping_path=mapping_path,
        schema_path=schema_path,
        db_file_path=db_file_path,
        limit=int(limit),
        max_pages=max_pages,
        reset_cursor=bool(reset_cursor),
        dry_run=bool(dry_run),
        backfill_missing=bool(backfill_missing),
        collect_ids=True,
        max_collected_ids=max(0, int(max_project_comments)),
        dump_json=normalized_dump,
    )

    people = sync_legacy_people(
        legacy_url=legacy_url,
        mapping_path=mapping_path,
        schema_path=schema_path,
        db_file_path=db_file_path,
        limit=int(limit),
        max_pages=max_pages,
        reset_cursor=bool(reset_cursor),
        dry_run=bool(dry_run),
        backfill_missing=bool(backfill_missing),
        dump_json=normalized_dump,
    )

    experiments = sync_legacy_experiments(
        legacy_url=legacy_url,
        mapping_path=mapping_path,
        schema_path=schema_path,
        db_file_path=db_file_path,
        limit=int(limit),
        max_pages=max_pages,
        reset_cursor=bool(reset_cursor),
        dry_run=bool(dry_run),
        backfill_missing=bool(backfill_missing),
        collect_ids=True,
        max_collected_ids=max(0, int(max_experiment_runs)),
        dump_json=normalized_dump,
    )

    touched_projects = _coerce_int_list(projects.get("touched_ids"))
    touched_experiments = _coerce_int_list(experiments.get("touched_ids"))

    comments_totals = {"items": 0, "inserted": 0, "updated": 0, "conflicted": 0}
    comments_rows: list[dict[str, Any]] = []
    for project_id in touched_projects[: max(0, int(max_project_comments))]:
        summary = sync_legacy_project_comments(
            legacy_url=legacy_url,
            schema_path=schema_path,
            db_file_path=db_file_path,
            project_id=int(project_id),
            dry_run=bool(dry_run),
            dump_json=normalized_dump,
        )
        comments_rows.append({"project_id": int(project_id), "summary": summary})
        for key in comments_totals:
            comments_totals[key] += int(summary.get(key) or 0)

    runs_totals = {"inserted": 0, "updated": 0, "backfilled": 0, "conflicted": 0}
    runs_rows: list[dict[str, Any]] = []
    for exp_id in touched_experiments[: max(0, int(max_experiment_runs))]:
        summary = sync_legacy_experiment_runs(
            legacy_url=legacy_url,
            mapping_path=mapping_path,
            schema_path=schema_path,
            db_file_path=db_file_path,
            experiment_id=int(exp_id),
            dry_run=bool(dry_run),
            backfill_missing=bool(backfill_missing),
            dump_json=normalized_dump,
        )
        runs_rows.append({"experiment_id": int(exp_id), "summary": summary})
        for key in runs_totals:
            runs_totals[key] += int(summary.get(key) or 0)

    return {
        "ok": True,
        "projects": projects,
        "people": people,
        "experiments": experiments,
        "project_comments": {
            "requested": len(touched_projects[: max(0, int(max_project_comments))]),
            "totals": comments_totals,
            "summaries": comments_rows[:10],
        },
        "experiment_runs": {
            "requested": len(touched_experiments[: max(0, int(max_experiment_runs))]),
            "totals": runs_totals,
            "summaries": runs_rows[:10],
        },
    }

