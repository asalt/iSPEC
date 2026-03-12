from __future__ import annotations

import configparser
import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import numpy as np
from sqlalchemy import Boolean, DateTime, Float, Integer
from sqlalchemy.exc import MultipleResultsFound

from ispec.db.connect import get_db_path, get_session
from ispec.db.models import (
    Experiment,
    ExperimentRun,
    LegacySyncState,
    Person,
    Project,
    ProjectComment,
)
from ispec.logging import get_logger

logger = get_logger(__file__)

_TRUTHY = {"1", "true", "yes", "y", "on"}

_PROJECT_COMMENT_TABLE_CANDIDATES = (
    "iSPEC_ProjectHistory",
    "ProjectHistory",
    "iSPEC_ProjectComments",
    "ProjectComments",
)
_PROJECT_COMMENT_PROJECT_FIELD_CANDIDATES = (
    "prh_PRJRecNo",
    "prjc_PRJRecNo",
    "prjcom_PRJRecNo",
    "project_PRJRecNo",
    "ProjectRecNo",
    "PRJRecNo",
)
_PROJECT_COMMENT_NOTE_FIELD_CANDIDATES = (
    "prh_Comment",
    "prjc_Comment",
    "prjc_Note",
    "prjcom_Comment",
    "prjcom_Note",
    "Comment",
    "Note",
)
_PROJECT_COMMENT_AUTHOR_FIELD_CANDIDATES = (
    "prh_AddedBy",
    "prjc_EnteredBy",
    "prjc_User",
    "prjcom_EnteredBy",
    "prjcom_User",
    "EnteredBy",
    "User",
)
_PROJECT_COMMENT_CREATED_TS_FIELD_CANDIDATES = (
    "prh_CreationTS",
    "prh_ModificationTS",
    "prjc_CreatedTS",
    "prjc_ModificationTS",
    "prjcom_CreatedTS",
    "prjcom_ModificationTS",
    "CreatedTS",
    "ModificationTS",
)
_PROJECT_COMMENT_TYPE_FIELD_CANDIDATES = (
    "prh_CommentType",
    "CommentType",
)


def _repo_ispec_dir() -> Path:
    return Path(__file__).resolve().parents[3]


def default_mapping_path() -> Path:
    return _repo_ispec_dir() / "data" / "legacy-mapping.json"


def default_schema_path() -> Path:
    return _repo_ispec_dir() / "data" / "ispec-legacy-schema.json"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _normalize_base_url(raw: str) -> str:
    raw = (raw or "").strip()
    if not raw:
        raise ValueError("missing legacy base url")
    if not raw.startswith(("http://", "https://")):
        raw = f"http://{raw}"
    return raw.rstrip("/")


def _is_truthy(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in _TRUTHY


def _legacy_debug_requests_enabled() -> bool:
    return _is_truthy(os.getenv("ISPEC_LEGACY_DEBUG_REQUESTS")) or _is_truthy(
        os.getenv("ISPEC_LEGACY_TRACE_REQUESTS")
    )


def _prepared_request_url(url: str, params: dict[str, Any]) -> str:
    try:
        prepared = requests.Request("GET", url, params=params).prepare()
        return prepared.url or url
    except Exception:  # pragma: no cover - best effort debug helper
        return url


def _summarize_legacy_params(params: dict[str, Any]) -> dict[str, Any]:
    summarized: dict[str, Any] = {}
    for key, value in params.items():
        if key != "fields":
            summarized[key] = value
            continue
        if isinstance(value, list):
            summarized[key] = f"<{len(value)} fields>"
        elif isinstance(value, str):
            count = len([part for part in value.split(",") if part])
            summarized[key] = f"<{count} fields>"
        else:
            summarized[key] = value
    return summarized


def _legacy_params_with_fields(
    params: dict[str, Any],
    *,
    fields: list[str],
    mode: str,
) -> dict[str, Any]:
    params_with_fields: dict[str, Any] = dict(params)
    if mode == "repeat":
        params_with_fields["fields"] = list(fields)
    elif mode == "csv":
        params_with_fields["fields"] = ",".join(fields)
    elif mode == "none":
        params_with_fields.pop("fields", None)
    return params_with_fields


def _resolve_legacy_dump_targets(
    dump_json: str | Path | None,
) -> tuple[Path | None, Path | None]:
    dump_file: str | None = None
    dump_dir: str | None = None
    if dump_json is not None:
        dump_file = str(dump_json).strip()
    else:
        dump_file = (os.getenv("ISPEC_LEGACY_DUMP_JSON") or "").strip() or None
        dump_dir = (os.getenv("ISPEC_LEGACY_DUMP_DIR") or "").strip() or None

    if dump_file:
        path = Path(dump_file).expanduser()
        if path.exists() and path.is_dir():
            return None, path
        if path.suffix.lower() == ".json":
            return path, None
        return None, path

    if dump_dir:
        return None, Path(dump_dir).expanduser()

    return None, None


def _dump_legacy_payload(payload: dict[str, Any], *, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )


def _legacy_dump_path_for_request(
    dump_file: Path | None,
    dump_dir: Path | None,
    *,
    table: str,
    page: int,
    mode: str,
    id_value: int | None,
) -> Path | None:
    if dump_file is not None:
        if page <= 1:
            return dump_file
        stem = dump_file.stem or dump_file.name
        suffix = dump_file.suffix or ".json"
        return dump_file.with_name(f"{stem}.page{page}{suffix}")

    if dump_dir is None:
        return None

    id_part = f"-{id_value}" if id_value is not None else ""
    return dump_dir / f"{table}{id_part}.page{page}.mode{mode}.json"


def _coerce_bool(value: Any) -> bool | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        return bool(int(value))
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off", ""}:
            return False
    return None


def _coerce_datetime(value: Any) -> datetime | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        parsed = pd.to_datetime(value, errors="coerce", utc=True)
        if pd.isna(parsed):
            return None
        return parsed.to_pydatetime()
    return None


def _normalize_datetime(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value
    return value.astimezone(UTC).replace(tzinfo=None)


def _format_cursor_datetime(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _can_update_imported(
    *,
    obj: Any,
    added_by_field: str,
    created_field: str,
    modified_field: str,
    import_ts_field: str,
) -> bool:
    from datetime import timedelta

    # When writing via imports/sync, our local ``*_ModificationTS`` timestamps can
    # drift slightly past the recorded import timestamp (e.g., defaults or
    # onupdate firing during flush/commit). Treat small drifts as safe updates so
    # we don't incorrectly mark fresh legacy imports as "conflicted".
    drift_grace = timedelta(seconds=60)

    imported_at = _normalize_datetime(getattr(obj, import_ts_field, None))
    modified_at = _normalize_datetime(getattr(obj, modified_field, None))
    if imported_at is None:
        if getattr(obj, added_by_field, None) != "legacy_import":
            return False
        created_at = _normalize_datetime(getattr(obj, created_field, None))
        if created_at is None or modified_at is None:
            return True
        if modified_at <= created_at:
            return True
        return (modified_at - created_at) <= timedelta(seconds=5)
    if modified_at is None:
        return True
    if modified_at <= imported_at:
        return True
    return (modified_at - imported_at) <= drift_grace


def _can_update_imported_comment(comment: ProjectComment) -> bool:
    imported_at = _normalize_datetime(getattr(comment, "com_LegacyImportTS", None))
    modified_at = _normalize_datetime(getattr(comment, "com_ModificationTS", None))
    if imported_at is None:
        if int(getattr(comment, "person_id", -1)) != 0:
            return False
        created_at = _normalize_datetime(getattr(comment, "com_CreationTS", None))
        return bool(created_at is not None and modified_at == created_at)
    if modified_at is None:
        return True
    return modified_at <= imported_at


def _legacy_headers() -> dict[str, str]:
    api_key = (os.getenv("ISPEC_LEGACY_API_KEY") or "").strip()
    if not api_key:
        return {}
    return {"X-API-Key": api_key}


def _load_ispec_conf(path: Path) -> dict[str, str]:  #configparser  would also probably work here but can leave this as is
    """Load a minimal INI-like config file.

    Supports both:
      - Standard INI with sections
      - Simple KEY=value files (no section header)
    """

    if not path.exists():
        return {}

    raw = path.read_text(encoding="utf-8")
    stripped_lines = [
        line.strip()
        for line in raw.splitlines()
        if line.strip() and not line.lstrip().startswith(("#", ";"))
    ]
    if not stripped_lines:
        return {}

    normalized = raw
    if not any(line.startswith("[") and "]" in line for line in stripped_lines[:5]):
        normalized = "[DEFAULT]\n" + raw

    parser = configparser.ConfigParser()
    try:
        parser.read_string(normalized)
    except configparser.Error:
        return {}

    def pick(key: str) -> str | None:
        candidates: list[str | None] = []
        if "DEFAULT" in parser:
            candidates.append(parser["DEFAULT"].get(key))
        for section in parser.sections():
            candidates.append(parser[section].get(key))
        for value in candidates:
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None

    result: dict[str, str] = {}
    for key in ("url", "user", "pw"):
        value = pick(key)
        if value is not None:
            result[key] = value
    return result


def _legacy_basic_auth():
    """Resolve legacy HTTP Basic auth (user/pw) from env or ~/.ispec/ispec.conf."""

    from requests.auth import HTTPBasicAuth

    user = (os.getenv("ISPEC_LEGACY_USER") or "").strip()
    pw = (os.getenv("ISPEC_LEGACY_PASSWORD") or "").strip()

    if user and pw:
        return HTTPBasicAuth(user, pw)

    conf_path = Path(
        os.getenv("ISPEC_LEGACY_CONF") or (Path.home() / ".ispec" / "ispec.conf")
    )
    conf = _load_ispec_conf(conf_path)
    user = user or conf.get("user", "")
    pw = pw or conf.get("pw", "")

    if user and pw:
        return HTTPBasicAuth(user, pw)

    return None


@dataclass(frozen=True)
class LegacyTablePlan:
    legacy_table: str
    legacy_pk_field: str
    legacy_modified_field: str
    legacy_created_field: str | None
    field_map: dict[str, str]


@dataclass(frozen=True)
class LegacyProjectCommentSource:
    table: str
    project_field: str
    note_field: str
    author_field: str | None
    created_ts_field: str | None
    comment_type_field: str | None


def _load_projects_plan(mapping_path: Path) -> LegacyTablePlan:
    return _load_table_plan(mapping_path, legacy_table="iSPEC_Projects")


def _load_table_plan(mapping_path: Path, *, legacy_table: str) -> LegacyTablePlan:
    mapping = _load_json(mapping_path)
    table = mapping.get("tables", {}).get(legacy_table)
    if not isinstance(table, dict):
        raise KeyError(f"legacy mapping missing tables.{legacy_table}")

    pk = table.get("pk", {})
    if not isinstance(pk, dict):
        raise KeyError(f"legacy mapping missing pk section for {legacy_table}")

    field_map = table.get("field_map", {})
    if not isinstance(field_map, dict):
        raise KeyError(f"legacy mapping missing field_map for {legacy_table}")

    modified = table.get("modified_ts")
    if not isinstance(modified, str) or not modified.strip():
        raise KeyError(f"legacy mapping missing modified_ts for {legacy_table}")

    created = table.get("created_ts")
    created_field = created if isinstance(created, str) and created.strip() else None

    pk_field = pk.get("legacy")
    if not isinstance(pk_field, str) or not pk_field.strip():
        raise KeyError(f"legacy mapping missing pk.legacy for {legacy_table}")

    return LegacyTablePlan(
        legacy_table=legacy_table,
        legacy_pk_field=pk_field,
        legacy_modified_field=modified,
        legacy_created_field=created_field,
        field_map={str(k): str(v) for k, v in field_map.items()},
    )


def _resolve_legacy_url(*, legacy_url: str | None, schema_path: Path) -> str:
    env = (os.getenv("ISPEC_LEGACY_API_URL") or "").strip()
    if legacy_url:
        return _normalize_base_url(legacy_url)
    if env:
        return _normalize_base_url(env)

    conf_path = Path(
        os.getenv("ISPEC_LEGACY_CONF") or (Path.home() / ".ispec" / "ispec.conf")
    )
    conf = _load_ispec_conf(conf_path)
    conf_url = (conf.get("url") or "").strip()
    if conf_url:
        return _normalize_base_url(conf_url)

    schema = _load_json(schema_path)
    base_url = str(schema.get("base_url", "")).strip()
    if base_url:
        return _normalize_base_url(base_url)
    raise ValueError(
        "missing legacy base url (set ISPEC_LEGACY_API_URL, pass --legacy-url, or configure ~/.ispec/ispec.conf)"
    )


def _legacy_get_json(url: str, *, params: dict[str, Any] | None = None) -> dict[str, Any]:
    resp = requests.get(
        url,
        params=params,
        headers=_legacy_headers(),
        auth=_legacy_basic_auth(),
        timeout=190,
    )
    resp.raise_for_status()
    payload = resp.json()
    if not isinstance(payload, dict):
        raise ValueError(f"Legacy API returned a non-object payload for {url}")
    if payload.get("ok") is False:
        raise ValueError(str(payload.get("error") or f"Legacy API request failed for {url}"))
    return payload


def _legacy_post_json(url: str, *, payload: dict[str, Any]) -> dict[str, Any]:
    resp = requests.post(
        url,
        json=payload,
        headers=_legacy_headers(),
        auth=_legacy_basic_auth(),
        timeout=190,
    )
    resp.raise_for_status()
    body = resp.json()
    if not isinstance(body, dict):
        raise ValueError(f"Legacy API returned a non-object payload for {url}")
    if body.get("ok") is False:
        raise ValueError(str(body.get("error") or f"Legacy API request failed for {url}"))
    return body


def _resolve_first_available_field(
    available_fields: list[str],
    candidates: tuple[str, ...],
) -> str | None:
    field_lookup = {
        str(name).strip().lower(): str(name)
        for name in available_fields
        if str(name).strip()
    }
    for candidate in candidates:
        match = field_lookup.get(str(candidate).strip().lower())
        if match:
            return match
    return None


def _normalize_project_comment_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).replace("\r\n", "\n").replace("\r", "\n").strip()
    return text or None


def _project_comment_compare_timestamp(value: Any) -> str | None:
    dt = _normalize_datetime(_coerce_datetime(value))
    if dt is None:
        return None
    return dt.replace(microsecond=0).isoformat(sep=" ")


def _project_comment_key(
    *,
    project_id: int,
    note: str | None,
    created_ts: str | None,
) -> tuple[int, str, str | None] | None:
    normalized_note = _normalize_project_comment_text(note)
    if normalized_note is None:
        return None
    return (int(project_id), normalized_note, created_ts)


def _person_display_label(person: Person | None) -> str | None:
    if person is None:
        return None
    first = (getattr(person, "ppl_Name_First", "") or "").strip()
    last = (getattr(person, "ppl_Name_Last", "") or "").strip()
    label = f"{last}, {first}".strip().strip(",")
    return label or None


def _resolve_legacy_project_comment_source(base_url: str) -> LegacyProjectCommentSource:
    errors: list[str] = []
    for table_candidate in _PROJECT_COMMENT_TABLE_CANDIDATES:
        url = f"{base_url}/api/v2/legacy/tables/{table_candidate}/rows"
        try:
            payload = _legacy_get_json(url, params={"limit": 1})
        except Exception as exc:
            errors.append(f"{table_candidate}: {exc}")
            continue

        available_fields = [str(name) for name in (payload.get("fields") or []) if str(name).strip()]
        if not available_fields:
            errors.append(f"{table_candidate}: no fields returned")
            continue

        project_field = _resolve_first_available_field(
            available_fields,
            _PROJECT_COMMENT_PROJECT_FIELD_CANDIDATES,
        )
        note_field = _resolve_first_available_field(
            available_fields,
            _PROJECT_COMMENT_NOTE_FIELD_CANDIDATES,
        )
        if project_field is None or note_field is None:
            errors.append(
                f"{table_candidate}: unable to infer required fields from {available_fields}"
            )
            continue

        return LegacyProjectCommentSource(
            table=str(payload.get("table") or table_candidate),
            project_field=project_field,
            note_field=note_field,
            author_field=_resolve_first_available_field(
                available_fields,
                _PROJECT_COMMENT_AUTHOR_FIELD_CANDIDATES,
            ),
            created_ts_field=_resolve_first_available_field(
                available_fields,
                _PROJECT_COMMENT_CREATED_TS_FIELD_CANDIDATES,
            ),
            comment_type_field=_resolve_first_available_field(
                available_fields,
                _PROJECT_COMMENT_TYPE_FIELD_CANDIDATES,
            ),
        )

    joined = "; ".join(errors) if errors else "no candidates attempted"
    raise ValueError(f"Unable to resolve legacy ProjectComments source: {joined}")


def _preferred_project_comment_author(
    *,
    comment: ProjectComment,
    person: Person | None,
) -> str | None:
    label = _person_display_label(person)
    if label and label != "System, System":
        return label
    return _normalize_project_comment_text(getattr(comment, "com_AddedBy", None))


def _fetch_legacy_project_comment_keys(
    *,
    base_url: str,
    source: LegacyProjectCommentSource,
    project_id: int,
    limit: int,
) -> tuple[set[tuple[int, str, str | None]], int]:
    fields = [source.project_field, source.note_field]
    if source.author_field:
        fields.append(source.author_field)
    if source.created_ts_field:
        fields.append(source.created_ts_field)

    merge_key_fields = [source.project_field, source.note_field]
    if source.created_ts_field:
        merge_key_fields.append(source.created_ts_field)
    if source.author_field:
        merge_key_fields.append(source.author_field)

    url = f"{base_url}/api/v2/legacy/tables/{source.table}/rows"
    params: dict[str, Any] = {
        "pk_field": source.project_field,
        "id": int(project_id),
        "limit": int(limit),
    }
    if source.created_ts_field:
        params["order_by"] = f"-{source.created_ts_field}"

    payload, _fields_mode = _fetch_legacy_rows_best_effort(
        url=url,
        params=params,
        modes=["repeat", "csv", "none"],
        fields=list(fields),
        expected_fields=set(fields),
        required_fields=[source.project_field, source.note_field],
        merge_key_fields=merge_key_fields,
        threshold_missing=max(1, len(fields) // 2),
        log_label="legacy project comments",
    )

    raw_items = payload.get("items") or payload.get("rows") or []
    items: list[dict[str, Any]] = [item for item in raw_items if isinstance(item, dict)]
    keys: set[tuple[int, str, str | None]] = set()
    for item in items:
        raw_project_id = item.get(source.project_field)
        try:
            legacy_project_id = int(raw_project_id)
        except Exception:
            continue
        if legacy_project_id != int(project_id):
            continue
        key = _project_comment_key(
            project_id=legacy_project_id,
            note=item.get(source.note_field),
            created_ts=_project_comment_compare_timestamp(
                item.get(source.created_ts_field) if source.created_ts_field else None
            ),
        )
        if key is not None:
            keys.add(key)
    return keys, len(items)


def sync_project_comments_to_legacy(
    *,
    legacy_url: str | None = None,
    schema_path: str | Path | None = None,
    db_file_path: str | None = None,
    project_id: int | None = None,
    limit: int = 5000,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Push local non-System project comments to the legacy comment/history API."""

    if not db_file_path:
        db_file_path = (os.getenv("ISPEC_DB_PATH") or "").strip() or get_db_path()

    selected_rows = 0
    skipped_blank = 0
    skipped_system = 0
    duplicates_skipped = 0
    local_candidates: list[dict[str, Any]] = []
    seen_local_keys: set[tuple[int, str, str | None]] = set()

    with get_session(file_path=db_file_path) as session:
        query = (
            session.query(ProjectComment, Person)
            .outerjoin(Person, ProjectComment.person_id == Person.id)
            .order_by(ProjectComment.project_id.asc(), ProjectComment.com_CreationTS.asc(), ProjectComment.id.asc())
        )
        if project_id is not None:
            query = query.filter(ProjectComment.project_id == int(project_id))

        rows = query.all()
        selected_rows = len(rows)

        for comment, person in rows:
            person_label = _person_display_label(person)
            if int(getattr(comment, "person_id", 0) or 0) == 0 or person_label == "System, System":
                skipped_system += 1
                continue

            note = _normalize_project_comment_text(getattr(comment, "com_Comment", None))
            if note is None:
                skipped_blank += 1
                continue

            created_key = _project_comment_compare_timestamp(getattr(comment, "com_CreationTS", None))
            key = _project_comment_key(
                project_id=int(comment.project_id),
                note=note,
                created_ts=created_key,
            )
            if key is None:
                skipped_blank += 1
                continue
            if key in seen_local_keys:
                duplicates_skipped += 1
                continue
            seen_local_keys.add(key)

            local_candidates.append(
                {
                    "key": key,
                    "project_id": int(comment.project_id),
                    "comment_id": int(comment.id),
                    "note": note,
                    "author": _preferred_project_comment_author(comment=comment, person=person),
                    "created_ts": _normalize_datetime(_coerce_datetime(getattr(comment, "com_CreationTS", None))),
                    "comment_type": _normalize_project_comment_text(getattr(comment, "com_CommentType", None)),
                }
            )

    if not local_candidates:
        return {
            "selected": selected_rows,
            "candidate_comments": 0,
            "projects": 0,
            "legacy_table": None,
            "legacy_existing_items": 0,
            "already_present": 0,
            "would_insert": 0,
            "inserted": 0,
            "skipped_blank": skipped_blank,
            "skipped_system": skipped_system,
            "duplicates_skipped": duplicates_skipped,
            "dry_run": bool(dry_run),
        }

    resolved_schema = Path(schema_path).expanduser().resolve() if schema_path else default_schema_path()
    base_url = _resolve_legacy_url(legacy_url=legacy_url, schema_path=resolved_schema)
    source = _resolve_legacy_project_comment_source(base_url)
    by_project: dict[int, list[dict[str, Any]]] = {}
    for candidate in local_candidates:
        by_project.setdefault(int(candidate["project_id"]), []).append(candidate)

    legacy_existing_items = 0
    already_present = 0
    would_insert = 0
    inserted = 0

    post_url = f"{base_url}/api/v2/legacy/project-comments"

    for candidate_project_id in sorted(by_project):
        legacy_keys, fetched_count = _fetch_legacy_project_comment_keys(
            base_url=base_url,
            source=source,
            project_id=int(candidate_project_id),
            limit=int(limit),
        )
        legacy_existing_items += fetched_count

        for candidate in by_project[candidate_project_id]:
            key = candidate["key"]
            if key in legacy_keys:
                already_present += 1
                continue

            payload: dict[str, Any] = {
                "project_recno": int(candidate["project_id"]),
                "note": str(candidate["note"]),
            }
            author = candidate.get("author")
            if author:
                payload["author"] = str(author)
            created_ts = candidate.get("created_ts")
            if created_ts is not None:
                payload["created_ts"] = _normalize_datetime(created_ts).isoformat(sep=" ")
            comment_type = candidate.get("comment_type")
            if comment_type:
                payload["comment_type"] = str(comment_type)

            would_insert += 1
            if not dry_run:
                _legacy_post_json(post_url, payload=payload)
                inserted += 1
            legacy_keys.add(key)

    return {
        "selected": selected_rows,
        "candidate_comments": len(local_candidates),
        "projects": len(by_project),
        "legacy_table": source.table,
        "legacy_existing_items": legacy_existing_items,
        "already_present": already_present,
        "would_insert": would_insert,
        "inserted": inserted,
        "skipped_blank": skipped_blank,
        "skipped_system": skipped_system,
        "duplicates_skipped": duplicates_skipped,
        "dry_run": bool(dry_run),
    }


def _plan_fields_to_fetch(plan: LegacyTablePlan) -> list[str]:
    fields: list[str] = [plan.legacy_pk_field, plan.legacy_modified_field]
    if plan.legacy_created_field:
        fields.append(plan.legacy_created_field)
    for field in plan.field_map:
        if field not in fields:
            fields.append(field)
    return fields


def _projects_fields_to_fetch(plan: LegacyTablePlan) -> list[str]:
    """Backward-compatible alias (use _plan_fields_to_fetch)."""

    return _plan_fields_to_fetch(plan)


def _expected_field_stats(
    item: dict[str, Any], expected_fields: set[str]
) -> tuple[int, list[str]]:
    """Return (expected_fields_present, missing_expected_fields)."""

    if not expected_fields:
        return 0, []
    present = expected_fields.intersection(item.keys())
    missing = sorted(expected_fields - present)
    return len(present), missing


def _legacy_fields_chunk_size() -> int:
    """Max number of non-required fields per legacy request.

    Some legacy endpoints appear to drop/ignore fields when the requested field
    list becomes large (or when the query string grows). Chunking is only used
    as a fallback when we detect missing keys in the response.
    """

    raw = (os.getenv("ISPEC_LEGACY_FIELDS_CHUNK_SIZE") or "").strip()
    if not raw:
        return 25
    try:
        value = int(raw)
    except ValueError:
        return 25
    return max(0, min(200, value))


def _merge_missing_values(target: dict[str, Any], source: dict[str, Any]) -> None:
    for key, value in source.items():
        if key not in target:
            target[key] = value
            continue
        current = target.get(key)
        missing_current = current is None
        if isinstance(current, str):
            missing_current = missing_current or not current.strip()
        if missing_current and value is not None:
            target[key] = value


def _normalize_merge_key_part(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, float):
        if pd.isna(value):
            return None
        if value.is_integer():
            return int(value)
        return float(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            numeric = float(stripped)
        except Exception:
            return stripped
        if numeric.is_integer():
            return int(numeric)
        return numeric
    return value


def _row_merge_key(row: dict[str, Any], key_fields: list[str]) -> tuple[Any, ...] | None:
    parts: list[Any] = []
    for field in key_fields:
        part = _normalize_merge_key_part(row.get(field))
        if part is None:
            return None
        parts.append(part)
    return tuple(parts)


def _normalize_label_value(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "0"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return str(value)
    text = str(value).strip()
    return text if text else "0"


def _fetch_legacy_rows_chunked(
    *,
    url: str,
    params: dict[str, Any],
    mode: str,
    fields: list[str],
    required_fields: list[str],
    merge_key_fields: list[str] | None = None,
    log_label: str | None = None,
) -> dict[str, Any]:
    chunk_size = _legacy_fields_chunk_size()
    if chunk_size <= 0:
        raise ValueError("legacy fields chunking disabled")
    if mode not in {"repeat", "csv"}:
        raise ValueError(f"unsupported fields mode for chunking: {mode}")

    required_unique: list[str] = []
    for field in required_fields:
        if field and field not in required_unique:
            required_unique.append(field)

    merge_key_unique: list[str] = []
    if merge_key_fields:
        for field in merge_key_fields:
            if field and field not in merge_key_unique:
                merge_key_unique.append(field)
        for field in merge_key_unique:
            if field not in required_unique:
                required_unique.append(field)

    optional_fields: list[str] = []
    for field in fields:
        if field in required_unique:
            continue
        if field and field not in optional_fields:
            optional_fields.append(field)

    # Even when the requested field list is relatively short, the legacy
    # endpoint may still drop keys. When chunking is requested, force at least
    # two chunks when possible by capping the chunk size to ~half of the
    # optional fields.
    if optional_fields:
        chunk_size = min(chunk_size, max(1, len(optional_fields) // 2))

    chunks: list[list[str]] = []
    for idx in range(0, len(optional_fields), chunk_size):
        chunks.append(optional_fields[idx : idx + chunk_size])
    if not chunks:
        chunks = [[]]

    base_payload: dict[str, Any] | None = None
    items_key: str = "items"
    merge_key_fields_use = merge_key_unique or required_unique[:1]
    merged_items_by_key: dict[tuple[Any, ...], dict[str, Any]] = {}
    merged_items: list[dict[str, Any]] = []

    debug_requests = _legacy_debug_requests_enabled()
    label = log_label or "legacy"

    for idx, chunk in enumerate(chunks):
        request_fields = [*required_unique, *chunk]
        request_params: dict[str, Any] = dict(params)
        if mode == "repeat":
            request_params["fields"] = request_fields
        else:  # mode == "csv"
            request_params["fields"] = ",".join(request_fields)

        if debug_requests:
            logger.info(
                "%s chunk_request[%d/%d] fields=%s",
                label,
                idx + 1,
                len(chunks),
                request_fields,
            )
            logger.info(
                "%s chunk_request_url[%d/%d] %s",
                label,
                idx + 1,
                len(chunks),
                _prepared_request_url(url, request_params),
            )

        resp = requests.get(
            url,
            params=request_params,
            headers=_legacy_headers(),
            auth=_legacy_basic_auth(),
            timeout=90,
        )
        resp.raise_for_status()
        candidate = resp.json()

        raw_items = candidate.get("items") or candidate.get("rows") or []
        items: list[dict[str, Any]] = list(raw_items)
        if idx == 0:
            base_payload = candidate
            items_key = (
                "items"
                if "items" in candidate
                else ("rows" if "rows" in candidate else "items")
            )
            merged_items = items
            merged_items_by_key = {}
            for row in merged_items:
                if not isinstance(row, dict):
                    continue
                key = _row_merge_key(row, merge_key_fields_use)
                if key is None:
                    continue
                merged_items_by_key[key] = row
            continue

        for row in items:
            if not isinstance(row, dict):
                continue
            key = _row_merge_key(row, merge_key_fields_use)
            if key is None:
                continue
            target = merged_items_by_key.get(key)
            if target is None:
                continue
            _merge_missing_values(target, row)

    if base_payload is None:
        raise RuntimeError("chunked legacy fetch returned no payload")

    base_payload[items_key] = merged_items
    return base_payload


def _fetch_legacy_rows_best_effort(
    *,
    url: str,
    params: dict[str, Any],
    modes: list[str],
    fields: list[str],
    expected_fields: set[str],
    required_fields: list[str],
    merge_key_fields: list[str] | None = None,
    threshold_missing: int,
    log_label: str,
) -> tuple[dict[str, Any], str]:
    """Fetch legacy rows using the best available ``fields`` encoding.

    The legacy endpoints are finicky about how ``fields`` is supplied:
    - ``repeat``: fields=["a","b"] -> ?fields=a&fields=b
    - ``csv``: fields="a,b"
    - ``none``: omit the parameter and accept the server defaults

    Additionally, some endpoints silently drop keys when too many fields are
    requested (or when the query string grows). When we detect missing keys we
    retry using chunked field requests and merge results by primary key.
    """

    attempted_modes: list[str] = []
    required_unique: list[str] = []
    for field in required_fields:
        if field and field not in required_unique:
            required_unique.append(field)
    required_set = set(required_unique)

    best_payload: dict[str, Any] | None = None
    best_mode: str | None = None
    best_required_present = -1
    best_present = -1
    best_missing_required: list[str] = []
    best_missing: list[str] = []

    debug_requests = _legacy_debug_requests_enabled()

    for mode in modes:
        if not mode:
            continue
        attempted_modes.append(mode)

        params_with_fields: dict[str, Any] = dict(params)
        if mode == "repeat":
            params_with_fields["fields"] = list(fields)
        elif mode == "csv":
            params_with_fields["fields"] = ",".join(fields)
        elif mode == "none":
            params_with_fields.pop("fields", None)
        else:  # pragma: no cover - defensive
            continue

        logger.info(
            "%s fetch: %s mode=%s params=%s",
            log_label,
            url,
            mode,
            _summarize_legacy_params(params_with_fields),
        )
        if debug_requests:
            logger.info(
                "%s request_url mode=%s %s",
                log_label,
                mode,
                _prepared_request_url(url, params_with_fields),
            )

        resp = requests.get(
            url,
            params=params_with_fields,
            headers=_legacy_headers(),
            auth=_legacy_basic_auth(),
            timeout=190,
        )
        resp.raise_for_status()
        candidate = resp.json()

        raw_items = candidate.get("items") or candidate.get("rows") or []
        items: list[dict[str, Any]] = list(raw_items)
        if not items:
            return candidate, mode

        present, missing = _expected_field_stats(items[0], expected_fields)
        missing_count = len(missing)
        required_present, missing_required = _expected_field_stats(items[0], required_set)
        missing_required_count = len(missing_required)
        keys_count = len(items[0].keys())

        if missing_count > 0 and mode in {"repeat", "csv"} and _legacy_fields_chunk_size() > 0:
            try:
                chunked = _fetch_legacy_rows_chunked(
                    url=url,
                    params=params,
                    mode=mode,
                    fields=list(fields),
                    required_fields=required_unique,
                    merge_key_fields=merge_key_fields,
                    log_label=log_label,
                )
                chunked_items = list(chunked.get("items") or chunked.get("rows") or [])
                if chunked_items:
                    chunked_present, chunked_missing = _expected_field_stats(
                        chunked_items[0], expected_fields
                    )
                    chunked_required_present, chunked_missing_required = _expected_field_stats(
                        chunked_items[0], required_set
                    )
                    if (
                        chunked_required_present > required_present
                        or chunked_present > present
                    ):
                        candidate = chunked
                        items = chunked_items
                        present = chunked_present
                        missing = chunked_missing
                        missing_count = len(chunked_missing)
                        required_present = chunked_required_present
                        missing_required = chunked_missing_required
                        missing_required_count = len(chunked_missing_required)
                        keys_count = len(chunked_items[0].keys())
                        logger.info(
                            "%s improved field coverage via chunking (mode=%s chunk_size=%d keys=%d required_present=%d/%d expected_present=%d/%d)",
                            log_label,
                            mode,
                            _legacy_fields_chunk_size(),
                            keys_count,
                            required_present,
                            len(required_set),
                            present,
                            len(expected_fields),
                        )
            except Exception as exc:
                logger.warning("%s chunked fetch failed (mode=%s): %s", log_label, mode, exc)

        logger.info(
            "%s fetch result: mode=%s keys=%d required_present=%d/%d expected_present=%d/%d",
            log_label,
            mode,
            keys_count,
            required_present,
            len(required_set),
            present,
            len(expected_fields),
        )

        if (
            required_present > best_required_present
            or (required_present == best_required_present and present > best_present)
        ):
            best_payload = candidate
            best_mode = mode
            best_required_present = required_present
            best_present = present
            best_missing_required = list(missing_required)
            best_missing = list(missing)

        if missing_required_count == 0 and missing_count <= threshold_missing:
            return candidate, mode

        if missing_required_count:
            sample_missing_required = ", ".join(missing_required[:10])
            logger.warning(
                "%s missing required fields (mode=%s); missing_required=%s",
                log_label,
                mode,
                sample_missing_required,
            )
        elif missing_count:
            sample_missing = ", ".join(missing[:10])
            logger.warning(
                "%s rows missing %d/%d expected fields (mode=%s); sample_missing=%s",
                log_label,
                missing_count,
                len(expected_fields),
                mode,
                sample_missing,
            )

    if best_payload is None or best_mode is None:  # pragma: no cover - defensive
        raise RuntimeError(f"{log_label} fetch failed after modes={attempted_modes}")

    if best_missing_required:
        logger.warning(
            "%s response still missing required fields in best mode=%s; missing_required=%s",
            log_label,
            best_mode,
            ", ".join(best_missing_required[:10]),
        )
    elif best_missing:
        logger.warning(
            "%s rows missing %d/%d expected fields even in best mode=%s; using best anyway (sample_missing=%s)",
            log_label,
            len(expected_fields) - best_present,
            len(expected_fields),
            best_mode,
            ", ".join(best_missing[:10]),
        )

    return best_payload, best_mode


def _coerce_model_field(model, field: str, value: Any) -> Any:
    column = model.__table__.columns.get(field)  # type: ignore[attr-defined]
    if column is None:
        return value

    if value is None or (isinstance(value, float) and pd.isna(value)):
        if not column.nullable:
            if isinstance(column.type, Boolean):
                return False
            if isinstance(column.type, Integer):
                return 0
            if isinstance(column.type, Float):
                return 0.0
        return None

    if isinstance(column.type, Boolean):
        coerced = _coerce_bool(value)
        if coerced is None and not column.nullable:
            return False
        return coerced

    if isinstance(column.type, DateTime):
        return _normalize_datetime(_coerce_datetime(value))

    if isinstance(column.type, Integer):
        if isinstance(value, str) and not value.strip():
            return None if column.nullable else 0
        try:
            return int(value)
        except Exception:
            return None if column.nullable else 0

    if isinstance(column.type, Float):
        if isinstance(value, str) and not value.strip():
            return None if column.nullable else 0.0
        try:
            return float(value)
        except Exception:
            return None if column.nullable else 0.0

    if isinstance(value, str):
        stripped = value.strip()
        return stripped if stripped else None

    return value


def _ensure_placeholder_project(session, project_id: int) -> None:
    if session.get(Project, project_id) is not None:
        return

    display_id = f"MSPC{project_id:06d}"
    title = f"Untitled (PRJ {project_id})"
    session.add(
        Project(
            id=project_id,
            prj_AddedBy="legacy_import",
            prj_ProjectTitle=title,
            prj_PRJ_DisplayID=display_id,
            prj_PRJ_DisplayTitle=f"{display_id} - {title}",
        )
    )
    session.flush()


def _ensure_system_person(session, *, imported_at: datetime) -> None:
    if session.get(Person, 0) is not None:
        return

    session.add(
        Person(
            id=0,
            ppl_AddedBy="legacy_import",
            ppl_LegacyImportTS=imported_at,
            ppl_Name_First="System",
            ppl_Name_Last="System",
            ppl_ModificationTS=imported_at,
        )
    )
    session.flush()


def _ensure_placeholder_experiment(
    session,
    experiment_id: int,
    *,
    project_id: int | None = None,
) -> None:
    if session.get(Experiment, experiment_id) is not None:
        return
    if project_id is not None:
        _ensure_placeholder_project(session, project_id)
    session.add(
        Experiment(
            id=experiment_id,
            project_id=project_id,
            record_no=str(experiment_id),
        )
    )
    session.flush()


def _build_person_record(
    item: dict[str, Any],
    *,
    plan: LegacyTablePlan,
    imported_at: datetime,
) -> tuple[int, dict[str, Any], datetime | None] | None:
    legacy_pk = item.get(plan.legacy_pk_field)
    if legacy_pk is None:
        return None
    try:
        person_id = int(legacy_pk)
    except Exception:
        return None
    if person_id <= 0:
        return None

    record: dict[str, Any] = {
        "ppl_LegacyPersonID": person_id,
        "ppl_LegacyImportTS": imported_at,
    }

    for legacy_field, local_field in plan.field_map.items():
        if legacy_field not in item:
            continue
        # Keep local modification timestamps app-managed; preserve legacy creation.
        if local_field.endswith("_ModificationTS"):
            continue
        record[local_field] = _coerce_model_field(Person, local_field, item.get(legacy_field))

    if plan.legacy_created_field:
        created_dt = _normalize_datetime(_coerce_datetime(item.get(plan.legacy_created_field)))
        if created_dt is not None:
            record["ppl_CreationTS"] = created_dt

    added_by = record.get("ppl_AddedBy")
    if not (isinstance(added_by, str) and added_by.strip()):
        record["ppl_AddedBy"] = "legacy_import"

    first = record.get("ppl_Name_First")
    if not (isinstance(first, str) and first.strip()):
        record["ppl_Name_First"] = "Unknown"

    last = record.get("ppl_Name_Last")
    if not (isinstance(last, str) and last.strip()):
        record["ppl_Name_Last"] = f"Unknown ({person_id})"

    last_modified_raw = item.get(plan.legacy_modified_field)
    last_modified_dt = _normalize_datetime(_coerce_datetime(last_modified_raw))
    return person_id, record, last_modified_dt


def _apply_person_record(
    session,
    *,
    legacy_person_id: int,
    record: dict[str, Any],
    dry_run: bool,
    backfill_missing: bool,
) -> str:
    legacy_person_id = int(legacy_person_id)

    def _person_name_matches(obj: Person, first: str, last: str) -> bool:
        obj_first = str(getattr(obj, "ppl_Name_First", "") or "").strip().lower()
        obj_last = str(getattr(obj, "ppl_Name_Last", "") or "").strip().lower()
        return obj_first == first.strip().lower() and obj_last == last.strip().lower()

    def _person_email_matches(obj: Person, email: str | None) -> bool:
        if not isinstance(email, str) or not email.strip():
            return False
        obj_email = str(getattr(obj, "ppl_Email", "") or "").strip().lower()
        return bool(obj_email) and obj_email == email.strip().lower()

    mapping_linked = False
    existing = (
        session.query(Person)
        .filter(Person.ppl_LegacyPersonID == legacy_person_id)
        .one_or_none()
    )

    if existing is None:
        by_id = session.get(Person, legacy_person_id)
        if by_id is not None:
            linked_id = getattr(by_id, "ppl_LegacyPersonID", None)
            row_name_ok = _person_name_matches(
                by_id,
                str(record.get("ppl_Name_First") or ""),
                str(record.get("ppl_Name_Last") or ""),
            )
            row_email_ok = _person_email_matches(by_id, record.get("ppl_Email"))
            if linked_id in {None, legacy_person_id} and (
                _can_update_imported(
                    obj=by_id,
                    added_by_field="ppl_AddedBy",
                    created_field="ppl_CreationTS",
                    modified_field="ppl_ModificationTS",
                    import_ts_field="ppl_LegacyImportTS",
                )
                or row_name_ok
                or row_email_ok
            ):
                existing = by_id
                if linked_id is None:
                    if not dry_run:
                        existing.ppl_LegacyPersonID = legacy_person_id
                    mapping_linked = True

    if existing is None:
        email = str(record.get("ppl_Email") or "").strip()
        if email:
            email_matches = (
                session.query(Person)
                .filter(Person.ppl_Email.ilike(email))
                .all()
            )
            valid_matches = [
                row
                for row in email_matches
                if getattr(row, "ppl_LegacyPersonID", None) in {None, legacy_person_id}
            ]
            if len(valid_matches) == 1:
                existing = valid_matches[0]
                if getattr(existing, "ppl_LegacyPersonID", None) is None:
                    if not dry_run:
                        existing.ppl_LegacyPersonID = legacy_person_id
                    mapping_linked = True

    if existing is None:
        first = str(record.get("ppl_Name_First") or "").strip()
        last = str(record.get("ppl_Name_Last") or "").strip()
        if first and last:
            name_matches = (
                session.query(Person)
                .filter(Person.ppl_Name_First.ilike(first))
                .filter(Person.ppl_Name_Last.ilike(last))
                .all()
            )
            valid_matches = [
                row
                for row in name_matches
                if getattr(row, "ppl_LegacyPersonID", None) in {None, legacy_person_id}
            ]
            if len(valid_matches) == 1:
                existing = valid_matches[0]
                if getattr(existing, "ppl_LegacyPersonID", None) is None:
                    if not dry_run:
                        existing.ppl_LegacyPersonID = legacy_person_id
                    mapping_linked = True

    if existing is None:
        insert_record = dict(record)
        # Preserve legacy IDs when available, but avoid collisions with local rows.
        if session.get(Person, legacy_person_id) is None:
            insert_record["id"] = legacy_person_id
        if not dry_run:
            session.add(Person(**insert_record))
        return "inserted"

    if _can_update_imported(
        obj=existing,
        added_by_field="ppl_AddedBy",
        created_field="ppl_CreationTS",
        modified_field="ppl_ModificationTS",
        import_ts_field="ppl_LegacyImportTS",
    ):
        if not dry_run:
            for key, value in record.items():
                if key == "id":
                    continue
                if hasattr(existing, key):
                    setattr(existing, key, value)
        return "updated"

    if not backfill_missing:
        return "backfilled" if mapping_linked else "conflicted"

    changed = False
    for key, value in record.items():
        if key in {"id", "ppl_LegacyImportTS"}:
            continue
        if not hasattr(existing, key):
            continue
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue

        current = getattr(existing, key, None)
        missing_current = current is None
        if isinstance(current, str):
            missing_current = missing_current or not current.strip()
        if key.endswith("_AddedBy") and current == "legacy_import":
            missing_current = True

        if not missing_current:
            continue

        if not dry_run:
            setattr(existing, key, value)
        changed = True

    if changed or mapping_linked:
        return "backfilled"
    return "conflicted"


def _build_project_record(
    item: dict[str, Any],
    *,
    plan: LegacyTablePlan,
    imported_at: datetime,
) -> tuple[int, dict[str, Any], datetime | None] | None:
    legacy_pk = item.get(plan.legacy_pk_field)
    if legacy_pk is None:
        return None

    try:
        legacy_project_id = int(legacy_pk)
    except Exception:
        return None

    record: dict[str, Any] = {
        "id": legacy_project_id,
        "prj_LegacyImportTS": imported_at,
    }

    for legacy_field, local_field in plan.field_map.items():
        if legacy_field not in item:
            continue
        # Keep local modification timestamps app-managed; preserve legacy creation.
        if local_field.endswith("_ModificationTS"):
            continue
        record[local_field] = _coerce_model_field(Project, local_field, item.get(legacy_field))

    if plan.legacy_created_field:
        created_dt = _normalize_datetime(_coerce_datetime(item.get(plan.legacy_created_field)))
        if created_dt is not None:
            record["prj_CreationTS"] = created_dt

    added_by = record.get("prj_AddedBy")
    if not (isinstance(added_by, str) and added_by.strip()):
        record["prj_AddedBy"] = "legacy_import"

    for ts_field in ("prj_CreationTS", "prj_ModificationTS"):
        if record.get(ts_field) is None:
            record.pop(ts_field, None)

    title = record.get("prj_ProjectTitle")
    if not (isinstance(title, str) and title.strip()):
        record["prj_ProjectTitle"] = f"Untitled (PRJ {legacy_project_id})"

    display_id = record.get("prj_PRJ_DisplayID")
    if not (isinstance(display_id, str) and display_id.strip()):
        record["prj_PRJ_DisplayID"] = f"MSPC{legacy_project_id:06d}"

    display_title = record.get("prj_PRJ_DisplayTitle")
    if not (isinstance(display_title, str) and display_title.strip()):
        record["prj_PRJ_DisplayTitle"] = (
            f"{record['prj_PRJ_DisplayID']} - {record['prj_ProjectTitle']}"
        )

    for bool_key in (
        "prj_Current_FLAG",
        "prj_Billing_ReadyToBill",
        "prj_PaymentReceived",
        "prj_RnD",
    ):
        if bool_key in record and record[bool_key] is None:
            record[bool_key] = False

    last_modified_raw = item.get(plan.legacy_modified_field)
    last_modified_dt = _normalize_datetime(_coerce_datetime(last_modified_raw))

    return legacy_project_id, record, last_modified_dt


def _build_experiment_record(
    item: dict[str, Any],
    *,
    plan: LegacyTablePlan,
    model_columns: set[str],
    imported_at: datetime,
) -> tuple[int, dict[str, Any], int | None, datetime | None] | None:
    legacy_pk = item.get(plan.legacy_pk_field)
    if legacy_pk is None:
        return None
    try:
        exp_id = int(legacy_pk)
    except Exception:
        return None

    record: dict[str, Any] = {
        "id": exp_id,
        "record_no": str(exp_id),
        "Experiment_LegacyImportTS": imported_at,
    }

    for legacy_field, local_field in plan.field_map.items():
        if legacy_field not in item:
            continue
        if local_field not in model_columns:
            continue
        # Keep local modification timestamps app-managed; preserve legacy creation.
        if local_field.endswith("_ModificationTS"):
            continue
        record[local_field] = _coerce_model_field(Experiment, local_field, item.get(legacy_field))

    if plan.legacy_created_field:
        created_dt = _normalize_datetime(_coerce_datetime(item.get(plan.legacy_created_field)))
        if created_dt is not None:
            record["Experiment_CreationTS"] = created_dt

    if "exp_LabelFLAG" in record and record["exp_LabelFLAG"] is None:
        record["exp_LabelFLAG"] = 0

    for bool_key in ("exp_DTT", "exp_IAA", "exp_Data_FLAG", "exp_exp2gene_FLAG"):
        if bool_key in record and record[bool_key] is None:
            record[bool_key] = False

    for ts_field in ("Experiment_CreationTS", "Experiment_ModificationTS"):
        if record.get(ts_field) is None:
            record.pop(ts_field, None)

    if "project_id" not in record:
        return None

    project_id = record.get("project_id")
    if project_id is None:
        record["project_id"] = None
        last_modified_dt = _normalize_datetime(_coerce_datetime(item.get(plan.legacy_modified_field)))
        return exp_id, record, None, last_modified_dt

    try:
        project_id_int = int(project_id)
    except Exception:
        return None

    if project_id_int <= 0:
        record["project_id"] = None
        last_modified_dt = _normalize_datetime(_coerce_datetime(item.get(plan.legacy_modified_field)))
        return exp_id, record, None, last_modified_dt

    record["project_id"] = project_id_int
    last_modified_dt = _normalize_datetime(_coerce_datetime(item.get(plan.legacy_modified_field)))
    return exp_id, record, project_id_int, last_modified_dt


def _apply_experiment_record(
    session,
    *,
    exp_id: int,
    record: dict[str, Any],
    dry_run: bool,
    backfill_missing: bool,
) -> str:
    existing = session.get(Experiment, exp_id)
    if existing is None:
        if not dry_run:
            session.add(Experiment(**record))
        return "inserted"

    if _can_update_imported(
        obj=existing,
        added_by_field="__legacy_added_by__",  # Experiment has no AddedBy column
        created_field="Experiment_CreationTS",
        modified_field="Experiment_ModificationTS",
        import_ts_field="Experiment_LegacyImportTS",
    ):
        if not dry_run:
            for key, value in record.items():
                if key == "id":
                    continue
                if hasattr(existing, key):
                    setattr(existing, key, value)
        return "updated"

    if not backfill_missing:
        return "conflicted"

    changed = False
    for key, value in record.items():
        if key in {"id", "Experiment_LegacyImportTS"}:
            continue
        if not hasattr(existing, key):
            continue
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue

        current = getattr(existing, key, None)
        missing_current = current is None
        if isinstance(current, str):
            missing_current = missing_current or not current.strip()

        if not missing_current:
            continue

        if not dry_run:
            setattr(existing, key, value)
        changed = True

    return "backfilled" if changed else "conflicted"


def _build_experiment_run_record(
    item: dict[str, Any],
    *,
    plan: LegacyTablePlan,
    model_columns: set[str],
) -> tuple[tuple[int, int, int, str], dict[str, Any]] | None:
    record: dict[str, Any] = {}
    for legacy_field, local_field in plan.field_map.items():
        if legacy_field not in item:
            continue
        if local_field not in model_columns:
            continue
        record[local_field] = _coerce_model_field(ExperimentRun, local_field, item.get(legacy_field))

    if plan.legacy_created_field and "ExperimentRun_CreationTS" in model_columns:
        created_dt = _normalize_datetime(_coerce_datetime(item.get(plan.legacy_created_field)))
        if created_dt is not None:
            record["ExperimentRun_CreationTS"] = created_dt

    exp_id = record.get("experiment_id")
    run_no = record.get("run_no")
    search_no = record.get("search_no")
    if exp_id is None or run_no is None or search_no is None:
        return None

    try:
        exp_id_int = int(exp_id)
        run_no_int = int(run_no)
        search_no_int = int(search_no)
    except Exception:
        return None

    record["experiment_id"] = exp_id_int
    record["run_no"] = run_no_int
    record["search_no"] = search_no_int

    if "label" in model_columns:
        label_value = record.get("label")
        if label_value is None:
            label_value = (
                item.get("label")
                or item.get("exprun_Label")
                or item.get("exprun_LabelFLAG")
                or item.get("exprun_LabelFlag")
            )
        record["label"] = _normalize_label_value(label_value)

    if "label_type" in model_columns and "label_type" not in record:
        label_type_value = item.get("exprun_LabelType") or item.get("label_type")
        if isinstance(label_type_value, str):
            label_type_value = label_type_value.strip() or None
        record["label_type"] = label_type_value

    for ts_field in ("ExperimentRun_CreationTS", "ExperimentRun_ModificationTS"):
        if record.get(ts_field) is None:
            record.pop(ts_field, None)

    label_key = record.get("label")
    label_norm = _normalize_label_value(label_key)
    record["label"] = label_norm
    return (exp_id_int, run_no_int, search_no_int, label_norm), record


def _apply_experiment_run_record(
    session,
    *,
    key: tuple[int, int, int, str],
    record: dict[str, Any],
    dry_run: bool,
    backfill_missing: bool,
) -> str:
    exp_id_int, run_no_int, search_no_int, label = key
    try:
        existing = (
            session.query(ExperimentRun)
            .filter(
                ExperimentRun.experiment_id == exp_id_int,
                ExperimentRun.run_no == run_no_int,
                ExperimentRun.search_no == search_no_int,
                ExperimentRun.label == label,
            )
            .one_or_none()
        )
    except MultipleResultsFound:
        return "conflicted"

    if existing is None:
        if session.get(Experiment, exp_id_int) is None:
            _ensure_placeholder_experiment(session, exp_id_int, project_id=None)
        if not dry_run:
            session.add(ExperimentRun(**record))
        return "inserted"

    if _can_update_imported(
        obj=existing,
        added_by_field="__legacy_added_by__",  # ExperimentRun has no AddedBy column
        created_field="ExperimentRun_CreationTS",
        modified_field="ExperimentRun_ModificationTS",
        import_ts_field="ExperimentRun_LegacyImportTS",
    ):
        if not dry_run:
            for key_name, value in record.items():
                if hasattr(existing, key_name):
                    setattr(existing, key_name, value)
        return "updated"

    if not backfill_missing:
        return "conflicted"

    changed = False
    for key_name, value in record.items():
        if key_name == "ExperimentRun_LegacyImportTS":
            continue
        if not hasattr(existing, key_name):
            continue
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue

        current = getattr(existing, key_name, None)
        missing_current = current is None
        if isinstance(current, str):
            missing_current = missing_current or not current.strip()

        if not missing_current:
            continue
        if not dry_run:
            setattr(existing, key_name, value)
        changed = True

    return "backfilled" if changed else "conflicted"


def _apply_project_record(
    session,
    *,
    project_id: int,
    record: dict[str, Any],
    dry_run: bool,
    backfill_missing: bool,
) -> str:
    existing = session.get(Project, project_id)
    if existing is None:
        if not dry_run:
            session.add(Project(**record))
        return "inserted"

    if _can_update_imported(
        obj=existing,
        added_by_field="prj_AddedBy",
        created_field="prj_CreationTS",
        modified_field="prj_ModificationTS",
        import_ts_field="prj_LegacyImportTS",
    ):
        if not dry_run:
            for key, value in record.items():
                if key == "id":
                    continue
                if hasattr(existing, key):
                    setattr(existing, key, value)
        return "updated"

    if not backfill_missing:
        return "conflicted"

    changed = False
    for key, value in record.items():
        if key in {"id", "prj_LegacyImportTS"}:
            continue
        if not hasattr(existing, key):
            continue
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue

        current = getattr(existing, key, None)
        missing_current = current is None
        if isinstance(current, str):
            missing_current = missing_current or not current.strip()
        if key.endswith("_AddedBy") and current == "legacy_import":
            missing_current = True

        if not missing_current:
            continue

        if not dry_run:
            setattr(existing, key, value)
        changed = True

    return "backfilled" if changed else "conflicted"


def sync_legacy_projects(
    *,
    legacy_url: str | None = None,
    mapping_path: str | Path | None = None,
    schema_path: str | Path | None = None,
    db_file_path: str | None = None,
    project_id: int | None = None,
    limit: int = 1000,
    max_pages: int | None = None,
    reset_cursor: bool = False,
    since: str | None = None,
    since_pk: int | None = None,
    dry_run: bool = False,
    backfill_missing: bool = False,
    collect_ids: bool = False,
    max_collected_ids: int = 200,
    dump_json: str | Path | None = None,
) -> dict[str, Any]:
    """Incrementally sync legacy iSPEC Projects into the local Project table."""

    resolved_mapping = Path(mapping_path).expanduser().resolve() if mapping_path else default_mapping_path()
    resolved_schema = Path(schema_path).expanduser().resolve() if schema_path else default_schema_path()
    base_url = _resolve_legacy_url(legacy_url=legacy_url, schema_path=resolved_schema)

    plan = _load_projects_plan(resolved_mapping)
    fields = _plan_fields_to_fetch(plan)
    expected_fields = set(fields)
    fields_mode: str | None = None

    if not db_file_path:
        db_file_path = (os.getenv("ISPEC_DB_PATH") or "").strip() or get_db_path()

    inserted = updated = backfilled = conflicted = duplicates_skipped = 0
    touched_ids: list[int] = []
    touched_set: set[int] = set()
    processed_ids: set[int] = set()
    pages = 0
    total_items = 0
    seen_cursors: set[tuple[str | None, int | None]] = set()
    dump_file, dump_dir = _resolve_legacy_dump_targets(dump_json)

    with get_session(file_path=db_file_path) as session:
        cursor_state: LegacySyncState | None = None
        if project_id is None:
            cursor_state = session.get(LegacySyncState, plan.legacy_table)
            if cursor_state is None:
                cursor_state = LegacySyncState(legacy_table=plan.legacy_table)
                session.add(cursor_state)
                session.flush()

        if cursor_state is not None and reset_cursor:
            cursor_state.since = None
            cursor_state.since_pk = None

        cursor_since_dt = cursor_state.since if cursor_state is not None else None
        cursor_since_pk = cursor_state.since_pk if cursor_state is not None else None

        if since:
            cursor_since_dt = _normalize_datetime(_coerce_datetime(since))
        if since_pk is not None:
            cursor_since_pk = int(since_pk)

        while True:
            pages += 1
            if max_pages is not None and pages > max_pages:
                break

            params: dict[str, Any] = {
                "modified_field": plan.legacy_modified_field,
                "limit": int(limit),
                "order_by": f"-{plan.legacy_modified_field},-{plan.legacy_pk_field}",
            }
            if cursor_since_dt is not None:
                params["since"] = _format_cursor_datetime(cursor_since_dt)
            if cursor_since_pk is not None:
                params["since_pk"] = int(cursor_since_pk)
            if project_id is not None:
                params["id"] = int(project_id)

            cursor_key = (
                params.get("since"),
                int(params["since_pk"]) if "since_pk" in params else None,
            )
            if cursor_key in seen_cursors:
                logger.warning(
                    "legacy sync cursor repeated; legacy endpoint may be ignoring since/since_pk (%s)",
                    cursor_key,
                )
                break
            seen_cursors.add(cursor_key)

            url = f"{base_url}/api/v2/legacy/tables/{plan.legacy_table}/rows"
            modes = [fields_mode] if fields_mode else ["repeat", "csv", "none"]
            threshold_missing = max(1, int(0.1 * len(expected_fields)))

            required_fields = [plan.legacy_pk_field, plan.legacy_modified_field]
            if plan.legacy_created_field:
                required_fields.append(plan.legacy_created_field)

            payload, fields_mode = _fetch_legacy_rows_best_effort(
                url=url,
                params=params,
                modes=modes,
                fields=list(fields),
                expected_fields=expected_fields,
                required_fields=required_fields,
                merge_key_fields=[plan.legacy_pk_field],
                threshold_missing=threshold_missing,
                log_label="legacy projects",
            )

            dump_path = _legacy_dump_path_for_request(
                dump_file,
                dump_dir,
                table=plan.legacy_table,
                page=pages,
                mode=fields_mode,
                id_value=project_id,
            )
            if dump_path is not None:
                if not _legacy_debug_requests_enabled():
                    request_url = _prepared_request_url(
                        url,
                        _legacy_params_with_fields(
                            params, fields=list(fields), mode=str(fields_mode)
                        ),
                    )
                    logger.info("legacy projects request_url mode=%s %s", fields_mode, request_url)
                _dump_legacy_payload(payload, path=dump_path)
                logger.info("legacy projects dumped payload to %s", dump_path)

            raw_items = payload.get("items") or payload.get("rows") or []
            items: list[dict[str, Any]] = list(raw_items)
            has_more_raw = payload.get("has_more")
            has_more = bool(has_more_raw) if has_more_raw is not None else len(items) >= int(limit)
            total_items += len(items)

            if not items:
                break

            last_modified_dt: datetime | None = None
            last_pk: int | None = None

            for item in items:
                if not isinstance(item, dict):
                    continue

                imported_at = _normalize_datetime(datetime.now(UTC)) or datetime.utcnow()
                built = _build_project_record(item, plan=plan, imported_at=imported_at)
                if built is None:
                    continue

                legacy_project_id, record, item_modified_dt = built
                if legacy_project_id in processed_ids:
                    duplicates_skipped += 1
                    last_modified_dt = item_modified_dt
                    last_pk = legacy_project_id
                    continue
                processed_ids.add(legacy_project_id)

                outcome = _apply_project_record(
                    session,
                    project_id=legacy_project_id,
                    record=record,
                    dry_run=dry_run,
                    backfill_missing=backfill_missing,
                )
                if outcome == "inserted":
                    inserted += 1
                elif outcome == "updated":
                    updated += 1
                elif outcome == "backfilled":
                    backfilled += 1
                else:
                    conflicted += 1
                if collect_ids and outcome != "conflicted":
                    if legacy_project_id not in touched_set and len(touched_ids) < max_collected_ids:
                        touched_set.add(legacy_project_id)
                        touched_ids.append(legacy_project_id)

                last_modified_dt = item_modified_dt
                last_pk = legacy_project_id

            if project_id is None and not dry_run and cursor_state is not None:
                next_since = payload.get("next_since")
                next_since_pk = payload.get("next_since_pk")

                if isinstance(next_since, str) and next_since.strip():
                    cursor_since_dt = _normalize_datetime(_coerce_datetime(next_since))
                else:
                    cursor_since_dt = last_modified_dt

                cursor_since_pk = int(next_since_pk) if next_since_pk is not None else last_pk

                cursor_state.since = cursor_since_dt
                cursor_state.since_pk = cursor_since_pk

                session.flush()

            if not has_more:
                break
            if project_id is not None:
                break

        if dry_run:
            session.rollback()

    result: dict[str, Any] = {
        "pages": pages,
        "items": total_items,
        "inserted": inserted,
        "updated": updated,
        "backfilled": backfilled,
        "conflicted": conflicted,
        "duplicates_skipped": duplicates_skipped,
    }
    if collect_ids:
        result["touched_ids"] = touched_ids
    return result


def sync_legacy_people(
    *,
    legacy_url: str | None = None,
    mapping_path: str | Path | None = None,
    schema_path: str | Path | None = None,
    db_file_path: str | None = None,
    person_id: int | None = None,
    limit: int = 1000,
    max_pages: int | None = None,
    reset_cursor: bool = False,
    since: str | None = None,
    since_pk: int | None = None,
    dry_run: bool = False,
    backfill_missing: bool = False,
    collect_ids: bool = False,
    max_collected_ids: int = 200,
    dump_json: str | Path | None = None,
) -> dict[str, Any]:
    """Incrementally sync legacy iSPEC People into the local Person table."""

    resolved_mapping = (
        Path(mapping_path).expanduser().resolve() if mapping_path else default_mapping_path()
    )
    resolved_schema = (
        Path(schema_path).expanduser().resolve() if schema_path else default_schema_path()
    )
    base_url = _resolve_legacy_url(legacy_url=legacy_url, schema_path=resolved_schema)

    plan = _load_table_plan(resolved_mapping, legacy_table="iSPEC_People")
    fields = _plan_fields_to_fetch(plan)
    model_columns = set(Person.__table__.columns.keys())
    expected_fields: set[str] = {plan.legacy_pk_field, plan.legacy_modified_field}
    if plan.legacy_created_field:
        expected_fields.add(plan.legacy_created_field)
    for legacy_field, local_field in plan.field_map.items():
        if local_field in model_columns:
            expected_fields.add(legacy_field)

    required_fields: list[str] = [plan.legacy_pk_field, plan.legacy_modified_field]
    if plan.legacy_created_field:
        required_fields.append(plan.legacy_created_field)

    if not db_file_path:
        db_file_path = (os.getenv("ISPEC_DB_PATH") or "").strip() or get_db_path()

    inserted = updated = backfilled = conflicted = duplicates_skipped = 0
    touched_ids: list[int] = []
    touched_set: set[int] = set()
    processed_ids: set[int] = set()
    pages = 0
    total_items = 0
    seen_cursors: set[tuple[str | None, int | None]] = set()
    fields_mode: str | None = None
    dump_file, dump_dir = _resolve_legacy_dump_targets(dump_json)

    with get_session(file_path=db_file_path) as session:
        cursor_state: LegacySyncState | None = None
        if person_id is None:
            cursor_state = session.get(LegacySyncState, plan.legacy_table)
            if cursor_state is None:
                cursor_state = LegacySyncState(legacy_table=plan.legacy_table)
                session.add(cursor_state)
                session.flush()

        if cursor_state is not None and reset_cursor:
            cursor_state.since = None
            cursor_state.since_pk = None

        cursor_since_dt = cursor_state.since if cursor_state is not None else None
        cursor_since_pk = cursor_state.since_pk if cursor_state is not None else None

        if since:
            cursor_since_dt = _normalize_datetime(_coerce_datetime(since))
        if since_pk is not None:
            cursor_since_pk = int(since_pk)

        url = f"{base_url}/api/v2/legacy/tables/{plan.legacy_table}/rows"

        while True:
            pages += 1
            if max_pages is not None and pages > max_pages:
                break

            params: dict[str, Any] = {
                "pk_field": plan.legacy_pk_field,
                "modified_field": plan.legacy_modified_field,
                "limit": int(limit),
                "order_by": f"-{plan.legacy_modified_field},-{plan.legacy_pk_field}",
            }
            if cursor_since_dt is not None:
                params["since"] = _format_cursor_datetime(cursor_since_dt)
            if cursor_since_pk is not None:
                params["since_pk"] = int(cursor_since_pk)
            if person_id is not None:
                params["id"] = int(person_id)

            cursor_key = (
                params.get("since"),
                int(params["since_pk"]) if "since_pk" in params else None,
            )
            if cursor_key in seen_cursors:
                logger.warning(
                    "legacy sync cursor repeated; legacy endpoint may be ignoring since/since_pk (%s)",
                    cursor_key,
                )
                break
            seen_cursors.add(cursor_key)

            modes = [fields_mode] if fields_mode else ["repeat", "csv", "none"]
            threshold_missing = max(1, int(0.1 * len(expected_fields)))

            payload, fields_mode = _fetch_legacy_rows_best_effort(
                url=url,
                params=params,
                modes=modes,
                fields=list(fields),
                expected_fields=expected_fields,
                required_fields=required_fields,
                merge_key_fields=[plan.legacy_pk_field],
                threshold_missing=threshold_missing,
                log_label="legacy people",
            )

            dump_path = _legacy_dump_path_for_request(
                dump_file,
                dump_dir,
                table=plan.legacy_table,
                page=pages,
                mode=fields_mode,
                id_value=person_id,
            )
            if dump_path is not None:
                if not _legacy_debug_requests_enabled():
                    request_url = _prepared_request_url(
                        url,
                        _legacy_params_with_fields(
                            params, fields=list(fields), mode=str(fields_mode)
                        ),
                    )
                    logger.info("legacy people request_url mode=%s %s", fields_mode, request_url)
                _dump_legacy_payload(payload, path=dump_path)
                logger.info("legacy people dumped payload to %s", dump_path)

            raw_items = payload.get("items") or payload.get("rows") or []
            items: list[dict[str, Any]] = list(raw_items)
            has_more_raw = payload.get("has_more")
            has_more = bool(has_more_raw) if has_more_raw is not None else len(items) >= int(limit)
            total_items += len(items)

            if not items:
                break

            last_modified_dt: datetime | None = None
            last_pk: int | None = None

            for item in items:
                if not isinstance(item, dict):
                    continue

                imported_at = _normalize_datetime(datetime.now(UTC)) or datetime.utcnow()
                built = _build_person_record(item, plan=plan, imported_at=imported_at)
                if built is None:
                    continue

                legacy_person_id, record, item_modified_dt = built
                if legacy_person_id in processed_ids:
                    duplicates_skipped += 1
                    last_modified_dt = item_modified_dt
                    last_pk = legacy_person_id
                    continue
                processed_ids.add(legacy_person_id)
                outcome = _apply_person_record(
                    session,
                    legacy_person_id=legacy_person_id,
                    record=record,
                    dry_run=dry_run,
                    backfill_missing=backfill_missing,
                )
                if outcome == "inserted":
                    inserted += 1
                elif outcome == "updated":
                    updated += 1
                elif outcome == "backfilled":
                    backfilled += 1
                else:
                    conflicted += 1
                if collect_ids and outcome != "conflicted":
                    if legacy_person_id not in touched_set and len(touched_ids) < max_collected_ids:
                        touched_set.add(legacy_person_id)
                        touched_ids.append(legacy_person_id)

                last_modified_dt = item_modified_dt
                last_pk = legacy_person_id

            if person_id is None and not dry_run and cursor_state is not None:
                next_since = payload.get("next_since")
                next_since_pk = payload.get("next_since_pk")

                if isinstance(next_since, str) and next_since.strip():
                    cursor_since_dt = _normalize_datetime(_coerce_datetime(next_since))
                else:
                    cursor_since_dt = last_modified_dt

                cursor_since_pk = int(next_since_pk) if next_since_pk is not None else last_pk

                cursor_state.since = cursor_since_dt
                cursor_state.since_pk = cursor_since_pk

                session.flush()

            if not has_more:
                break
            if person_id is not None:
                break

        if dry_run:
            session.rollback()

    result: dict[str, Any] = {
        "pages": pages,
        "items": total_items,
        "inserted": inserted,
        "updated": updated,
        "backfilled": backfilled,
        "conflicted": conflicted,
        "duplicates_skipped": duplicates_skipped,
    }
    if collect_ids:
        result["touched_ids"] = touched_ids
    return result


def sync_legacy_project_comments(
    *,
    legacy_url: str | None = None,
    schema_path: str | Path | None = None,
    db_file_path: str | None = None,
    project_id: int,
    limit: int = 5000,
    dry_run: bool = False,
    dump_json: str | Path | None = None,
) -> dict[str, Any]:
    """Sync legacy iSPEC ProjectHistory rows into the local ProjectComment table."""

    resolved_schema = Path(schema_path).expanduser().resolve() if schema_path else default_schema_path()
    base_url = _resolve_legacy_url(legacy_url=legacy_url, schema_path=resolved_schema)

    if not db_file_path:
        db_file_path = (os.getenv("ISPEC_DB_PATH") or "").strip() or get_db_path()

    legacy_table = "iSPEC_ProjectHistory"
    pk_field = "prh_PRJRecNo"
    modified_field = "prh_ModificationTS"
    fields = [
        pk_field,
        modified_field,
        "prh_CreationTS",
        "prh_AddedBy",
        "prh_CommentType",
        "prh_Comment",
    ]
    expected_fields = set(fields)
    required_fields = [pk_field, modified_field]

    url = f"{base_url}/api/v2/legacy/tables/{legacy_table}/rows"
    params: dict[str, Any] = {
        "pk_field": pk_field,
        "modified_field": modified_field,
        "id": int(project_id),
        "limit": int(limit),
        "order_by": f"-{modified_field}",
    }

    threshold_missing = max(1, int(0.1 * len(expected_fields)))
    payload, fields_mode = _fetch_legacy_rows_best_effort(
        url=url,
        params=params,
        modes=["repeat", "csv", "none"],
        fields=list(fields),
        expected_fields=expected_fields,
        required_fields=required_fields,
        merge_key_fields=[pk_field, modified_field, "prh_CreationTS"],
        threshold_missing=threshold_missing,
        log_label="legacy project history",
    )

    dump_file, dump_dir = _resolve_legacy_dump_targets(dump_json)
    dump_path = _legacy_dump_path_for_request(
        dump_file,
        dump_dir,
        table=legacy_table,
        page=1,
        mode=fields_mode,
        id_value=project_id,
    )
    if dump_path is not None:
        if not _legacy_debug_requests_enabled():
            request_url = _prepared_request_url(
                url,
                _legacy_params_with_fields(params, fields=list(fields), mode=str(fields_mode)),
            )
            logger.info("legacy project history request_url mode=%s %s", fields_mode, request_url)
        _dump_legacy_payload(payload, path=dump_path)
        logger.info("legacy project history dumped payload to %s", dump_path)

    raw_items = payload.get("items") or payload.get("rows") or []
    items: list[dict[str, Any]] = [i for i in raw_items if isinstance(i, dict)]

    inserted = updated = conflicted = skipped_blank = skipped_other_project = 0

    imported_at = _normalize_datetime(datetime.now(UTC)) or datetime.utcnow()

    def comment_key_from_record(record: dict[str, Any]) -> tuple[Any, ...]:
        created_at = _normalize_datetime(record.get("com_CreationTS"))
        created_key = created_at.isoformat() if created_at else None
        return (
            int(record.get("project_id")),
            int(record.get("person_id", 0)),
            created_key,
            (record.get("com_CommentType") or "").strip(),
            (record.get("com_AddedBy") or "").strip(),
        )

    def comment_key_from_obj(comment: ProjectComment) -> tuple[Any, ...]:
        created_at = _normalize_datetime(getattr(comment, "com_CreationTS", None))
        created_key = created_at.isoformat() if created_at else None
        return (
            int(comment.project_id),
            int(comment.person_id),
            created_key,
            (comment.com_CommentType or "").strip(),
            (comment.com_AddedBy or "").strip(),
        )

    with get_session(file_path=db_file_path) as session:
        _ensure_placeholder_project(session, int(project_id))
        _ensure_system_person(session, imported_at=imported_at)

        existing_comments = (
            session.query(ProjectComment)
            .filter(ProjectComment.project_id == int(project_id))
            .filter(ProjectComment.person_id == 0)
            .all()
        )
        comment_index: dict[tuple[Any, ...], ProjectComment] = {}
        for comment in existing_comments:
            comment_index[comment_key_from_obj(comment)] = comment

        processed_keys: set[tuple[Any, ...]] = set()
        for item in items:
            legacy_project_id = item.get(pk_field)
            if legacy_project_id is None:
                continue
            try:
                legacy_project_id_int = int(legacy_project_id)
            except Exception:
                continue
            if legacy_project_id_int != int(project_id):
                skipped_other_project += 1
                continue

            comment = item.get("prh_Comment")
            if comment is None or (isinstance(comment, str) and not comment.strip()):
                skipped_blank += 1
                continue

            created_dt = _normalize_datetime(_coerce_datetime(item.get("prh_CreationTS")))
            if created_dt is None:
                created_dt = _normalize_datetime(_coerce_datetime(item.get(modified_field)))

            record: dict[str, Any] = {
                "project_id": legacy_project_id_int,
                "person_id": 0,
                "com_LegacyImportTS": imported_at,
                "com_Comment": comment,
                "com_CommentType": item.get("prh_CommentType"),
                "com_AddedBy": item.get("prh_AddedBy"),
                "com_ModificationTS": imported_at,
            }
            if created_dt is not None:
                record["com_CreationTS"] = created_dt

            key = comment_key_from_record(record)
            if key in processed_keys:
                continue
            processed_keys.add(key)
            existing = comment_index.get(key)
            if existing is None:
                comment_obj = ProjectComment(**record)
                if not dry_run:
                    session.add(comment_obj)
                comment_index[key] = comment_obj
                inserted += 1
                continue

            if _can_update_imported_comment(existing):
                if not dry_run:
                    for field in ("com_Comment", "com_CommentType", "com_AddedBy", "com_LegacyImportTS"):
                        if field in record:
                            setattr(existing, field, record[field])
                    existing.com_ModificationTS = imported_at
                updated += 1
            else:
                conflicted += 1

        if dry_run:
            session.rollback()

    return {
        "items": len(items),
        "inserted": inserted,
        "updated": updated,
        "conflicted": conflicted,
        "skipped_blank": skipped_blank,
        "skipped_other_project": skipped_other_project,
    }


def sync_legacy_experiments(
    *,
    legacy_url: str | None = None,
    mapping_path: str | Path | None = None,
    schema_path: str | Path | None = None,
    db_file_path: str | None = None,
    experiment_id: int | None = None,
    limit: int = 1000,
    max_pages: int | None = None,
    reset_cursor: bool = False,
    since: str | None = None,
    since_pk: int | None = None,
    dry_run: bool = False,
    backfill_missing: bool = False,
    collect_ids: bool = False,
    max_collected_ids: int = 200,
    dump_json: str | Path | None = None,
) -> dict[str, Any]:
    """Incrementally sync legacy iSPEC Experiments into the local Experiment table."""

    resolved_mapping = (
        Path(mapping_path).expanduser().resolve() if mapping_path else default_mapping_path()
    )
    resolved_schema = (
        Path(schema_path).expanduser().resolve() if schema_path else default_schema_path()
    )
    base_url = _resolve_legacy_url(legacy_url=legacy_url, schema_path=resolved_schema)

    plan = _load_table_plan(resolved_mapping, legacy_table="iSPEC_Experiments")
    fields = _plan_fields_to_fetch(plan)
    model_columns = set(Experiment.__table__.columns.keys())
    expected_fields: set[str] = {plan.legacy_pk_field, plan.legacy_modified_field}
    if plan.legacy_created_field:
        expected_fields.add(plan.legacy_created_field)
    for legacy_field, local_field in plan.field_map.items():
        if local_field in model_columns:
            expected_fields.add(legacy_field)

    required_fields: list[str] = [plan.legacy_pk_field, plan.legacy_modified_field]
    if plan.legacy_created_field:
        required_fields.append(plan.legacy_created_field)
    project_legacy_field = next(
        (legacy for legacy, local in plan.field_map.items() if local == "project_id"),
        None,
    )
    if project_legacy_field and project_legacy_field not in required_fields:
        required_fields.append(project_legacy_field)

    if not db_file_path:
        db_file_path = (os.getenv("ISPEC_DB_PATH") or "").strip() or get_db_path()

    inserted = updated = backfilled = conflicted = duplicates_skipped = 0
    touched_ids: list[int] = []
    touched_set: set[int] = set()
    processed_ids: set[int] = set()
    pages = 0
    total_items = 0
    seen_cursors: set[tuple[str | None, int | None]] = set()
    fields_mode: str | None = None
    dump_file, dump_dir = _resolve_legacy_dump_targets(dump_json)

    url = f"{base_url}/api/v2/legacy/tables/{plan.legacy_table}/rows"
    with get_session(file_path=db_file_path) as session:
        cursor_state: LegacySyncState | None = None
        if experiment_id is None:
            cursor_state = session.get(LegacySyncState, plan.legacy_table)
            if cursor_state is None:
                cursor_state = LegacySyncState(legacy_table=plan.legacy_table)
                session.add(cursor_state)
                session.flush()

        if cursor_state is not None and reset_cursor:
            cursor_state.since = None
            cursor_state.since_pk = None

        cursor_since_dt = cursor_state.since if cursor_state is not None else None
        cursor_since_pk = cursor_state.since_pk if cursor_state is not None else None

        if since:
            cursor_since_dt = _normalize_datetime(_coerce_datetime(since))
        if since_pk is not None:
            cursor_since_pk = int(since_pk)

        while True:
            pages += 1
            if max_pages is not None and pages > max_pages:
                break

            params: dict[str, Any] = {
                "pk_field": plan.legacy_pk_field,
                "modified_field": plan.legacy_modified_field,
                "limit": int(limit),
                "order_by": f"-{plan.legacy_modified_field},-{plan.legacy_pk_field}",
            }
            if cursor_since_dt is not None:
                params["since"] = _format_cursor_datetime(cursor_since_dt)
            if cursor_since_pk is not None:
                params["since_pk"] = int(cursor_since_pk)
            if experiment_id is not None:
                params["id"] = int(experiment_id)

            cursor_key = (
                params.get("since"),
                int(params["since_pk"]) if "since_pk" in params else None,
            )
            if cursor_key in seen_cursors:
                logger.warning(
                    "legacy sync cursor repeated; legacy endpoint may be ignoring since/since_pk (%s)",
                    cursor_key,
                )
                break
            seen_cursors.add(cursor_key)

            modes = [fields_mode] if fields_mode else ["repeat", "csv", "none"]
            threshold_missing = max(1, int(0.1 * len(expected_fields)))
            payload, fields_mode = _fetch_legacy_rows_best_effort(
                url=url,
                params=params,
                modes=modes,
                fields=list(fields),
                expected_fields=expected_fields,
                required_fields=required_fields,
                merge_key_fields=[plan.legacy_pk_field],
                threshold_missing=threshold_missing,
                log_label="legacy experiments",
            )

            dump_path = _legacy_dump_path_for_request(
                dump_file,
                dump_dir,
                table=plan.legacy_table,
                page=pages,
                mode=fields_mode,
                id_value=experiment_id,
            )
            if dump_path is not None:
                if not _legacy_debug_requests_enabled():
                    request_url = _prepared_request_url(
                        url,
                        _legacy_params_with_fields(
                            params, fields=list(fields), mode=str(fields_mode)
                        ),
                    )
                    logger.info(
                        "legacy experiments request_url mode=%s %s", fields_mode, request_url
                    )
                _dump_legacy_payload(payload, path=dump_path)
                logger.info("legacy experiments dumped payload to %s", dump_path)

            raw_items = payload.get("items") or payload.get("rows") or []
            items: list[dict[str, Any]] = list(raw_items)
            has_more_raw = payload.get("has_more")
            has_more = bool(has_more_raw) if has_more_raw is not None else len(items) >= int(limit)
            total_items += len(items)

            if not items:
                break

            last_modified_dt: datetime | None = None
            last_pk: int | None = None

            for item in items:
                if not isinstance(item, dict):
                    continue

                imported_at = _normalize_datetime(datetime.now(UTC)) or datetime.utcnow()
                built = _build_experiment_record(
                    item,
                    plan=plan,
                    model_columns=model_columns,
                    imported_at=imported_at,
                )
                if built is None:
                    conflicted += 1
                    continue

                exp_id, record, project_id_int, item_modified_dt = built
                if exp_id in processed_ids:
                    duplicates_skipped += 1
                    last_modified_dt = item_modified_dt
                    last_pk = exp_id
                    continue
                processed_ids.add(exp_id)
                if project_id_int is not None:
                    _ensure_placeholder_project(session, project_id_int)

                outcome = _apply_experiment_record(
                    session,
                    exp_id=exp_id,
                    record=record,
                    dry_run=dry_run,
                    backfill_missing=backfill_missing,
                )
                if outcome == "inserted":
                    inserted += 1
                elif outcome == "updated":
                    updated += 1
                elif outcome == "backfilled":
                    backfilled += 1
                else:
                    conflicted += 1
                if collect_ids and outcome != "conflicted":
                    if exp_id not in touched_set and len(touched_ids) < max_collected_ids:
                        touched_set.add(exp_id)
                        touched_ids.append(exp_id)

                last_modified_dt = item_modified_dt
                last_pk = exp_id

            if experiment_id is None and not dry_run and cursor_state is not None:
                next_since = payload.get("next_since")
                next_since_pk = payload.get("next_since_pk")

                if isinstance(next_since, str) and next_since.strip():
                    cursor_since_dt = _normalize_datetime(_coerce_datetime(next_since))
                else:
                    cursor_since_dt = last_modified_dt

                cursor_since_pk = int(next_since_pk) if next_since_pk is not None else last_pk

                cursor_state.since = cursor_since_dt
                cursor_state.since_pk = cursor_since_pk

                session.flush()

            if not has_more:
                break
            if experiment_id is not None:
                break

        if dry_run:
            session.rollback()

    result: dict[str, Any] = {
        "pages": pages,
        "items": total_items,
        "inserted": inserted,
        "updated": updated,
        "backfilled": backfilled,
        "conflicted": conflicted,
        "duplicates_skipped": duplicates_skipped,
    }
    if collect_ids:
        result["touched_ids"] = touched_ids
    return result


def sync_legacy_experiment_runs(
    *,
    legacy_url: str | None = None,
    mapping_path: str | Path | None = None,
    schema_path: str | Path | None = None,
    db_file_path: str | None = None,
    experiment_id: int,
    limit: int = 5000,
    dry_run: bool = False,
    backfill_missing: bool = False,
    dump_json: str | Path | None = None,
) -> dict[str, Any]:
    """Sync legacy iSPEC ExperimentRuns for a specific experiment."""

    resolved_mapping = (
        Path(mapping_path).expanduser().resolve() if mapping_path else default_mapping_path()
    )
    resolved_schema = (
        Path(schema_path).expanduser().resolve() if schema_path else default_schema_path()
    )
    base_url = _resolve_legacy_url(legacy_url=legacy_url, schema_path=resolved_schema)

    plan = _load_table_plan(resolved_mapping, legacy_table="iSPEC_ExperimentRuns")
    fields = _plan_fields_to_fetch(plan)
    model_columns = set(ExperimentRun.__table__.columns.keys())
    expected_fields: set[str] = {plan.legacy_pk_field, plan.legacy_modified_field}
    if plan.legacy_created_field:
        expected_fields.add(plan.legacy_created_field)
    for legacy_field, local_field in plan.field_map.items():
        if local_field in model_columns:
            expected_fields.add(legacy_field)

    required_fields: list[str] = [plan.legacy_pk_field, plan.legacy_modified_field]
    if plan.legacy_created_field:
        required_fields.append(plan.legacy_created_field)
    merge_key_fields: list[str] = []
    for required_local in ("experiment_id", "run_no", "search_no", "label"):
        legacy_field = next(
            (legacy for legacy, local in plan.field_map.items() if local == required_local),
            None,
        )
        if legacy_field and legacy_field not in required_fields:
            required_fields.append(legacy_field)
        if legacy_field and legacy_field not in merge_key_fields:
            merge_key_fields.append(legacy_field)

    if not db_file_path:
        db_file_path = (os.getenv("ISPEC_DB_PATH") or "").strip() or get_db_path()

    inserted = updated = backfilled = conflicted = 0
    dump_file, dump_dir = _resolve_legacy_dump_targets(dump_json)

    url = f"{base_url}/api/v2/legacy/tables/{plan.legacy_table}/rows"
    order_fields = [plan.legacy_modified_field]
    for required_local in ("run_no", "search_no"):
        legacy_field = next(
            (legacy for legacy, local in plan.field_map.items() if local == required_local),
            None,
        )
        if legacy_field and legacy_field not in order_fields:
            order_fields.append(legacy_field)
    params: dict[str, Any] = {
        "pk_field": plan.legacy_pk_field,
        "modified_field": plan.legacy_modified_field,
        "limit": int(limit),
        "id": int(experiment_id),
        "order_by": ",".join(f"-{field}" for field in order_fields if field),
    }

    threshold_missing = max(1, int(0.1 * len(expected_fields)))
    payload, fields_mode = _fetch_legacy_rows_best_effort(
        url=url,
        params=params,
        modes=["repeat", "csv", "none"],
        fields=list(fields),
        expected_fields=expected_fields,
        required_fields=required_fields,
        merge_key_fields=merge_key_fields,
        threshold_missing=threshold_missing,
        log_label="legacy experiment runs",
    )

    dump_path = _legacy_dump_path_for_request(
        dump_file,
        dump_dir,
        table=plan.legacy_table,
        page=1,
        mode=fields_mode,
        id_value=experiment_id,
    )
    if dump_path is not None:
        if not _legacy_debug_requests_enabled():
            request_url = _prepared_request_url(
                url,
                _legacy_params_with_fields(params, fields=list(fields), mode=str(fields_mode)),
            )
            logger.info("legacy experiment runs request_url mode=%s %s", fields_mode, request_url)
        _dump_legacy_payload(payload, path=dump_path)
        logger.info("legacy experiment runs dumped payload to %s", dump_path)
    items = list(payload.get("items") or payload.get("rows") or [])

    with get_session(file_path=db_file_path) as session:
        for item in items:
            if not isinstance(item, dict):
                continue

            built = _build_experiment_run_record(item, plan=plan, model_columns=model_columns)
            if built is None:
                conflicted += 1
                continue

            run_key, record = built
            imported_at = _normalize_datetime(datetime.now(UTC)) or datetime.utcnow()
            record["ExperimentRun_LegacyImportTS"] = imported_at
            outcome = _apply_experiment_run_record(
                session,
                key=run_key,
                record=record,
                dry_run=dry_run,
                backfill_missing=backfill_missing,
            )
            if outcome == "inserted":
                inserted += 1
            elif outcome == "updated":
                updated += 1
            elif outcome == "backfilled":
                backfilled += 1
            else:
                conflicted += 1

        if dry_run:
            session.rollback()

    return {
        "inserted": inserted,
        "updated": updated,
        "backfilled": backfilled,
        "conflicted": conflicted,
    }
