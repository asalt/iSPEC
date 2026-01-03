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

from ispec.db.connect import get_db_path, get_session
from ispec.db.models import LegacySyncState, Project
from ispec.logging import get_logger

logger = get_logger(__file__)


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
    return modified_at <= imported_at


def _legacy_headers() -> dict[str, str]:
    api_key = (os.getenv("ISPEC_LEGACY_API_KEY") or "").strip()
    if not api_key:
        return {}
    return {"X-API-Key": api_key}


def _load_ispec_conf(path: Path) -> dict[str, str]:
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


def _load_projects_plan(mapping_path: Path) -> LegacyTablePlan:
    mapping = _load_json(mapping_path)
    table = mapping.get("tables", {}).get("iSPEC_Projects")
    if not isinstance(table, dict):
        raise KeyError("legacy mapping missing tables.iSPEC_Projects")

    pk = table.get("pk", {})
    if not isinstance(pk, dict):
        raise KeyError("legacy mapping missing pk section for iSPEC_Projects")

    field_map = table.get("field_map", {})
    if not isinstance(field_map, dict):
        raise KeyError("legacy mapping missing field_map for iSPEC_Projects")

    modified = table.get("modified_ts")
    if not isinstance(modified, str) or not modified.strip():
        raise KeyError("legacy mapping missing modified_ts for iSPEC_Projects")

    created = table.get("created_ts")
    created_field = created if isinstance(created, str) and created.strip() else None

    pk_field = pk.get("legacy")
    if not isinstance(pk_field, str) or not pk_field.strip():
        raise KeyError("legacy mapping missing pk.legacy for iSPEC_Projects")

    return LegacyTablePlan(
        legacy_table="iSPEC_Projects",
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


def _projects_fields_to_fetch(plan: LegacyTablePlan) -> list[str]:
    fields: list[str] = [plan.legacy_pk_field, plan.legacy_modified_field]
    if plan.legacy_created_field:
        fields.append(plan.legacy_created_field)
    for field in plan.field_map:
        if field not in fields:
            fields.append(field)
    return fields


def _expected_field_stats(
    item: dict[str, Any], expected_fields: set[str]
) -> tuple[int, list[str]]:
    """Return (expected_fields_present, missing_expected_fields)."""

    if not expected_fields:
        return 0, []
    present = expected_fields.intersection(item.keys())
    missing = sorted(expected_fields - present)
    return len(present), missing


def _coerce_project_field(field: str, value: Any) -> Any:
    column = Project.__table__.columns.get(field)
    if column is None:
        return value
    if value is None:
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
) -> dict[str, int]:
    """Incrementally sync legacy iSPEC Projects into the local Project table."""

    resolved_mapping = Path(mapping_path).expanduser().resolve() if mapping_path else default_mapping_path()
    resolved_schema = Path(schema_path).expanduser().resolve() if schema_path else default_schema_path()
    base_url = _resolve_legacy_url(legacy_url=legacy_url, schema_path=resolved_schema)

    plan = _load_projects_plan(resolved_mapping)
    fields = _projects_fields_to_fetch(plan)
    expected_fields = set(fields)
    fields_mode: str | None = None

    if not db_file_path:
        db_file_path = (os.getenv("ISPEC_DB_PATH") or "").strip() or get_db_path()

    imported_at = _normalize_datetime(datetime.now(UTC))
    if imported_at is None:  # pragma: no cover - defensive
        imported_at = datetime.utcnow()

    inserted = updated = backfilled = conflicted = 0
    pages = 0
    total_items = 0
    seen_cursors: set[tuple[str | None, int | None]] = set()

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
                "order_by": f"{plan.legacy_modified_field},{plan.legacy_pk_field}",
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
            payload: dict[str, Any] | None = None
            attempted_modes: list[str] = []
            modes = [fields_mode] if fields_mode else ["repeat", "csv", "none"]
            threshold_missing = max(5, int(0.1 * len(expected_fields)))

            best_payload: dict[str, Any] | None = None
            best_mode: str | None = None
            best_present = -1
            best_missing: list[str] = []

            for mode in modes:
                if mode is None:
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
                    "legacy sync fetch: %s mode=%s params=%s", url, mode, params_with_fields
                )

                resp = requests.get(
                    url,
                    params=params_with_fields,
                    headers=_legacy_headers(),
                    auth=_legacy_basic_auth(),
                    timeout=30,
                )
                resp.raise_for_status()
                candidate = resp.json()

                raw_items = candidate.get("items") or candidate.get("rows") or []
                items: list[dict[str, Any]] = list(raw_items)
                if not items:
                    payload = candidate
                    fields_mode = mode
                    best_payload = candidate
                    best_mode = mode
                    best_present = 0
                    best_missing = []
                    break

                present, missing = _expected_field_stats(items[0], expected_fields)
                missing_count = len(missing)
                keys_count = len(items[0].keys())

                logger.info(
                    "legacy sync fetch result: mode=%s keys=%d expected_present=%d/%d",
                    mode,
                    keys_count,
                    present,
                    len(expected_fields),
                )

                if present > best_present:
                    best_payload = candidate
                    best_mode = mode
                    best_present = present
                    best_missing = missing

                if missing_count <= threshold_missing:
                    payload = candidate
                    if fields_mode != mode:
                        logger.info(
                            "legacy sync selected fields mode=%s (missing=%d/%d)",
                            mode,
                            missing_count,
                            len(expected_fields),
                        )
                    fields_mode = mode
                    break

                sample_missing = ", ".join(missing[:10])
                logger.warning(
                    "legacy rows missing %d/%d expected fields (mode=%s); sample_missing=%s",
                    missing_count,
                    len(expected_fields),
                    mode,
                    sample_missing,
                )

            if payload is None:
                if best_payload is None or best_mode is None:
                    raise RuntimeError(
                        f"legacy sync fetch failed after modes={attempted_modes}"
                    )
                missing_count = len(expected_fields) - best_present
                sample_missing = ", ".join(best_missing[:10])
                logger.warning(
                    "legacy rows missing %d/%d expected fields even in best mode=%s; using best anyway (sample_missing=%s)",
                    missing_count,
                    len(expected_fields),
                    best_mode,
                    sample_missing,
                )
                payload = best_payload
                fields_mode = best_mode

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
                legacy_pk = item.get(plan.legacy_pk_field)
                if legacy_pk is None:
                    continue
                try:
                    project_id = int(legacy_pk)
                except Exception:
                    continue

                record: dict[str, Any] = {"id": project_id, "prj_LegacyImportTS": imported_at}

                for legacy_field, local_field in plan.field_map.items():
                    if legacy_field not in item:
                        continue
                    coerced = _coerce_project_field(local_field, item.get(legacy_field))
                    record[local_field] = coerced

                added_by = record.get("prj_AddedBy")
                if not (isinstance(added_by, str) and added_by.strip()):
                    record["prj_AddedBy"] = "legacy_import"

                for ts_field in ("prj_CreationTS", "prj_ModificationTS"):
                    if record.get(ts_field) is None:
                        record.pop(ts_field, None)

                title = record.get("prj_ProjectTitle")
                if not (isinstance(title, str) and title.strip()):
                    record["prj_ProjectTitle"] = f"Untitled (PRJ {project_id})"

                display_id = record.get("prj_PRJ_DisplayID")
                if not (isinstance(display_id, str) and display_id.strip()):
                    record["prj_PRJ_DisplayID"] = f"MSPC{project_id:06d}"

                display_title = record.get("prj_PRJ_DisplayTitle")
                if not (isinstance(display_title, str) and display_title.strip()):
                    record["prj_PRJ_DisplayTitle"] = (
                        f"{record['prj_PRJ_DisplayID']} - {record['prj_ProjectTitle']}"
                    )

                for bool_key in ("prj_Current_FLAG", "prj_Billing_ReadyToBill", "prj_PaymentReceived", "prj_RnD"):
                    if bool_key in record and record[bool_key] is None:
                        record[bool_key] = False

                existing = session.get(Project, project_id)
                if existing is None:
                    if not dry_run:
                        session.add(Project(**record))
                    inserted += 1
                else:
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
                        updated += 1
                    elif backfill_missing:
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

                        if changed:
                            backfilled += 1
                        else:
                            conflicted += 1
                    else:
                        conflicted += 1

                last_modified_raw = item.get(plan.legacy_modified_field)
                last_modified_dt = _normalize_datetime(_coerce_datetime(last_modified_raw))
                last_pk = project_id

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

    return {
        "pages": pages,
        "items": total_items,
        "inserted": inserted,
        "updated": updated,
        "backfilled": backfilled,
        "conflicted": conflicted,
    }
