import logging
from typing import Any

from sqlalchemy import inspect, text
import pandas as pd
import numpy as np

from ispec.db.init import initialize_db
from ispec.db.connect import get_session, get_db_path
from ispec.logging import get_logger
from ispec.io.io_file import get_writer


logger = get_logger(__file__, propagate=True)


def _log_info(message: str, *args: Any) -> None:
    """Log ``message`` using both the module logger and the root logger."""

    logger.info(message, *args)
    logging.getLogger().info(message, *args)


def check_status():
    """Query the database for its SQLite version and log/return it."""
    _log_info("checking db status...")
    with get_session() as session:
        result = session.execute(text("SELECT sqlite_version();")).fetchone()
        if result:
            version = result[0]
            _log_info("sqlite version: %s", version)
            return version
        logger.warning("sqlite version query returned no result")
        return None


def show_tables(file_path: str | None = None) -> dict[str, list[dict[str, Any]]]:
    """Return table and column metadata for the SQLite database.

    Parameters
    ----------
    file_path:
        Optional path to the SQLite database file. When ``None`` the default
        configuration from :func:`ispec.db.connect.get_session` is used.

    Returns
    -------
    dict
        Mapping of table names to a list of column definitions. Each column
        definition contains ``name``, ``type``, ``nullable`` and ``default``
        keys.
    """

    _log_info("showing tables..")
    with get_session(file_path=file_path) as session:
        inspector = inspect(session.bind)
        table_names = sorted(inspector.get_table_names())
        _log_info("tables: %s", table_names)

        table_definitions: dict[str, list[dict[str, Any]]] = {}
        for table_name in table_names:
            column_details: list[dict[str, Any]] = []
            for column in inspector.get_columns(table_name):
                column_details.append(
                    {
                        "name": column.get("name", ""),
                        "type": str(column.get("type", "")),
                        "nullable": bool(column.get("nullable", True)),
                        "default": column.get("default"),
                    }
                )
            table_definitions[table_name] = column_details

        return table_definitions


def import_file(file_path, table_name, db_file_path=None):
    from ispec.io import io_file

    logger.info("preparing to import file.. %s", file_path)
    io_file.import_file(file_path, table_name, db_file_path=db_file_path)
    # need to validate the file input, and understand which table we are meant to update


def initialize(file_path=None):
    """
    file_path can be gathered from environment variable or a sensible default if not provided
    """
    return initialize_db(file_path=file_path)


def export_table(table_name: str, file_path: str) -> None:
    """Export a database table to a file.

    Parameters
    ----------
    table_name:
        Name of the table to export.
    file_path:
        Destination path for the exported file. Supported extensions include
        ``.csv`` and ``.json``.
    """

    logger.info("exporting table %s to %s", table_name, file_path)
    with get_session() as session:
        df = pd.read_sql_table(table_name, session.bind)
    writer = get_writer(file_path)
    writer(df)


def import_legacy_dump(
    *,
    data_dir: str,
    db_file_path: str | None = None,
    mode: str = "merge",
    overwrite: bool = False,
    dry_run: bool = False,
) -> None:
    """Import the legacy FileMaker XLSX exports into the iSPEC database.

    Expected files inside ``data_dir``:
      - 20260101_people.xlsx
      - 20260101_projects.xlsx
      - 20260101_project_history.xlsx

    The importer:
      - Uses the legacy record numbers as primary keys (Project.id/Person.id)
      - Imports project history rows as ``project_comment`` rows with a
        synthetic "System" person (id=0)
    """

    from datetime import UTC, datetime
    from pathlib import Path

    from ispec.db.models import Person, Project, ProjectComment

    def normalize_column(name: str) -> str:
        # FileMaker exports often include relationship prefixes like
        # "iSPEC_Projects::prj_ProjectTitle".
        if "::" in name:
            return name.split("::")[-1].strip()
        return name.strip()

    def coerce_bool(value: Any) -> bool | None:
        if value is None or (isinstance(value, float) and np.isnan(value)):
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

    def coerce_datetime(value: Any) -> datetime | None:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return None
        if isinstance(value, pd.Timestamp):
            return value.to_pydatetime()
        if isinstance(value, datetime):
            return value
        if isinstance(value, str):
            parsed = pd.to_datetime(value, errors="coerce")
            if pd.isna(parsed):
                return None
            return parsed.to_pydatetime()
        return None

    def normalize_datetime(value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            return value
        return value.astimezone(UTC).replace(tzinfo=None)

    data_root = Path(data_dir).expanduser().resolve()
    people_path = data_root / "20260101_people.xlsx"
    projects_path = data_root / "20260101_projects.xlsx"
    history_path = data_root / "20260101_project_history.xlsx"

    missing = [p.name for p in (people_path, projects_path, history_path) if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing expected files in {data_root}: {', '.join(missing)}")

    if overwrite:
        mode = "overwrite"

    if mode not in {"merge", "overwrite"}:
        raise ValueError(f"Unsupported import mode: {mode}")

    # Resolve destination DB path and optionally overwrite.
    if db_file_path is None:
        db_file_path = get_db_path()

    # Normalize URL vs file path for overwrite checks.
    db_arg = str(db_file_path).strip()
    db_fs_path: Path | None = None
    if db_arg.startswith("sqlite:///"):
        db_fs_path = Path(db_arg.removeprefix("sqlite:///"))
    elif "://" not in db_arg:
        db_fs_path = Path(db_arg)

    if db_fs_path is not None:
        db_fs_path = db_fs_path.expanduser()
        db_fs_path.parent.mkdir(parents=True, exist_ok=True)
        if mode == "overwrite":
            db_fs_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                db_fs_path.unlink(missing_ok=True)
            except TypeError:  # py<3.8 compat (shouldn't happen here)
                if db_fs_path.exists():
                    db_fs_path.unlink()

    def read_excel(path: Path) -> pd.DataFrame:
        df = pd.read_excel(path)
        df = df.rename(columns={c: normalize_column(str(c)) for c in df.columns})
        df = df.replace({np.nan: None}).replace({pd.NaT: None})
        # Drop the ubiquitous FileMaker empty calc field if present
        df = df.drop(columns=[c for c in df.columns if c.lower().endswith("_gempty")], errors="ignore")
        return df

    people_df = read_excel(people_path)
    projects_df = read_excel(projects_path)
    history_df = read_excel(history_path)

    _log_info("legacy import: people rows=%d, projects rows=%d, history rows=%d", len(people_df), len(projects_df), len(history_df))

    if dry_run:
        _log_info("dry-run enabled; not writing to database")
        _log_info("people columns: %s", sorted(map(str, people_df.columns)))
        _log_info("projects columns: %s", sorted(map(str, projects_df.columns)))
        _log_info("history columns: %s", sorted(map(str, history_df.columns)))
        return

    # Build record dictionaries
    import_ts = normalize_datetime(datetime.now(UTC))
    people_records: list[dict[str, Any]] = []
    skipped_people = 0
    for row in people_df.to_dict(orient="records"):
        recno = row.pop("ppl_PPLRecNo", None)
        if recno is None:
            continue
        try:
            person_id = int(recno)
        except Exception:
            continue

        record = {k: v for k, v in row.items() if str(k).startswith("ppl_")}
        record["id"] = person_id
        added_by = record.get("ppl_AddedBy")
        if not (isinstance(added_by, str) and added_by.strip()):
            record["ppl_AddedBy"] = "legacy_import"
        record["ppl_LegacyImportTS"] = import_ts
        first = record.get("ppl_Name_First")
        last = record.get("ppl_Name_Last")
        if not (isinstance(first, str) and first.strip()) or not (isinstance(last, str) and last.strip()):
            skipped_people += 1
            continue

        created = coerce_datetime(record.get("ppl_CreationTS"))
        if created is not None:
            record["ppl_CreationTS"] = normalize_datetime(created)
        else:
            record.pop("ppl_CreationTS", None)

        modified = coerce_datetime(record.get("ppl_ModificationTS"))
        if modified is not None:
            record["ppl_ModificationTS"] = normalize_datetime(modified)
        else:
            record["ppl_ModificationTS"] = import_ts

        people_records.append(record)

    project_records: list[dict[str, Any]] = []
    placeholder_projects = 0
    for row in projects_df.to_dict(orient="records"):
        recno = row.pop("prj_PRJRecNo", None)
        if recno is None:
            continue
        try:
            project_id = int(recno)
        except Exception:
            continue

        record = {k: v for k, v in row.items() if str(k).startswith("prj_")}
        record["id"] = project_id
        added_by = record.get("prj_AddedBy")
        if not (isinstance(added_by, str) and added_by.strip()):
            record["prj_AddedBy"] = "legacy_import"
        record["prj_LegacyImportTS"] = import_ts

        title = record.get("prj_ProjectTitle")
        if not (isinstance(title, str) and title.strip()):
            placeholder_projects += 1
            record["prj_ProjectTitle"] = f"Untitled (PRJ {project_id})"

        # Normalize booleans/datetimes where present
        for bool_key in ("prj_Current_FLAG", "prj_Billing_ReadyToBill", "prj_PaymentReceived", "prj_RnD"):
            if bool_key in record:
                coerced = coerce_bool(record.get(bool_key))
                record[bool_key] = coerced if coerced is not None else False

        created = coerce_datetime(record.get("prj_CreationTS"))
        if created is not None:
            created = normalize_datetime(created)
            record["prj_CreationTS"] = created
        else:
            record.pop("prj_CreationTS", None)

        modified = coerce_datetime(record.get("prj_ModificationTS"))
        if modified is not None:
            record["prj_ModificationTS"] = normalize_datetime(modified)
        else:
            record["prj_ModificationTS"] = import_ts

        project_records.append(record)

    # Import history as project_comment rows.
    comment_records: list[dict[str, Any]] = []
    for row in history_df.to_dict(orient="records"):
        prj_recno = row.get("prh_PRJRecNo")
        if prj_recno is None:
            continue
        try:
            project_id = int(prj_recno)
        except Exception:
            continue

        comment = row.get("prh_Comment")
        if comment is None or (isinstance(comment, str) and not comment.strip()):
            continue

        created = coerce_datetime(row.get("prh_CreationTS"))
        record: dict[str, Any] = {
            "project_id": project_id,
            "person_id": 0,
            "com_LegacyImportTS": import_ts,
            "com_Comment": comment,
            "com_CommentType": row.get("prh_CommentType"),
            "com_AddedBy": row.get("prh_AddedBy"),
        }
        if created is not None:
            record["com_CreationTS"] = normalize_datetime(created)

        record["com_ModificationTS"] = import_ts
        comment_records.append(record)

    # Sanity checks (avoid FK errors if history references missing projects)
    project_ids = {r["id"] for r in project_records if "id" in r}
    before_comments = len(comment_records)
    comment_records = [r for r in comment_records if r.get("project_id") in project_ids]
    skipped_comments = before_comments - len(comment_records)
    if skipped_comments:
        _log_info("skipping %d history rows with unknown project_id", skipped_comments)

    def can_update_imported(
        *,
        obj: Any,
        added_by_field: str,
        created_field: str,
        modified_field: str,
        import_ts_field: str,
    ) -> bool:
        from datetime import timedelta

        imported_at = normalize_datetime(getattr(obj, import_ts_field, None))
        modified_at = normalize_datetime(getattr(obj, modified_field, None))
        if imported_at is None:
            if getattr(obj, added_by_field, None) != "legacy_import":
                return False
            created_at = normalize_datetime(getattr(obj, created_field, None))
            if created_at is None or modified_at is None:
                return True
            if modified_at <= created_at:
                return True
            return (modified_at - created_at) <= timedelta(seconds=5)
        if modified_at is None:
            return True
        return modified_at <= imported_at

    def can_update_imported_comment(comment: ProjectComment) -> bool:
        imported_at = normalize_datetime(getattr(comment, "com_LegacyImportTS", None))
        modified_at = normalize_datetime(getattr(comment, "com_ModificationTS", None))
        if imported_at is None:
            if int(getattr(comment, "person_id", -1)) != 0:
                return False
            created_at = normalize_datetime(getattr(comment, "com_CreationTS", None))
            return bool(created_at is not None and modified_at == created_at)
        if modified_at is None:
            return True
        return modified_at <= imported_at

    # Write to database
    with get_session(file_path=db_file_path) as session:
        # Create a synthetic system person used by imported history rows.
        system_person = session.get(Person, 0)
        if system_person is None:
            session.add(
                Person(
                    id=0,
                    ppl_AddedBy="legacy_import",
                    ppl_LegacyImportTS=import_ts,
                    ppl_Name_First="System",
                    ppl_Name_Last="System",
                    ppl_ModificationTS=import_ts,
                )
            )
            session.flush()

        inserted_people = updated_people = conflicted_people = 0
        for record in people_records:
            person_id = int(record["id"])
            existing = session.get(Person, person_id)
            if existing is None:
                record.setdefault("ppl_ModificationTS", import_ts)
                session.add(Person(**record))
                inserted_people += 1
                continue

            if can_update_imported(
                obj=existing,
                added_by_field="ppl_AddedBy",
                created_field="ppl_CreationTS",
                modified_field="ppl_ModificationTS",
                import_ts_field="ppl_LegacyImportTS",
            ):
                for key, value in record.items():
                    if key == "id":
                        continue
                    if hasattr(existing, key):
                        setattr(existing, key, value)
                if not getattr(existing, "ppl_ModificationTS", None):
                    existing.ppl_ModificationTS = import_ts
                updated_people += 1
            else:
                conflicted_people += 1

        inserted_projects = updated_projects = conflicted_projects = 0
        for record in project_records:
            project_id = int(record["id"])
            existing = session.get(Project, project_id)
            if existing is None:
                session.add(Project(**record))
                inserted_projects += 1
                continue

            if can_update_imported(
                obj=existing,
                added_by_field="prj_AddedBy",
                created_field="prj_CreationTS",
                modified_field="prj_ModificationTS",
                import_ts_field="prj_LegacyImportTS",
            ):
                for key, value in record.items():
                    if key == "id":
                        continue
                    if hasattr(existing, key):
                        setattr(existing, key, value)
                if not getattr(existing, "prj_ModificationTS", None):
                    existing.prj_ModificationTS = import_ts
                updated_projects += 1
            else:
                conflicted_projects += 1

        # Import history as project_comment rows with de-duplication.
        inserted_comments = updated_comments = conflicted_comments = 0

        def comment_key(record: dict[str, Any]) -> tuple[Any, ...]:
            created_at = normalize_datetime(record.get("com_CreationTS"))
            created_key = created_at.isoformat() if created_at else None
            return (
                int(record.get("project_id")),
                0,
                created_key,
                (record.get("com_CommentType") or "").strip(),
                (record.get("com_AddedBy") or "").strip(),
            )

        existing_comments = (
            session.query(ProjectComment)
            .filter(ProjectComment.project_id.in_(sorted(project_ids)))
            .filter(ProjectComment.person_id == 0)
            .all()
        )
        comment_index: dict[tuple[Any, ...], ProjectComment] = {}
        for comment in existing_comments:
            created_at = normalize_datetime(getattr(comment, "com_CreationTS", None))
            created_key = created_at.isoformat() if created_at else None
            key = (
                int(comment.project_id),
                int(comment.person_id),
                created_key,
                (comment.com_CommentType or "").strip(),
                (comment.com_AddedBy or "").strip(),
            )
            comment_index[key] = comment

        seen_comment_keys = set(comment_index)
        for record in comment_records:
            key = comment_key(record)
            if key in seen_comment_keys and key not in comment_index:
                continue
            existing = comment_index.get(key)
            if existing is None:
                session.add(ProjectComment(**record))
                inserted_comments += 1
                seen_comment_keys.add(key)
                continue

            if can_update_imported_comment(existing):
                for field in ("com_Comment", "com_CommentType", "com_AddedBy", "com_LegacyImportTS"):
                    if field in record:
                        setattr(existing, field, record[field])
                existing.com_ModificationTS = import_ts
                updated_comments += 1
            else:
                conflicted_comments += 1

        session.flush()

    _log_info(
        "legacy import complete (%s): people=%d (new=%d, updated=%d, conflicts=%d), projects=%d (new=%d, updated=%d, conflicts=%d), project_comment=%d (new=%d, updated=%d, conflicts=%d)",
        mode,
        inserted_people + updated_people,
        inserted_people,
        updated_people,
        conflicted_people,
        inserted_projects + updated_projects,
        inserted_projects,
        updated_projects,
        conflicted_projects,
        inserted_comments + updated_comments,
        inserted_comments,
        updated_comments,
        conflicted_comments,
    )
    if skipped_people or placeholder_projects:
        _log_info(
            "legacy import placeholders/skips: people=%d, projects=%d (missing required fields)",
            skipped_people,
            placeholder_projects,
        )


def import_e2g(
    *,
    data_dir: str | None = None,
    qual_paths: list[str] | None = None,
    quant_paths: list[str] | None = None,
    db_file_path: str | None = None,
    create_missing_runs: bool = True,
    create_missing_experiments: bool = True,
    store_metadata: bool = False,
    skip_imported: bool = True,
    force: bool = False,
) -> dict[str, Any]:
    """Import gpgrouper experiment-to-gene tables (QUAL/QUANT TSVs).

    Parameters
    ----------
    data_dir:
        Directory containing ``*_e2g_QUAL.tsv`` and/or ``*_e2g_QUANT.tsv`` files.
    qual_paths:
        Explicit QUAL TSV paths (repeatable).
    quant_paths:
        Explicit QUANT TSV paths (repeatable).
    db_file_path:
        SQLite database URL or filesystem path to write to.
    create_missing_runs:
        When True, create missing ExperimentRun rows (requires experiment; see create_missing_experiments).
    create_missing_experiments:
        When True, create missing Experiment rows (project_id may be NULL until backfilled).
    store_metadata:
        When True, store a small subset of extra columns in ``metadata_json``.
    skip_imported:
        When True, skip importing a file if that run already has QUAL/QUANT fields populated.
    force:
        When True, delete existing E2G rows for affected ExperimentRuns and re-import.
    """

    from pathlib import Path

    from ispec.omics.e2g_import import discover_e2g_tsvs, import_e2g_files

    sources = [bool(data_dir), bool(qual_paths), bool(quant_paths)]
    if not any(sources):
        raise ValueError("Provide data_dir, qual_paths, or quant_paths.")

    resolved_qual: list[Path] = []
    resolved_quant: list[Path] = []

    if data_dir:
        qual_found, quant_found = discover_e2g_tsvs(data_dir)
        resolved_qual.extend(qual_found)
        resolved_quant.extend(quant_found)

    if qual_paths:
        resolved_qual.extend(Path(p).expanduser().resolve() for p in qual_paths if str(p).strip())

    if quant_paths:
        resolved_quant.extend(Path(p).expanduser().resolve() for p in quant_paths if str(p).strip())

    # De-dupe while keeping a stable order.
    def uniq(paths: list[Path]) -> list[Path]:
        seen: set[str] = set()
        out: list[Path] = []
        for p in paths:
            key = str(p)
            if key in seen:
                continue
            seen.add(key)
            out.append(p)
        return out

    resolved_qual = uniq(resolved_qual)
    resolved_quant = uniq(resolved_quant)

    _log_info(
        "importing E2G TSVs: qual=%d, quant=%d, create_missing_runs=%s, create_missing_experiments=%s, skip_imported=%s, force=%s, store_metadata=%s",
        len(resolved_qual),
        len(resolved_quant),
        bool(create_missing_runs),
        bool(create_missing_experiments),
        bool(skip_imported),
        bool(force),
        bool(store_metadata),
    )

    with get_session(file_path=db_file_path) as session:
        return import_e2g_files(
            session,
            qual_paths=resolved_qual,
            quant_paths=resolved_quant,
            create_missing_runs=create_missing_runs,
            create_missing_experiments=create_missing_experiments,
            store_metadata=store_metadata,
            skip_imported=skip_imported,
            force=force,
        )
