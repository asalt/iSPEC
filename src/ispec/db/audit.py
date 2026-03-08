from __future__ import annotations

import csv
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import inspect, text

from ispec.config.paths import resolve_db_location
from ispec.db.models import sqlite_engine
from ispec.logging import get_logger


logger = get_logger(__file__)

ANALYSIS_TABLES = (
    "experiment_to_gene",
    "gene_contrast",
    "gene_contrast_stat",
    "gsea_analysis",
    "gsea_result",
)

PSM_TABLES = (
    "psm",
)

_INCREMENTAL_SYNC_COMMANDS = {
    "iSPEC_Projects": "sync-legacy-projects",
    "iSPEC_People": "sync-legacy-people",
    "iSPEC_Experiments": "sync-legacy-experiments",
    "iSPEC_ExperimentRuns": "sync-legacy-experiment-runs",
}


def _utcnow_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def default_artifact_dir() -> Path:
    return _repo_root() / "data"


def _default_path(path: str | None, fallback_name: str) -> Path:
    if path:
        return Path(path).expanduser().resolve()
    return (default_artifact_dir() / fallback_name).resolve()


def _sqlite_path(raw: str | Path | None) -> Path | None:
    if raw is None:
        return None
    value = str(raw).strip()
    if not value:
        return None
    if value.startswith("sqlite:///"):
        return Path(value.removeprefix("sqlite:///")).expanduser()
    if "://" in value:
        return None
    return Path(value).expanduser()


def _sqlite_uri(raw: str | Path) -> str:
    value = str(raw).strip()
    if value.startswith("sqlite"):
        return value
    return "sqlite:///" + str(Path(value).expanduser())


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        logger.warning("Unable to parse JSON: %s", path)
        return {}


def _read_table_list(path: Path) -> list[str]:
    if not path.exists():
        return []
    items: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "\t" in stripped and stripped.lower().startswith("tables\t"):
            continue
        if stripped.lower() == "tables":
            continue
        items.append(stripped.split("\t", 1)[0])
    return items


def _table_columns_and_counts(db_path: Path | None) -> dict[str, Any]:
    if db_path is None:
        return {"exists": False, "path": None, "tables": []}

    resolved = db_path.resolve()
    if not resolved.exists():
        return {"exists": False, "path": str(resolved), "tables": []}

    engine = sqlite_engine(_sqlite_uri(resolved))
    inspector = inspect(engine)
    table_names = sorted(inspector.get_table_names())
    tables: list[dict[str, Any]] = []

    with engine.connect() as conn:
        for table_name in table_names:
            try:
                row_count = int(
                    conn.execute(text(f'SELECT COUNT(*) FROM "{table_name}"')).scalar() or 0
                )
            except Exception:
                row_count = None
            tables.append(
                {
                    "name": table_name,
                    "row_count": row_count,
                    "columns": [column["name"] for column in inspector.get_columns(table_name)],
                }
            )

    return {"exists": True, "path": str(resolved), "tables": tables}


def _registry_rows(db_path: Path | None) -> list[dict[str, Any]]:
    if db_path is None or not db_path.exists():
        return []
    engine = sqlite_engine(_sqlite_uri(db_path))
    inspector = inspect(engine)
    if "omics_database_registry" not in set(inspector.get_table_names()):
        return []

    with engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT omdb_LogicalName, omdb_DBURI, omdb_DBPath, omdb_Status, "
                "omdb_LastCheckedTS, omdb_LastAvailableTS, omdb_LastError "
                "FROM omics_database_registry ORDER BY id ASC"
            )
        ).mappings()
        return [dict(row) for row in rows]


def _legacy_sync_state_rows(db_path: Path | None) -> list[dict[str, Any]]:
    if db_path is None or not db_path.exists():
        return []
    engine = sqlite_engine(_sqlite_uri(db_path))
    inspector = inspect(engine)
    if "legacy_sync_state" not in set(inspector.get_table_names()):
        return []

    with engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT legacy_table, since, since_pk "
                "FROM legacy_sync_state ORDER BY legacy_table ASC"
            )
        ).mappings()
        return [dict(row) for row in rows]


def _known_workflows(scripts_dir: Path) -> dict[str, Any]:
    helper_scripts = sorted(
        {
            path.name
            for path in scripts_dir.glob("import_project*.sh")
            if path.is_file()
        }
    )
    return {
        "legacy_sync": [
            {"command": "sync-legacy-projects", "target": "project"},
            {"command": "sync-legacy-people", "target": "person"},
            {"command": "sync-legacy-experiments", "target": "experiment"},
            {"command": "sync-legacy-experiment-runs", "target": "experiment_run"},
            {"command": "sync-legacy-all", "target": "core metadata"},
        ],
        "structured_imports": [
            {"command": "import-e2g", "target": "experiment_to_gene"},
            {"command": "import-psm", "target": "psm"},
            {"command": "import-volcano", "target": "gene_contrast"},
            {"command": "import-gsea", "target": "gsea_result"},
            {"command": "import-results", "target": "project_file + volcano"},
        ],
        "helper_scripts": helper_scripts,
    }


def _guess_local_destination(table_name: str) -> str:
    lower = table_name.lower()
    if "projecthistory" in lower or lower.endswith("_history"):
        return "project_comment"
    if "people" in lower:
        return "person"
    if "project" in lower:
        return "project"
    if "experimentruns" in lower or "expruns" in lower:
        return "experiment_run"
    if "experiment" in lower:
        return "experiment"
    if "psm" in lower:
        return "psm"
    if "peptide" in lower:
        return "psm"
    if "exp2gene" in lower or "_e2g" in lower or lower.endswith("e2g"):
        return "experiment_to_gene"
    if "gsea" in lower:
        return "gsea_result"
    if "volcano" in lower or "contrast" in lower:
        return "gene_contrast"
    if "attachment" in lower or "file" in lower:
        return "project_file"
    return ""


def _classify_legacy_table(
    *,
    table_name: str,
    mapping_entry: dict[str, Any] | None,
) -> tuple[str, str, str, str]:
    if mapping_entry is not None:
        command = _INCREMENTAL_SYNC_COMMANDS.get(table_name, "mapped_via_legacy_mapping")
        sync_mode = "incremental_api_sync"
        status = "implemented"
        blocker = ""
        return "mapped", command, sync_mode, blocker

    lower = table_name.lower()
    if "psm" in lower or "peptide" in lower:
        return (
            "file_imported",
            "import-psm",
            "structured_file_import",
            "Legacy table is not yet mapped into the incremental sync plan.",
        )
    if "exp2gene" in lower or "_e2g" in lower or lower.endswith("e2g"):
        return (
            "file_imported",
            "import-e2g",
            "structured_file_import",
            "Legacy table is not yet mapped into the incremental sync plan.",
        )
    if "gsea" in lower:
        return (
            "file_imported",
            "import-gsea",
            "structured_file_import",
            "Legacy table is not yet mapped into the incremental sync plan.",
        )
    if "volcano" in lower or "contrast" in lower:
        return (
            "file_imported",
            "import-volcano",
            "structured_file_import",
            "Legacy table is not yet mapped into the incremental sync plan.",
        )
    if "attachment" in lower or "file" in lower:
        return (
            "attachment_only",
            "import-results",
            "attachment_ingest",
            "Structured metadata is not modeled; files are stored in project_file.",
        )
    if table_name.startswith("iSPEC_") or table_name.startswith("BCM_"):
        return (
            "deferred",
            "",
            "manual_review",
            "No current mapping or importer; needs product/schema review.",
        )
    return (
        "unsupported",
        "",
        "manual_review",
        "No current mapping or importer; needs product/schema review.",
    )


def build_legacy_gap_rows(
    *,
    legacy_schema_path: str | None = None,
    legacy_mapping_path: str | None = None,
    legacy_tables_file_path: str | None = None,
) -> list[dict[str, Any]]:
    schema_path = _default_path(legacy_schema_path, "ispec-legacy-schema.json")
    mapping_path = _default_path(legacy_mapping_path, "legacy-mapping.json")
    tables_path = _default_path(legacy_tables_file_path, "ispec-legacy-tables.tsv")

    schema_payload = _read_json(schema_path)
    mapping_payload = _read_json(mapping_path)
    schema_tables = schema_payload.get("tables", {})
    schema_tables = schema_tables if isinstance(schema_tables, dict) else {}
    mapping_tables = mapping_payload.get("tables", {})
    mapping_tables = mapping_tables if isinstance(mapping_tables, dict) else {}

    discovered_tables = set(schema_tables.keys())
    discovered_tables.update(_read_table_list(tables_path))

    rows: list[dict[str, Any]] = []
    for table_name in sorted(discovered_tables):
        schema_entry = schema_tables.get(table_name, {})
        schema_entry = schema_entry if isinstance(schema_entry, dict) else {}
        mapping_entry = mapping_tables.get(table_name)
        mapping_entry = mapping_entry if isinstance(mapping_entry, dict) else None
        classification, ingest_path, sync_mode, blocker = _classify_legacy_table(
            table_name=table_name,
            mapping_entry=mapping_entry,
        )

        field_metadata_available = bool(schema_entry.get("fields"))
        if not field_metadata_available:
            prefix = "Field metadata missing from current schema snapshot; rerun data/fetch-legacy.py."
            blocker = f"{prefix} {blocker}".strip() if blocker else prefix

        rows.append(
            {
                "legacy_table": table_name,
                "pk_field": (
                    (mapping_entry or {}).get("pk", {}).get("legacy")
                    or schema_entry.get("pk_guess")
                    or ""
                ),
                "modified_field": (
                    (mapping_entry or {}).get("modified_ts")
                    or schema_entry.get("modification_ts_guess")
                    or ""
                ),
                "field_count": len(schema_entry.get("fields", []) or []),
                "field_metadata_available": field_metadata_available,
                "local_destination": (
                    (mapping_entry or {}).get("local_table")
                    or _guess_local_destination(table_name)
                ),
                "classification": classification,
                "sync_mode": sync_mode,
                "current_ingest_path": ingest_path,
                "status": "implemented" if classification in {"mapped", "file_imported", "attachment_only"} else "review",
                "blocker": blocker,
            }
        )

    return rows


def build_import_audit(
    *,
    db_file_path: str | None = None,
    analysis_db_file_path: str | None = None,
    psm_db_file_path: str | None = None,
    omics_db_file_path: str | None = None,
    legacy_schema_path: str | None = None,
    legacy_mapping_path: str | None = None,
    legacy_tables_file_path: str | None = None,
    scripts_dir: str | None = None,
) -> dict[str, Any]:
    core_path = _sqlite_path(db_file_path)
    if core_path is None:
        core_default = resolve_db_location("core")
        core_path = _sqlite_path(core_default.uri or core_default.value)
    if core_path is None:
        core_path = (default_artifact_dir() / "ispec-import.db").resolve()
    analysis_path = _sqlite_path(analysis_db_file_path or omics_db_file_path)
    if analysis_path is None:
        analysis_default = resolve_db_location("analysis")
        analysis_path = _sqlite_path(analysis_default.uri or analysis_default.value)
    if analysis_path is None:
        analysis_path = core_path.parent / "ispec-analysis.db"
    psm_path = _sqlite_path(psm_db_file_path)
    if psm_path is None:
        psm_default = resolve_db_location("psm")
        psm_path = _sqlite_path(psm_default.uri or psm_default.value)
    if psm_path is None:
        psm_path = core_path.parent / "ispec-psm.db"
    scripts_root = Path(scripts_dir).expanduser().resolve() if scripts_dir else (_repo_root() / "scripts")

    core_info = _table_columns_and_counts(core_path)
    analysis_info = _table_columns_and_counts(analysis_path)
    psm_info = _table_columns_and_counts(psm_path)

    core_table_map = {item["name"]: item for item in core_info.get("tables", [])}
    analysis_subset = [core_table_map[name] for name in ANALYSIS_TABLES if name in core_table_map]
    psm_subset = [core_table_map[name] for name in PSM_TABLES if name in core_table_map]
    if analysis_path != core_path:
        analysis_subset = [item for item in analysis_info.get("tables", []) if item["name"] in ANALYSIS_TABLES]
    if psm_path != core_path:
        psm_subset = [item for item in psm_info.get("tables", []) if item["name"] in PSM_TABLES]

    gap_rows = build_legacy_gap_rows(
        legacy_schema_path=legacy_schema_path,
        legacy_mapping_path=legacy_mapping_path,
        legacy_tables_file_path=legacy_tables_file_path,
    )
    classification_counts: dict[str, int] = {}
    for row in gap_rows:
        classification = str(row.get("classification") or "unknown")
        classification_counts[classification] = classification_counts.get(classification, 0) + 1

    return {
        "generated_at": _utcnow_iso(),
        "core_database": {
            "path": core_info.get("path"),
            "exists": bool(core_info.get("exists")),
            "table_count": len(core_info.get("tables", [])),
            "tables": core_info.get("tables", []),
            "omics_registry": _registry_rows(core_path),
            "legacy_sync_state": _legacy_sync_state_rows(core_path),
        },
        "analysis_database": {
            "path": analysis_info.get("path"),
            "exists": bool(analysis_info.get("exists")),
            "same_as_core": core_path == analysis_path,
            "table_count": len(analysis_subset),
            "tables": analysis_subset,
        },
        "psm_database": {
            "path": psm_info.get("path"),
            "exists": bool(psm_info.get("exists")),
            "same_as_core": core_path == psm_path,
            "same_as_analysis": analysis_path == psm_path,
            "table_count": len(psm_subset),
            "tables": psm_subset,
        },
        "omics_database": {
            "path": analysis_info.get("path"),
            "exists": bool(analysis_info.get("exists")),
            "same_as_core": core_path == analysis_path,
            "table_count": len(analysis_subset),
            "tables": analysis_subset,
        },
        "legacy": {
            "schema_path": str(_default_path(legacy_schema_path, "ispec-legacy-schema.json")),
            "mapping_path": str(_default_path(legacy_mapping_path, "legacy-mapping.json")),
            "tables_file_path": str(_default_path(legacy_tables_file_path, "ispec-legacy-tables.tsv")),
            "gap_summary": classification_counts,
        },
        "workflows": _known_workflows(scripts_root),
        "legacy_gap_rows": gap_rows,
    }


def _markdown_table(rows: list[dict[str, Any]], *, columns: list[tuple[str, str]]) -> list[str]:
    if not rows:
        return ["_None_"]
    header = "| " + " | ".join(label for label, _ in columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    lines = [header, sep]
    for row in rows:
        values = []
        for _, key in columns:
            value = row.get(key, "")
            values.append("" if value is None else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def render_import_audit_markdown(audit: dict[str, Any]) -> str:
    core_database = audit.get("core_database", {})
    analysis_database = audit.get("analysis_database", audit.get("omics_database", {}))
    psm_database = audit.get("psm_database", {})
    workflows = audit.get("workflows", {})
    legacy = audit.get("legacy", {})
    gap_rows = list(audit.get("legacy_gap_rows", []))

    lines: list[str] = [
        "# iSPEC DB / Import Audit",
        "",
        f"- Generated at: `{audit.get('generated_at', '')}`",
        f"- Core DB: `{core_database.get('path')}`",
        f"- Analysis DB: `{analysis_database.get('path')}`",
        f"- Analysis same as core: `{bool(analysis_database.get('same_as_core'))}`",
        f"- PSM DB: `{psm_database.get('path')}`",
        f"- PSM same as core: `{bool(psm_database.get('same_as_core'))}`",
        f"- PSM same as analysis: `{bool(psm_database.get('same_as_analysis'))}`",
        "",
        "## Core Tables",
        "",
        *_markdown_table(
            list(core_database.get("tables", [])),
            columns=[("Table", "name"), ("Rows", "row_count")],
        ),
        "",
        "## Analysis Tables",
        "",
        *_markdown_table(
            list(analysis_database.get("tables", [])),
            columns=[("Table", "name"), ("Rows", "row_count")],
        ),
        "",
        "## PSM Tables",
        "",
        *_markdown_table(
            list(psm_database.get("tables", [])),
            columns=[("Table", "name"), ("Rows", "row_count")],
        ),
        "",
        "## Legacy Sync State",
        "",
        *_markdown_table(
            list(core_database.get("legacy_sync_state", [])),
            columns=[("Legacy Table", "legacy_table"), ("Since", "since"), ("Since PK", "since_pk")],
        ),
        "",
        "## Legacy Coverage",
        "",
        f"- Schema path: `{legacy.get('schema_path')}`",
        f"- Mapping path: `{legacy.get('mapping_path')}`",
        f"- Tables file: `{legacy.get('tables_file_path')}`",
        f"- Gap summary: `{legacy.get('gap_summary', {})}`",
        "",
        *_markdown_table(
            gap_rows,
            columns=[
                ("Legacy Table", "legacy_table"),
                ("Destination", "local_destination"),
                ("Classification", "classification"),
                ("Ingest Path", "current_ingest_path"),
                ("Status", "status"),
            ],
        ),
        "",
        "## Workflows",
        "",
        *_markdown_table(
            list(workflows.get("structured_imports", [])),
            columns=[("Command", "command"), ("Target", "target")],
        ),
        "",
        "## Helper Scripts",
        "",
    ]
    helper_scripts = list(workflows.get("helper_scripts", []))
    if helper_scripts:
        lines.extend(f"- `{name}`" for name in helper_scripts)
    else:
        lines.append("_None_")
    return "\n".join(lines).rstrip() + "\n"


def write_import_audit_artifacts(
    *,
    db_file_path: str | None = None,
    analysis_db_file_path: str | None = None,
    psm_db_file_path: str | None = None,
    omics_db_file_path: str | None = None,
    legacy_schema_path: str | None = None,
    legacy_mapping_path: str | None = None,
    legacy_tables_file_path: str | None = None,
    scripts_dir: str | None = None,
    out_dir: str | None = None,
) -> dict[str, Any]:
    target_dir = Path(out_dir).expanduser().resolve() if out_dir else default_artifact_dir()
    target_dir.mkdir(parents=True, exist_ok=True)

    audit = build_import_audit(
        db_file_path=db_file_path,
        analysis_db_file_path=analysis_db_file_path,
        psm_db_file_path=psm_db_file_path,
        omics_db_file_path=omics_db_file_path,
        legacy_schema_path=legacy_schema_path,
        legacy_mapping_path=legacy_mapping_path,
        legacy_tables_file_path=legacy_tables_file_path,
        scripts_dir=scripts_dir,
    )
    gap_rows = list(audit.get("legacy_gap_rows", []))

    audit_json_path = target_dir / "ispec-db-audit.json"
    audit_md_path = target_dir / "ispec-db-audit.md"
    gap_tsv_path = target_dir / "ispec-legacy-gap-matrix.tsv"

    audit_json_path.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    audit_md_path.write_text(render_import_audit_markdown(audit), encoding="utf-8")

    fieldnames = [
        "legacy_table",
        "pk_field",
        "modified_field",
        "field_count",
        "field_metadata_available",
        "local_destination",
        "classification",
        "sync_mode",
        "current_ingest_path",
        "status",
        "blocker",
    ]
    with gap_tsv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in gap_rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})

    summary = {
        "audit_json": str(audit_json_path),
        "audit_md": str(audit_md_path),
        "gap_tsv": str(gap_tsv_path),
        "core_tables": int(audit.get("core_database", {}).get("table_count", 0) or 0),
        "analysis_tables": int(audit.get("analysis_database", {}).get("table_count", 0) or 0),
        "psm_tables": int(audit.get("psm_database", {}).get("table_count", 0) or 0),
        "omics_tables": int(audit.get("analysis_database", {}).get("table_count", 0) or 0),
        "legacy_tables": len(gap_rows),
        "gap_summary": audit.get("legacy", {}).get("gap_summary", {}),
    }
    logger.info("Wrote DB/import audit artifacts: %s", summary)
    return summary
