"""Command-line helpers for interacting with the iSPEC database.

This module provides utilities to register database-related subcommands with an
``argparse`` parser and to dispatch parsed arguments to the appropriate
database operations.
"""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from alembic import command
from alembic.config import Config
from rich.console import Console
from rich.table import Table

from ispec.db import operations
from ispec.logging import get_logger


def register_subcommands(subparsers):
    """Attach database subcommands to an ``argparse`` parser.

    Parameters
    ----------
    subparsers : :class:`argparse._SubParsersAction`
        The ``argparse`` subparsers object to which database commands will be
        registered.

    Examples
    --------
    >>> import argparse
    >>> parser = argparse.ArgumentParser(prog="ispec db")
    >>> subparsers = parser.add_subparsers(dest="subcommand", required=True)
    >>> register_subcommands(subparsers)
    >>> parser.parse_args(["init", "--file", "test.db"])
    Namespace(subcommand='init', file='test.db')

    Exporting data::

    >>> parser.parse_args(["export", "--table-name", "person", "--file", "out.json"])
    Namespace(subcommand='export', table_name='person', file='out.json')

    """

    init_parser = subparsers.add_parser("init", help="initialize db")
    init_parser.add_argument("--file", required=False)

    _ = subparsers.add_parser("status", help="Check DB status")
    _ = subparsers.add_parser("show", help="Show tables")

    import_parser = subparsers.add_parser("import", help="Import file")
    import_parser.add_argument(
        "--table-name",
        required=True,
        choices=("person", "project", "comment", "letter"),
    )
    import_parser.add_argument("--file", required=True)

    import_e2g_parser = subparsers.add_parser(
        "import-e2g", help="Import gpgrouper E2G QUAL/QUANT TSVs"
    )
    import_e2g_parser.add_argument(
        "--dir",
        dest="data_dir",
        help="Directory containing *_e2g_QUAL.tsv / *_e2g_QUANT.tsv files",
    )
    import_e2g_parser.add_argument(
        "--qual",
        dest="qual_paths",
        action="append",
        default=[],
        help="Path to a *_e2g_QUAL.tsv file (repeatable)",
    )
    import_e2g_parser.add_argument(
        "--quant",
        dest="quant_paths",
        action="append",
        default=[],
        help="Path to a *_e2g_QUANT.tsv file (repeatable)",
    )
    import_e2g_parser.add_argument(
        "--database",
        dest="database",
        help="SQLite database URL or filesystem path to write to (defaults to ISPEC_DB_PATH/default)",
    )
    import_e2g_parser.add_argument(
        "--create-missing-runs",
        action="store_true",
        help="Create missing ExperimentRun rows when needed (experiment must exist).",
    )
    import_e2g_parser.add_argument(
        "--store-metadata",
        action="store_true",
        help="Store a small subset of extra columns in metadata_json.",
    )

    export_parser = subparsers.add_parser("export", help="Export table to CSV or JSON")
    export_parser.add_argument("--table-name", required=True, choices=("person", "project"))
    export_parser.add_argument("--file", required=True, help="Output file (CSV or JSON)")

    legacy_parser = subparsers.add_parser(
        "import-legacy", help="Import a FileMaker XLSX dump (projects/people/history)"
    )
    legacy_parser.add_argument(
        "--data-dir",
        required=True,
        help="Directory containing 20260101_projects.xlsx, 20260101_people.xlsx, 20260101_project_history.xlsx",
    )
    legacy_parser.add_argument(
        "--database",
        dest="database",
        help="SQLite database URL or filesystem path to write to (defaults to ISPEC_DB_PATH/default)",
    )
    legacy_parser.add_argument(
        "--mode",
        choices=("merge", "overwrite"),
        default="merge",
        help="Import mode (default: merge). Use overwrite to replace the destination DB file.",
    )
    legacy_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the destination database file if it already exists (deprecated; use --mode overwrite)",
    )
    legacy_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and report what would be imported without writing",
    )

    upgrade_parser = subparsers.add_parser(
        "upgrade", help="Apply Alembic migrations up to a revision"
    )
    upgrade_parser.add_argument(
        "revision",
        nargs="?",
        default="head",
        help="Alembic revision identifier to upgrade to (default: head)",
    )
    upgrade_parser.add_argument(
        "--database",
        dest="database",
        help="Database URL or filesystem path to migrate",
    )

    downgrade_parser = subparsers.add_parser(
        "downgrade", help="Revert Alembic migrations"
    )
    downgrade_parser.add_argument(
        "revision",
        nargs="?",
        default="-1",
        help="Alembic revision identifier to downgrade to (default: -1)",
    )
    downgrade_parser.add_argument(
        "--database",
        dest="database",
        help="Database URL or filesystem path to migrate",
    )

    sync_projects_parser = subparsers.add_parser(
        "sync-legacy-projects",
        help="Incrementally sync legacy iSPEC Projects via the legacy API",
    )
    sync_projects_parser.add_argument(
        "--database",
        dest="database",
        help="SQLite database URL or filesystem path to write to (defaults to ISPEC_DB_PATH/default)",
    )
    sync_projects_parser.add_argument(
        "--legacy-url",
        dest="legacy_url",
        help="Legacy API base URL (defaults to ISPEC_LEGACY_API_URL or iSPEC/data/ispec-legacy-schema.json base_url)",
    )
    sync_projects_parser.add_argument(
        "--id",
        dest="project_id",
        type=int,
        help="Sync a single legacy project by PRJRecNo (debug/verification)",
    )
    sync_projects_parser.add_argument(
        "--mapping",
        dest="mapping",
        help="Path to legacy-mapping.json (default: iSPEC/data/legacy-mapping.json)",
    )
    sync_projects_parser.add_argument(
        "--schema",
        dest="schema",
        help="Path to ispec-legacy-schema.json (default: iSPEC/data/ispec-legacy-schema.json)",
    )
    sync_projects_parser.add_argument("--limit", type=int, default=1000)
    sync_projects_parser.add_argument("--max-pages", type=int, default=None)
    sync_projects_parser.add_argument(
        "--reset-cursor",
        action="store_true",
        help="Ignore stored sync cursor and start from the beginning (fetches all rows)",
    )
    sync_projects_parser.add_argument(
        "--since",
        dest="since",
        help="Override cursor timestamp (ISO-8601 recommended)",
    )
    sync_projects_parser.add_argument(
        "--since-pk",
        dest="since_pk",
        type=int,
        help="Override cursor PK tie-breaker (integer)",
    )
    sync_projects_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch + compute changes without writing to the DB or advancing the cursor",
    )
    sync_projects_parser.add_argument(
        "--backfill-missing",
        action="store_true",
        help="When a row is conflicted, still fill NULL/blank fields from legacy without overwriting existing values.",
    )
    sync_projects_parser.add_argument(
        "--dump-json",
        dest="dump_json",
        help="Write raw legacy API payload(s) to a JSON file (ends with .json) or a directory path. Also supports ISPEC_LEGACY_DUMP_JSON/ISPEC_LEGACY_DUMP_DIR.",
    )

    sync_experiments_parser = subparsers.add_parser(
        "sync-legacy-experiments",
        help="Sync legacy iSPEC Experiments via the legacy API",
    )
    sync_experiments_parser.add_argument(
        "--database",
        dest="database",
        help="SQLite database URL or filesystem path to write to (defaults to ISPEC_DB_PATH/default)",
    )
    sync_experiments_parser.add_argument(
        "--legacy-url",
        dest="legacy_url",
        help="Legacy API base URL (defaults to ISPEC_LEGACY_API_URL or iSPEC/data/ispec-legacy-schema.json base_url)",
    )
    sync_experiments_parser.add_argument(
        "--id",
        dest="experiment_id",
        type=int,
        help="Sync a single legacy experiment by EXPRecNo (debug/verification)",
    )
    sync_experiments_parser.add_argument(
        "--mapping",
        dest="mapping",
        help="Path to legacy-mapping.json (default: iSPEC/data/legacy-mapping.json)",
    )
    sync_experiments_parser.add_argument(
        "--schema",
        dest="schema",
        help="Path to ispec-legacy-schema.json (default: iSPEC/data/ispec-legacy-schema.json)",
    )
    sync_experiments_parser.add_argument("--limit", type=int, default=1000)
    sync_experiments_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch + compute changes without writing to the DB",
    )
    sync_experiments_parser.add_argument(
        "--dump-json",
        dest="dump_json",
        help="Write raw legacy API payload(s) to a JSON file (ends with .json) or a directory path. Also supports ISPEC_LEGACY_DUMP_JSON/ISPEC_LEGACY_DUMP_DIR.",
    )

    sync_runs_parser = subparsers.add_parser(
        "sync-legacy-experiment-runs",
        help="Sync legacy iSPEC ExperimentRuns for a specific experiment",
    )
    sync_runs_parser.add_argument(
        "--database",
        dest="database",
        help="SQLite database URL or filesystem path to write to (defaults to ISPEC_DB_PATH/default)",
    )
    sync_runs_parser.add_argument(
        "--legacy-url",
        dest="legacy_url",
        help="Legacy API base URL (defaults to ISPEC_LEGACY_API_URL or iSPEC/data/ispec-legacy-schema.json base_url)",
    )
    sync_runs_parser.add_argument(
        "--experiment-id",
        dest="experiment_id",
        type=int,
        required=True,
        help="Experiment EXPRecNo to sync runs for",
    )
    sync_runs_parser.add_argument(
        "--mapping",
        dest="mapping",
        help="Path to legacy-mapping.json (default: iSPEC/data/legacy-mapping.json)",
    )
    sync_runs_parser.add_argument(
        "--schema",
        dest="schema",
        help="Path to ispec-legacy-schema.json (default: iSPEC/data/ispec-legacy-schema.json)",
    )
    sync_runs_parser.add_argument("--limit", type=int, default=5000)
    sync_runs_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch + compute changes without writing to the DB",
    )
    sync_runs_parser.add_argument(
        "--dump-json",
        dest="dump_json",
        help="Write raw legacy API payload(s) to a JSON file (ends with .json) or a directory path. Also supports ISPEC_LEGACY_DUMP_JSON/ISPEC_LEGACY_DUMP_DIR.",
    )


def dispatch(args):
    """Run the database operation associated with ``args.subcommand``.

    Parameters
    ----------
    args : :class:`argparse.Namespace`
        Parsed arguments containing the targeted ``subcommand`` and any
        additional options required by that subcommand.

    Examples
    --------
    >>> import types
    >>> args = types.SimpleNamespace(subcommand="status")
    >>> dispatch(args)  # doctest: +SKIP

    Importing data::

        >>> args = types.SimpleNamespace(subcommand="import", file="data.csv", table_name="person")
        >>> dispatch(args)  # doctest: +SKIP

    Exporting data::

        >>> args = types.SimpleNamespace(subcommand="export", table_name="person", file="out.json")
        >>> dispatch(args)  # doctest: +SKIP

    """

    logger = get_logger(__file__)

    if args.subcommand == "status":
        operations.check_status()
    elif args.subcommand == "show":
        table_definitions = operations.show_tables()
        _render_table_overview(table_definitions)
    elif args.subcommand == "import":
        operations.import_file(args.file, args.table_name)
    elif args.subcommand == "import-e2g":
        summary = operations.import_e2g(
            data_dir=getattr(args, "data_dir", None),
            qual_paths=list(getattr(args, "qual_paths", []) or []),
            quant_paths=list(getattr(args, "quant_paths", []) or []),
            db_file_path=getattr(args, "database", None),
            create_missing_runs=bool(getattr(args, "create_missing_runs", False)),
            store_metadata=bool(getattr(args, "store_metadata", False)),
        )
        logger.info("E2G import summary: %s", summary)
    elif args.subcommand == "export":
        operations.export_table(args.table_name, args.file)
    elif args.subcommand == "init":
        operations.initialize(file_path=args.file)
    elif args.subcommand == "import-legacy":
        mode = getattr(args, "mode", "merge")
        overwrite = bool(getattr(args, "overwrite", False))
        if overwrite:
            mode = "overwrite"
        operations.import_legacy_dump(
            data_dir=args.data_dir,
            db_file_path=args.database,
            mode=mode,
            overwrite=overwrite,
            dry_run=bool(getattr(args, "dry_run", False)),
        )
    elif args.subcommand == "upgrade":
        _run_alembic_command("upgrade", args.revision, database=args.database)
    elif args.subcommand == "downgrade":
        _run_alembic_command("downgrade", args.revision, database=args.database)
    elif args.subcommand == "sync-legacy-projects":
        from ispec.db.legacy_sync import sync_legacy_projects

        summary = sync_legacy_projects(
            legacy_url=getattr(args, "legacy_url", None),
            mapping_path=getattr(args, "mapping", None),
            schema_path=getattr(args, "schema", None),
            db_file_path=getattr(args, "database", None),
            project_id=getattr(args, "project_id", None),
            limit=int(getattr(args, "limit", 1000)),
            max_pages=getattr(args, "max_pages", None),
            reset_cursor=bool(getattr(args, "reset_cursor", False)),
            since=getattr(args, "since", None),
            since_pk=getattr(args, "since_pk", None),
            dry_run=bool(getattr(args, "dry_run", False)),
            backfill_missing=bool(getattr(args, "backfill_missing", False)),
            dump_json=getattr(args, "dump_json", None),
        )
        logger.info("legacy projects sync summary: %s", summary)
    elif args.subcommand == "sync-legacy-experiments":
        from ispec.db.legacy_sync import sync_legacy_experiments

        summary = sync_legacy_experiments(
            legacy_url=getattr(args, "legacy_url", None),
            mapping_path=getattr(args, "mapping", None),
            schema_path=getattr(args, "schema", None),
            db_file_path=getattr(args, "database", None),
            experiment_id=getattr(args, "experiment_id", None),
            limit=int(getattr(args, "limit", 1000)),
            dry_run=bool(getattr(args, "dry_run", False)),
            dump_json=getattr(args, "dump_json", None),
        )
        logger.info("legacy experiments sync summary: %s", summary)
    elif args.subcommand == "sync-legacy-experiment-runs":
        from ispec.db.legacy_sync import sync_legacy_experiment_runs

        summary = sync_legacy_experiment_runs(
            legacy_url=getattr(args, "legacy_url", None),
            mapping_path=getattr(args, "mapping", None),
            schema_path=getattr(args, "schema", None),
            db_file_path=getattr(args, "database", None),
            experiment_id=int(getattr(args, "experiment_id")),
            limit=int(getattr(args, "limit", 5000)),
            dry_run=bool(getattr(args, "dry_run", False)),
            dump_json=getattr(args, "dump_json", None),
        )
        logger.info("legacy experiment runs sync summary: %s", summary)
    else:
        logger.info("no dispatched function provided for %s", args.subcommand)


def _render_table_overview(
    table_definitions: Mapping[str, Sequence[Mapping[str, Any]]],
    console: Console | None = None,
) -> None:
    """Pretty-print table metadata using ``rich``."""

    if console is None:
        console = Console()

    table = Table(title="iSPEC Database Schema", show_lines=True)
    table.add_column("Table", style="bold cyan")
    table.add_column("Column", style="magenta")
    table.add_column("Type", style="green")
    table.add_column("Nullable", justify="center", style="yellow")
    table.add_column("Default", style="bright_black")

    table_names = sorted(table_definitions)
    if not table_names:
        table.add_row("[dim]No tables found[/dim]", "", "", "", "")
        console.print(table)
        return

    for table_index, table_name in enumerate(table_names):
        columns = table_definitions[table_name]
        if not columns:
            table.add_row(table_name, "[dim]-[/dim]", "[dim]-[/dim]", "", "")
        else:
            for column_index, column in enumerate(columns):
                default = column.get("default")
                default_display = "" if default in (None, "") else str(default)
                table.add_row(
                    table_name if column_index == 0 else "",
                    str(column.get("name", "")),
                    str(column.get("type", "")),
                    "Yes" if column.get("nullable", True) else "No",
                    default_display,
                )
        if table_index < len(table_names) - 1:
            table.add_section()

    console.print(table)


def _run_alembic_command(action: str, revision: str, database: str | None) -> None:
    """Execute an Alembic migration command."""

    logger = get_logger(__file__)
    config = _build_alembic_config(database)
    logger.info("running alembic %s to %s", action, revision)

    if action == "upgrade":
        command.upgrade(config, revision)
    elif action == "downgrade":
        command.downgrade(config, revision)
    else:  # pragma: no cover - guarded by call sites
        raise ValueError(f"Unsupported Alembic action: {action}")


def _build_alembic_config(database: str | None) -> Config:
    """Create an Alembic :class:`~alembic.config.Config` instance."""

    project_root = _find_project_root()
    config_path = project_root / "alembic.ini"

    alembic_config = Config(str(config_path)) if config_path.exists() else Config()
    alembic_config.set_main_option("script_location", str(project_root / "alembic"))

    normalized = _normalize_database_option(database)
    if normalized:
        alembic_config.set_main_option("sqlalchemy.url", normalized)
    elif not alembic_config.get_main_option("sqlalchemy.url"):
        # Provide an empty value so env.py falls back to get_db_path().
        alembic_config.set_main_option("sqlalchemy.url", "")

    return alembic_config


def _find_project_root() -> Path:
    """Locate the repository root that contains the Alembic directory."""

    for parent in Path(__file__).resolve().parents:
        if (parent / "alembic").is_dir():
            return parent
    raise FileNotFoundError("Could not locate the Alembic directory.")


def _normalize_database_option(database: str | None) -> str | None:
    """Normalize a database argument into an Alembic-friendly URL."""

    if not database:
        return None

    database = database.strip()
    if not database:
        return None

    if "://" in database:
        return database

    expanded = str(Path(database).expanduser())
    if expanded.startswith("sqlite"):
        return expanded

    return f"sqlite:///{expanded}"
