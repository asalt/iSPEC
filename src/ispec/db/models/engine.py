import os
import sqlite3
from typing import Any

import pandas as pd
from sqlalchemy import create_engine, event, inspect, text
from sqlalchemy.engine import Engine

from ispec.logging import get_logger

from .base import Base

logger = get_logger(__file__)


def _sqlite_busy_timeout_ms() -> int:
    raw = (os.getenv("ISPEC_SQLITE_BUSY_TIMEOUT_MS") or "").strip()
    if not raw:
        return 30_000
    try:
        return max(0, int(raw))
    except ValueError:
        return 30_000


def _sqlite_journal_mode() -> str:
    raw = (os.getenv("ISPEC_SQLITE_JOURNAL_MODE") or "WAL").strip().upper()
    allowed = {"WAL", "DELETE", "TRUNCATE", "PERSIST", "MEMORY", "OFF"}
    return raw if raw in allowed else "WAL"


def adapt_timestamp(ts: Any):
    """Adapter for pandas/py datetime objects when writing to SQLite."""
    return ts.isoformat() if hasattr(ts, "isoformat") else str(ts)


def convert_timestamp(s: bytes):
    """Convert SQLite timestamp bytes back to pandas Timestamp."""
    return pd.Timestamp(s.decode())


def sqlite_engine(db_path: str = "sqlite:///./example.db") -> Engine:
    sqlite3.register_adapter(pd.Timestamp, adapt_timestamp)
    sqlite3.register_converter("TIMESTAMP", convert_timestamp)

    engine = create_engine(
        db_path,
        connect_args={
            "check_same_thread": False,
            "detect_types": sqlite3.PARSE_DECLTYPES,
        },
        echo=False,
    )

    trace_sql = os.getenv("ISPEC_SQL_TRACE")

    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        try:
            mode = _sqlite_journal_mode()
            cursor.execute(f"PRAGMA journal_mode={mode}")
            if mode == "WAL":
                cursor.execute("PRAGMA synchronous=NORMAL")
        except Exception:
            pass
        try:
            cursor.execute(f"PRAGMA busy_timeout={_sqlite_busy_timeout_ms()}")
        except Exception:
            pass
        if trace_sql:
            dbapi_connection.set_trace_callback(lambda x: logger.info(x))
        cursor.close()

    return engine


def initialize_db(engine: Engine):
    Base.metadata.create_all(bind=engine)
    _ensure_project_type_column(engine)
    _ensure_project_comment_columns(engine)
    _ensure_legacy_import_tracking_columns(engine)
    _ensure_experiment_columns(engine)
    _ensure_experiment_run_columns(engine)
    _ensure_e2g_columns(engine)
    _ensure_auth_user_columns(engine)


def _ensure_project_type_column(engine: Engine) -> None:
    """Ensure legacy SQLite schemas include newer project columns.

    Older dev databases may have been created before newer project fields were
    introduced. SQLAlchemy won't auto-migrate existing tables, so we add missing
    nullable columns to keep the API usable in-place.
    """

    try:
        columns = {col["name"] for col in inspect(engine).get_columns("project")}
    except Exception:
        return

    missing: list[str] = []
    if "prj_ProjectType" not in columns:
        missing.append("prj_ProjectType")
    if "prj_ProjectPriceLevel" not in columns:
        missing.append("prj_ProjectPriceLevel")

    if not missing:
        return

    with engine.begin() as conn:
        for column in missing:
            conn.execute(text(f'ALTER TABLE project ADD COLUMN "{column}" TEXT'))
    logger.info("Added missing columns project.%s", ", ".join(missing))


def _ensure_project_comment_columns(engine: Engine) -> None:
    """Ensure legacy SQLite schemas include newer project_comment columns."""

    try:
        columns = {col["name"] for col in inspect(engine).get_columns("project_comment")}
    except Exception:
        return

    missing = [c for c in ("com_CommentType", "com_AddedBy") if c not in columns]
    if not missing:
        return

    with engine.begin() as conn:
        for col in missing:
            conn.execute(text(f'ALTER TABLE project_comment ADD COLUMN "{col}" TEXT'))
    logger.info("Added missing columns project_comment.%s", ", ".join(missing))


def _ensure_legacy_import_tracking_columns(engine: Engine) -> None:
    """Ensure legacy SQLite schemas include import tracking timestamp columns."""

    desired = {
        "project": ("prj_LegacyImportTS", "DATETIME"),
        "person": ("ppl_LegacyImportTS", "DATETIME"),
        "project_comment": ("com_LegacyImportTS", "DATETIME"),
        "experiment": ("Experiment_LegacyImportTS", "DATETIME"),
        "experiment_run": ("ExperimentRun_LegacyImportTS", "DATETIME"),
    }

    for table, (column, col_type) in desired.items():
        try:
            columns = {col["name"] for col in inspect(engine).get_columns(table)}
        except Exception:
            continue

        if column in columns:
            continue

        with engine.begin() as conn:
            conn.execute(text(f'ALTER TABLE {table} ADD COLUMN "{column}" {col_type}'))
        logger.info("Added missing column %s.%s", table, column)


def _ensure_experiment_columns(engine: Engine) -> None:
    """Ensure legacy SQLite schemas include newer experiment columns."""

    desired: list[tuple[str, str]] = [
        ("exp_LabelFLAG", "INTEGER NOT NULL DEFAULT 0"),
        ("exp_Type", "TEXT"),
        ("exp_Name", "TEXT"),
        ("exp_Date", "DATETIME"),
        ("exp_PreparationNo", "TEXT"),
        ("exp_CellTissue", "TEXT"),
        ("exp_Genotype", "TEXT"),
        ("exp_Treatment", "TEXT"),
        ("exp_Fractions", "TEXT"),
        ("exp_Lysis", "TEXT"),
        ("exp_DTT", "BOOLEAN NOT NULL DEFAULT 0"),
        ("exp_IAA", "BOOLEAN NOT NULL DEFAULT 0"),
        ("exp_Amount", "TEXT"),
        ("exp_Adjustments", "TEXT"),
        ("exp_Batch", "TEXT"),
        ("exp_Data_FLAG", "BOOLEAN NOT NULL DEFAULT 0"),
        ("exp_exp2gene_FLAG", "BOOLEAN NOT NULL DEFAULT 0"),
        ("exp_Description", "TEXT"),
    ]

    try:
        columns = {col["name"] for col in inspect(engine).get_columns("experiment")}
    except Exception:
        return

    missing = [(name, ddl) for (name, ddl) in desired if name not in columns]
    if not missing:
        return

    with engine.begin() as conn:
        for name, ddl in missing:
            conn.execute(text(f'ALTER TABLE experiment ADD COLUMN "{name}" {ddl}'))

    logger.info(
        "Added missing columns experiment.%s", ", ".join(name for name, _ in missing)
    )


def _ensure_experiment_run_columns(engine: Engine) -> None:
    """Ensure legacy SQLite schemas include newer experiment_run columns."""

    desired: list[tuple[str, str]] = [
        ("ms_instrument", "TEXT"),
        ("acquisition_mode", "TEXT"),
        ("ref_database", "TEXT"),
        ("taxon_id", "INTEGER"),
    ]

    try:
        columns = {col["name"] for col in inspect(engine).get_columns("experiment_run")}
    except Exception:
        return

    missing = [(name, ddl) for (name, ddl) in desired if name not in columns]
    if not missing:
        return

    with engine.begin() as conn:
        for name, ddl in missing:
            conn.execute(text(f'ALTER TABLE experiment_run ADD COLUMN "{name}" {ddl}'))

    logger.info(
        "Added missing columns experiment_run.%s", ", ".join(name for name, _ in missing)
    )


def _ensure_e2g_columns(engine: Engine) -> None:
    """Ensure legacy SQLite schemas include newer experiment_to_gene columns."""

    desired: list[tuple[str, str]] = [
        ("gene_symbol", "TEXT"),
        ("description", "TEXT"),
        ("taxon_id", "INTEGER"),
        ("sra", "TEXT"),
        ("psms", "INTEGER"),
        ("psms_u2g", "INTEGER"),
        ("peptide_count", "INTEGER"),
        ("peptide_count_u2g", "INTEGER"),
        ("coverage", "FLOAT"),
        ("coverage_u2g", "FLOAT"),
        ("area_sum_u2g_0", "FLOAT"),
        ("area_sum_u2g_all", "FLOAT"),
        ("area_sum_max", "FLOAT"),
        ("area_sum_dstrAdj", "FLOAT"),
        ("iBAQ_dstrAdj", "FLOAT"),
        ("peptideprint", "TEXT"),
        ("metadata_json", "TEXT"),
    ]

    try:
        columns = {col["name"] for col in inspect(engine).get_columns("experiment_to_gene")}
    except Exception:
        return

    missing = [(name, ddl) for (name, ddl) in desired if name not in columns]
    if not missing:
        return

    with engine.begin() as conn:
        for name, ddl in missing:
            conn.execute(text(f'ALTER TABLE experiment_to_gene ADD COLUMN "{name}" {ddl}'))

    logger.info(
        "Added missing columns experiment_to_gene.%s", ", ".join(name for name, _ in missing)
    )


def _ensure_auth_user_columns(engine: Engine) -> None:
    """Ensure legacy SQLite schemas include newer auth_user columns."""

    desired: list[tuple[str, str]] = [
        ("must_change_password", "BOOLEAN NOT NULL DEFAULT 0"),
        ("last_login_at", "DATETIME"),
        ("password_changed_at", "DATETIME"),
    ]

    try:
        columns = {col["name"] for col in inspect(engine).get_columns("auth_user")}
    except Exception:
        return

    missing = [(name, ddl) for (name, ddl) in desired if name not in columns]
    if not missing:
        return

    with engine.begin() as conn:
        for name, ddl in missing:
            conn.execute(text(f'ALTER TABLE auth_user ADD COLUMN "{name}" {ddl}'))

    logger.info(
        "Added missing columns auth_user.%s", ", ".join(name for name, _ in missing)
    )
