import os
import sqlite3
from typing import Any

import pandas as pd
from sqlalchemy import create_engine, event, inspect, text
from sqlalchemy.engine import Engine

from ispec.logging import get_logger

from .base import Base

logger = get_logger(__file__)


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
        if trace_sql:
            dbapi_connection.set_trace_callback(lambda x: logger.info(x))
        cursor.close()

    return engine


def initialize_db(engine: Engine):
    Base.metadata.create_all(bind=engine)
    _ensure_project_type_column(engine)


def _ensure_project_type_column(engine: Engine) -> None:
    """Ensure the legacy SQLite schema includes ``project.prj_ProjectType``.

    Older dev databases may have been created before the enum-backed project type
    field was introduced. SQLAlchemy won't auto-migrate existing tables, so we
    add the missing nullable column to keep the API usable in-place.
    """

    try:
        columns = {col["name"] for col in inspect(engine).get_columns("project")}
    except Exception:
        return

    if "prj_ProjectType" in columns:
        return

    with engine.begin() as conn:
        conn.execute(text('ALTER TABLE project ADD COLUMN "prj_ProjectType" TEXT'))
    logger.info("Added missing column project.prj_ProjectType")
