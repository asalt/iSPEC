# ispec/db/connect.py

import sqlite3
import os
from datetime import datetime
from pathlib import Path
from contextlib import contextmanager

from functools import lru_cache

import pandas as pd

from ispec.logging import get_logger

logger = get_logger(__file__)


@lru_cache(maxsize=None)
def get_db_dir() -> Path:
    db_dir = Path(os.environ.get("ISPEC_DB_DIR", Path.home() / ".ispec"))
    db_dir.mkdir(parents=True, exist_ok=True)
    logger.info("setting db_dir to %s", str(db_dir))
    return db_dir


@lru_cache(maxsize=None)
def get_db_path(file=None) -> Path:
    if file is None:
        db_path = get_db_dir()
        db_file = db_path / "ispec.db"
    logger.info("setting db_path to %s", str(db_file))
    return db_file


@lru_cache(maxsize=None)
def get_sql_code_dir(path="sqlite"):
    sql_code_path = Path(__file__).parent.parent.parent.parent / "sql" / path
    if not sql_code_path.exists():
        raise ValueError(f"sql script path {sql_code_path} does not exist")
    return sql_code_path


@lru_cache(maxsize=None)
def get_sql_file(**kwargs):
    """
    **kwargs passed to get_sql_dir
    """
    sql_code_path = get_sql_code_dir(**kwargs)
    sql_code_file = sql_code_path / "init.sql"
    if not sql_code_file.exists():
        raise ValueError(f"sql script file {sql_code_file} does not exist")
    return sql_code_file


def initialize_db(file_path=None, conn=None):

    sql_def = get_sql_file()
    with open(sql_def) as f:
        sql_cmds = f.read().strip()
        # sql_cmds = f.read().strip().split(';')

    # Note add a check to see if the necessary tables are already created
    # if so, skip.
    if file_path is not None and conn is None:
        with get_connection(file_path) as conn:
            cursor = conn.cursor()
            # for sql_cmd in sql_cmds:
            cursor.executescript(sql_cmds)
            conn.commit()
        return
    elif conn is not None:
        cursor = conn.cursor()
        cursor.executescript(sql_cmds)
        conn.commit()
    return


# def ensure_db_dir():
#    logger.debug("ensuring db dir")
#    get_db_dir().mkdir(parents=True, exist_ok=True)


def adapt_timestamp(ts):  # pandas.timestamp
    return ts.isoformat()


def convert_timestamp(s: bytes):
    return pd.Timestamp(s.decode())


@contextmanager
def get_connection(db_path: Path = None):
    # ensure_db_dir()
    db_file = Path(db_path) if db_path else get_db_path()

    logger.info("connecting to db %s", str(db_file))

    sqlite3.enable_callback_tracebacks(True)
    conn = sqlite3.connect(
        str(db_file),
        check_same_thread=False,
        detect_types=sqlite3.PARSE_DECLTYPES,
        # autocommit=False,
    )
    conn.execute("PRAGMA foreign_keys = ON")
    sqlite3.register_adapter(pd.Timestamp, adapt_timestamp)
    sqlite3.register_converter("TIMESTAMP", convert_timestamp)

    conn.set_trace_callback(lambda x: logger.info(x))
    initialize_db(conn=conn)
    try:
        yield conn
    finally:
        conn.commit()
        conn.close()
