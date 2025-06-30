# ispec/db/connect.py

import sqlite3
import os
import re
from datetime import datetime
from pathlib import Path
from contextlib import contextmanager

from functools import lru_cache

import pandas as pd

from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.engine import Engine


from ispec.db.models import sqlite_engine, initialize_db
from ispec.logging import get_logger

logger = get_logger(__file__)


@lru_cache(maxsize=None)
def get_db_dir() -> Path:
    db_dir = Path(os.environ.get("ISPEC_DB_DIR", Path.home() / "ispec"))
    db_dir.mkdir(parents=True, exist_ok=True)
    logger.info("setting db_dir to %s", str(db_dir))
    return db_dir


@lru_cache(maxsize=None)
def get_db_path(file=None) -> Path:
    db_file = None
    if file is None:
        db_path = get_db_dir()
        db_file = db_path / "ispec.db"
        db_file = "sqlite:///" + str(db_file)
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


def table_exists(conn, table_name):
    cursor = conn.cursor()
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?;", (table_name,)
    )
    return cursor.fetchone() is not None


CREATE_TABLE_PATTERN = re.compile(r"CREATE TABLE IF NOT EXISTS (\w+)")


# def initialize_db(file_path=None, conn=None):
#     sql_def = get_sql_file()
#     with open(sql_def) as f:
#         sql_cmds = f.read().strip()
#     table_names = CREATE_TABLE_PATTERN.findall(sql_cmds)

#     def should_create(conn):
#         return not all(table_exists(conn, t) for t in table_names)

#     if file_path is not None and conn is None:
#         with get_connection(file_path) as conn:
#             if should_create(conn):
#                 conn.executescript(sql_cmds)
#                 conn.commit()
#         return
#     elif conn is not None:
#         if should_create(conn):
#             conn.executescript(sql_cmds)
#             conn.commit()
#     return


# def adapt_timestamp(ts):  # pandas.timestamp
#     return ts.isoformat()


# def convert_timestamp(s: bytes):
#     return pd.Timestamp(s.decode())


# @contextmanager
# def get_connection(db_path: Path = None):
#     # ensure_db_dir()
#     db_file = Path(db_path) if db_path else get_db_path()

#     logger.info("connecting to db %s", str(db_file))

#     sqlite3.enable_callback_tracebacks(True)
#     conn = sqlite3.connect(
#         str(db_file),
#         check_same_thread=False,
#         detect_types=sqlite3.PARSE_DECLTYPES,
#         # autocommit=False,
#     )
#     conn.execute("PRAGMA foreign_keys = ON")
#     sqlite3.register_adapter(pd.Timestamp, adapt_timestamp)
#     sqlite3.register_converter("TIMESTAMP", convert_timestamp)

#     conn.set_trace_callback(lambda x: logger.info(x))
#     initialize_db(conn=conn)
#     try:
#         yield conn
#     finally:
#         conn.commit()
#         conn.close()


def make_session_factory(engine: Engine):
    SessionLocal = sessionmaker(bind=engine)

    @contextmanager
    def get_session():
        session = SessionLocal()
        try:
            yield session
        finally:
            session.close()

    return get_session


# Session Context Manager
def get_session() -> Session:
    db_path = os.getenv("ISPEC_DB_PATH", get_db_path())
    engine = sqlite_engine(db_path)

    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    try:
        initialize_db(engine=engine)  # <- Call initializer once per session
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# def ensure_db_dir():
#    logger.debug("ensuring db dir")
#    get_db_dir().mkdir(parents=True, exist_ok=True)


# def initialize_db(file_path=None, conn=None):
#
#     sql_def = get_sql_file()
#     with open(sql_def) as f:
#         sql_cmds = f.read().strip()
#         # parse out the table names and check table_exists for eacH?
#         # sql_cmds = f.read().strip().split(';')
#
#     # Note add a check to see if the necessary tables are already created
#     # if so, skip.
#     if file_path is not None and conn is None:
#         with get_connection(file_path) as conn:
#             cursor = conn.cursor()
#             # for sql_cmd in sql_cmds:
#             cursor.executescript(sql_cmds)
#             conn.commit()
#         return
#     elif conn is not None:
#         cursor = conn.cursor()
#         cursor.executescript(sql_cmds)
#         conn.commit()
#     return
