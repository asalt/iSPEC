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
    else:
        db_file = Path(file)
    db_uri = "sqlite:///" + str(db_file)
    logger.info("setting db_path to %s", db_uri)
    return db_uri


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




def make_session_factory(engine: Engine):
    SessionLocal = sessionmaker(bind=engine)
    initialize_db(engine=engine)

    @contextmanager
    def get_session():
        session = SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    return get_session


# Session Context Manager
@contextmanager
def get_session(file_path: str | Path | None = None) -> Session:
    """Provide a transactional scope around a series of operations.

    Parameters
    ----------
    file_path:
        Optional path to the SQLite database file. If not provided, the
        ``ISPEC_DB_PATH`` environment variable or the default path from
        :func:`get_db_path` is used.
    """

    db_path = os.getenv("ISPEC_DB_PATH") if file_path is None else file_path
    if db_path is None:
        db_path = get_db_path()
    # ensure sqlite URI prefix
    db_uri = str(db_path)
    if not str(db_uri).startswith("sqlite"):
        db_uri = "sqlite:///" + str(db_uri)

    engine = sqlite_engine(db_uri)
    initialize_db(engine=engine)

    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    try:
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


