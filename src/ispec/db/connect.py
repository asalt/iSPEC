# ispec/db/connect.py

import sqlite3
import os
from pathlib import Path
from contextlib import contextmanager

from functools import lru_cache

from ispec.logging import get_logger

logger = get_logger(__file__)

@lru_cache(maxsize=None)
def get_db_dir() -> Path:
    db_dir = Path(os.environ.get("ISPEC_DB_DIR", Path.home() / ".ispec"))
    logger.info("setting db_dir to %s", str(db_dir))
    return db_dir

@lru_cache(maxsize=None)
def get_db_path() -> Path:
    db_path = get_db_dir() / "ispec.db"
    logger.info("setting db_path to %s", str(db_path))
    return db_path

def ensure_db_dir():
    logger.debug("ensuring db dir")
    get_db_dir().mkdir(parents=True, exist_ok=True)

@contextmanager
def get_connection(db_path: Path = None):
    ensure_db_dir()
    db_file = Path(db_path) if db_path else get_db_path()

    logger.info("connecting to db %s", str(db_file))

    sqlite3.enable_callback_tracebacks(True)
    conn = sqlite3.connect(str(db_file),
        check_same_thread=False,
        # autocommit=False,
    )
    conn.execute("PRAGMA foreign_keys = ON")
    conn.set_trace_callback(lambda x: logger.info(x))
    try:
        yield conn
    finally:
        conn.close()

