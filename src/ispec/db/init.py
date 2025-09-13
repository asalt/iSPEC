# db/init.py 
from pathlib import Path
from functools import lru_cache
from typing import Union

from ispec.db.connect import get_db_path
from ispec.db.models import sqlite_engine, initialize_db as orm_initialize_db



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


def initialize_db(file_path: Union[str, Path, None] = None):
    """Create all database tables using SQLAlchemy models.

    Parameters
    ----------
    file_path:
        Optional path to the SQLite database file. When ``None`` the default
        path returned by :func:`get_db_path` is used.

    Returns
    -------
    Engine
        The SQLAlchemy engine connected to the initialized database.
    """

    db_uri = file_path or get_db_path()
    if not str(db_uri).startswith("sqlite"):
        db_uri = "sqlite:///" + str(db_uri)

    engine = sqlite_engine(db_uri)
    orm_initialize_db(engine)
    return engine

