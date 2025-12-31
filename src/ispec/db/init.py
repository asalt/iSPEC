from pathlib import Path
from typing import Union

from ispec.db.connect import get_db_path
from ispec.db.models import sqlite_engine, initialize_db as orm_initialize_db


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
    db_uri_str = str(db_uri)
    if not db_uri_str.startswith("sqlite"):
        db_uri_str = "sqlite:///" + db_uri_str

    engine = sqlite_engine(db_uri_str)
    orm_initialize_db(engine)
    return engine
