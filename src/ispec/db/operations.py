from typing import Any

from sqlalchemy import inspect, text
import pandas as pd

from ispec.db.init import initialize_db
from ispec.db.connect import get_session
from ispec.logging import get_logger


logger = get_logger(__file__)


def check_status():
    """Query the database for its SQLite version and log/return it."""
    logger.info("checking db status...")
    with get_session() as session:
        result = session.execute(text("SELECT sqlite_version();")).fetchone()
        if result:
            version = result[0]
            logger.info("sqlite version: %s", version)
            return version
        logger.warning("sqlite version query returned no result")
        return None


def show_tables(file_path: str | None = None) -> dict[str, list[dict[str, Any]]]:
    """Return table and column metadata for the SQLite database.

    Parameters
    ----------
    file_path:
        Optional path to the SQLite database file. When ``None`` the default
        configuration from :func:`ispec.db.connect.get_session` is used.

    Returns
    -------
    dict
        Mapping of table names to a list of column definitions. Each column
        definition contains ``name``, ``type``, ``nullable`` and ``default``
        keys.
    """

    logger.info("showing tables..")
    with get_session(file_path=file_path) as session:
        inspector = inspect(session.bind)
        table_names = sorted(inspector.get_table_names())
        logger.info("tables: %s", table_names)

        table_definitions: dict[str, list[dict[str, Any]]] = {}
        for table_name in table_names:
            column_details: list[dict[str, Any]] = []
            for column in inspector.get_columns(table_name):
                column_details.append(
                    {
                        "name": column.get("name", ""),
                        "type": str(column.get("type", "")),
                        "nullable": bool(column.get("nullable", True)),
                        "default": column.get("default"),
                    }
                )
            table_definitions[table_name] = column_details

        return table_definitions


def import_file(file_path, table_name, db_file_path=None):
    from ispec.io import io_file

    logger.info("preparing to import file.. %s", file_path)
    io_file.import_file(file_path, table_name, db_file_path=db_file_path)
    # need to validate the file input, and understand which table we are meant to update


def initialize(file_path=None):
    """
    file_path can be gathered from environment variable or a sensible default if not provided
    """
    return initialize_db(file_path=file_path)


def export_table(table_name: str, file_path: str) -> None:
    """Export a database table to a CSV file.

    Parameters
    ----------
    table_name:
        Name of the table to export.
    file_path:
        Destination path for the CSV file.
    """

    logger.info("exporting table %s to %s", table_name, file_path)
    with get_session() as session:
        df = pd.read_sql_table(table_name, session.bind)
    df.to_csv(file_path, index=False)

