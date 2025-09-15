from sqlalchemy import text

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


def show_tables(file_path=None):
    """List all tables in the SQLite database."""
    logger.info("showing tables..")
    with get_session(file_path=file_path) as session:
        result = session.execute(
            text("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
        ).fetchall()
        tables = [row[0] for row in result]
        logger.info("tables: %s", tables)
        return tables


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

