from sqlalchemy import text

from ispec.db import init
from ispec.db.init import initialize_db
from ispec.db.connect import get_session
from ispec.logging import get_logger


logger = get_logger(__file__)


def check_status():
    logger.info("checking db status...")
    with get_session() as session:
        result = session.execute(text("SELECT sqlite_version();")).fetchone()
        if result:
            logger.info("sqlite version: %s", result[0])
        else:
            logger.warning("sqlite version query returned no result")
    init.get_sql_file()
    # logger.info(f"Sql file is : {sql_file}")


def show_tables(file_path=None):
    logger.info("showing tables..")
    with get_session(file_path=file_path) as session:
        result = session.execute(
            text("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
        ).fetchall()
        tables = [row[0] for row in result]
        logger.info("tables: %s", tables)


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

