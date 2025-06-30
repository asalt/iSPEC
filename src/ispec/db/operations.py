# from ispec.db import init
# from ispec.db.init import initialize_db

# from ispec.db.connect import get_connection
from ispec.db.connect import get_session
from ispec.logging import get_logger


logger = get_logger(__file__)


def check_status():
    logger.info("checking db status...")
    with get_session() as session:
        # session = get_connection()
        # logger.info(f"Session is : {session}")
        # write sql logic to check db status
        # e.g. session.execute("SELECT sqlite_version();")
        # for row in session.fetchall():
        #     print(row)
        pass
    sql_file = init.get_sql_file()
    # logger.info(f"Sql file is : {sql_file}")


def show_tables(file_path=None):
    logger.info("showing tables..")
    with get_session(file_path=file_path) as session:
        # session = get_connection(file_path)
        # logger.info(f"Session is : {session}")
        # write sql logic to display all tables
        # e.g. session.execute("SELECT name FROM sqlite_master WHERE type='table';")
        # for row in session.fetchall():
        #     print(row)
        pass


def import_file(file_path):
    from ispec.io import io_file

    print("preparing to import file.. %s", file_path)
    io_file.import_file(file_path)
    # need to validate the file input, and understand which table we are meant to update


def initialize(file_path=None):
    """
    file_path can be gathered from environment variable or a sensible default if not provided
    """
    return initialize_db(file_path=file_path)
