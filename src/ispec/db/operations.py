from ispec.db import init
from ispec.db.init import initialize_db
from ispec.db.connect import get_connection
from ispec.logging import get_logger


logger = get_logger(__file__)

def check_status():
    logger.info("checking db status...")
    sql_file = init.get_sql_file()
    # logger.info(f"Sql file is : {sql_file}")

def show_tables(file_path=None):
    logger.info("showing tables..")
    with get_connection(file_path) as conn:
        print('hello')
        # write sql logic to display all tables

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
