from ispec.db.init import initialize_db
from ispec.db.connect import get_connection
from ispec.logging import get_logger


logger = get_logger(__file__)

def check_status():
    logger.info("checking db status...")
    sql_file = init.get_sql_file()

def show_tables(file_path=None):
    logger.info("showing tables..")
    with get_connection(file_path) as conn:
        print('hello')

def import_file(file_path):
    print("preparing to import file.. %s", file_path)

def init(file_path=None):
    """
    file_path can be gathered from environment variable or a sensible default if not provided
    """
    return initialize_db(file_path=file_path)
