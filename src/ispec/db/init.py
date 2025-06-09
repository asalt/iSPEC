# db/init.py 
from ispec.db.connect import get_connection
from pathlib import Path
from functools import lru_cache



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


def initialize_db(file_path=None):

    sql_def = get_sql_file()
    with open(sql_def) as f:
        sql_cmds = f.read().strip()
        #sql_cmds = f.read().strip().split(';')
    
    with get_connection(file_path) as conn:
        cursor = conn.cursor()
        #for sql_cmd in sql_cmds:
        cursor.executescript(sql_cmds)
        conn.commit()

