# io/io_file.py
from functools import partial
import pandas as pd
from ispec.db.connect import get_connection
from ispec.db.crud import (
    Person,
    TableCRUD,
    Project,
    ProjectPerson,
    ProjectComment,
    ProjectNote,
    LetterOfSupport,
)
import numpy as np

tables = {
    "person": Person,
    "project": Project,
    "comment": ProjectComment,
    "note": ProjectNote,
    "letter": LetterOfSupport,
}


def get_reader(file: str, **kwargs):
    if file.endswith(".tsv"):
        return partial(pd.read_table, **kwargs)
    elif file.endswith(".csv"):
        return partial(pd.read_csv, **kwargs)
    elif file.endswith(".xlsx"):
        return partial(pd.read_excel, **kwargs)

    else:
        logger.error(f"do not know how to read file: {file}")
        return None


def connect_project_person(db_file_path):
    s = None
    with get_connection(db_file_path) as conn:
        ccc = conn.cursor()
        ccc.execute("PRAGMA foreign_keys;")
        ccc.execute(
            """
        INSERT OR IGNORE INTO project_person (project_id, person_id)
        SELECT project.id, person.id
        FROM project
        JOIN person ON project.id = person.id
        """
        )
        # thank u chat gpt
        # ccc.execute("""
        # SELECT * FROM (SELECT id, project_id FROM project_person) AS a
        # JOIN (SELECT id FROM project) AS b
        # ON a.project_id = b.id""")
        # ccc.execute("""
        # SELECT * FROM (SELECT id, person_id FROM project_person) AS a
        # JOIN (SELECT id FROM person) AS b
        # ON a.person_id = b.id""")
        conn.commit()
        s = ccc.fetchall()
    return s


def connect_project_comment(db_file_path):
    s = None
    with get_connection(db_file_path) as conn:
        ccc = conn.cursor()
        ccc.execute("PRAGMA foreign_keys;")
        ccc.execute(
            """
        UPDATE project_comment
        SET project_id = (
        SELECT project.id
        FROM project
        WHERE project.id = project_comment.i_id
        )
        WHERE EXISTS (
        SELECT 1
        FROM project
        WHERE project.id = project_comment.i_id
        )
       """
        )

        # ccc.execute("""
        # INSERT OR IGNORE INTO project_comment (project_id)
        # SELECT project.id
        # FROM project
        # JOIN project_comment ON project.id = project_comment.i_id
        # """)
        conn.commit()
        s = ccc.fetchall()
    return s


def import_file(file_path, table_name, db_file_path=None, **kwargs):

    reader = get_reader(file_path)
    df = reader(file_path)
    df = df.replace({np.nan: None}).replace({pd.NaT: None})
    df_dict = df.to_dict(orient="index")
    # import pdb; pdb.set_trace()
    # import pdb; pdb.set_trace()
    df_dict = [d for d in df_dict.values()]

    # df_dict_clean = clean_up_import(df_dict,db_file_path,table_name)
    table = tables.get(table_name)
    if table is None:
        raise ValueError(f"No such table {table} in db")

    with get_connection(db_file_path) as conn:
        table_instance = table(conn=conn)
        table_instance.bulk_insert(df_dict)

    if ("project" == table_name) or ("person" == table_name):
        a = connect_project_person(db_file_path)
    if "comment" == table_name:
        a = connect_project_comment(db_file_path)

    # with get_connection(db_path=db_file_path) as conn:
    #    table_obj = table(conn)
    #    #records = list(df.to_dict(orient="index").values())
    #    records = list(df_dict_clean)
    #    table_obj.bulk_insert(records)

    # done
    return


"""
def get_table_colu(db_file_path,table_name):
    if tables.get(table_name) is not None:
        with get_connection(db_path=db_file_path) as conn:
            c = conn.cursor()
            c.execute("SELECT * FROM " + table_name)
            res = c.fetchall()
        return res
        
        

def clean_up_import(dict,db_file_path,table_name):
    checkName = get_table_colu(db_file_path,table_name)
    colsRemove = []
    for bigKey in dict:
        for key in bigKey:
            if "RecNo" in key:
                key = "id"
            if key not in checkName and key != "id":
                colsRemove.append(key)
        for removable in colsRemove:
            bigKey.pop(removable)
    return dict   
"""
