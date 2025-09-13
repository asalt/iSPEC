# io/io_file.py
from functools import partial

import numpy as np
import pandas as pd
from sqlalchemy import text

from ispec.db.connect import get_session
from ispec.db.crud import (
    PersonCRUD,
    ProjectCRUD,
    ProjectCommentCRUD,
    LetterOfSupportCRUD,
)
from ispec.logging import get_logger

logger = get_logger(__file__)

tables = {
    "person": PersonCRUD(),
    "project": ProjectCRUD(),
    "comment": ProjectCommentCRUD(),
    "letter": LetterOfSupportCRUD(),
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
    with get_session(file_path=db_file_path) as session:
        session.execute(
            text(
                """
            INSERT OR IGNORE INTO project_person (project_id, person_id)
            SELECT project.id, person.id
            FROM project
            JOIN person ON project.id = person.id
            """
            )
        )


def connect_project_comment(db_file_path):
    with get_session(file_path=db_file_path) as session:
        session.execute(
            text(
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
        )


def import_file(file_path, table_name, db_file_path=None, **kwargs):

    reader = get_reader(file_path)
    df = reader(file_path)
    df = df.replace({np.nan: None}).replace({pd.NaT: None})
    df_dict = df.to_dict(orient="records")

    table_crud = tables.get(table_name)
    if table_crud is None:
        raise ValueError(f"No such table {table_name} in db")

    with get_session(file_path=db_file_path) as session:
        table_crud.bulk_create(session, df_dict)

    if table_name in {"project", "person"}:
        connect_project_person(db_file_path)
    if table_name == "comment":
        connect_project_comment(db_file_path)

    return


"""
def get_table_colu(db_file_path,table_name):
    if tables.get(table_name) is not None:
        with sqlite3.connect(db_file_path) as conn:
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
