# io/io_file.py
from functools import partial
import pandas as pd
from ispec.db.connect import get_connection
from ispec.db.crud import Person, TableCRUD, Project, ProjectPerson
import numpy as np

tables = {
    "person" : Person,
    "project" : Project
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
        table_instance = table(conn = conn)
        table_instance.bulk_insert(df_dict)

    #with get_connection(db_path=db_file_path) as conn:
    #    table_obj = table(conn)
    #    #records = list(df.to_dict(orient="index").values())
    #    records = list(df_dict_clean)
    #    table_obj.bulk_insert(records)

    # done
    return

def connect_project_person():
    
    return

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