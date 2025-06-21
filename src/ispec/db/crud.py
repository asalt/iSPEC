# crud.py
from functools import cache
import sqlite3
from ispec.db.connect import get_connection

from ispec.logging import get_logger
from typing import Dict, Optional

# create reference? update delete


logger = get_logger(__file__)


@cache
def get_table_cols(conn, table):
    return conn.execute(
        "PRAGMA table_info(?)",
    ).fetchall()


def add_person(person_dict: dict, conn: sqlite3.Connection):
    # Normalize case
    last_name = person_dict.get("ppl_Name_Last", "").strip().lower()
    first_name = person_dict.get("ppl_Name_First", "").strip().lower()

    # Check for existing person (case-insensitive)
    result = conn.execute(
        "SELECT id FROM person WHERE LOWER(ppl_Name_Last) = ? AND LOWER(ppl_Name_First) = ? LIMIT 1",
        (last_name, first_name),
    ).fetchone()

    if result:
        logger.info(
            f"Person with last name '{last_name}' already exists (ID {result[0]}). Skipping insert."
        )
        return result[0]  # existing ID

    # Insert
    keys = person_dict.keys()
    values = [person_dict[k] for k in keys]
    placeholders = ", ".join(["?"] * len(keys))
    columns = ", ".join(keys)

    conn.execute(f"INSERT INTO person ({columns}) VALUES ({placeholders})", values)
    conn.commit()
    logger.info(f"Added person: {person_dict}")

    # Return last inserted ID
    new_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    return new_id


def update_project_person_table(conn, project_id=None, person_id=None):
    result = conn.execute("SELECT id from project where id = ?", project_id).fetchall()
    if not result:
        add_project(project_dict=dict(id=project_id))


class TableCRUD:

    TABLE = None
    REQ_COLS = None

    def __init__(self, conn: sqlite3.Connection = None, table_name: str = None):
        self.table = table_name or self.TABLE
        self.conn = conn

    @cache
    def get_columns(self):
        result = self.conn.execute(f"PRAGMA table_info({self.table})")
        return [row[1] for row in result.fetchall()]

    def validate_input(self, record: dict):
        if hasattr(self.REQ_COLS, "__iter__"):
            for col in self.REQ_COLS:
                if col not in record.keys():
                    raise ValueError(f"{col} not in input records")
        cleaned_record = {k:v for k,v in record.items() if k in self.get_columns()}
        logger.debug(f"cleaned record keys {cleaned_record.keys()}")
        return cleaned_record


    def insert(self, record: Dict[str, str]) -> int:

        self.validate_input(record)

        keys = list(record.keys())
        values = [record[k] for k in keys]
        placeholders = ", ".join(["?"] * len(keys))
        columns = ", ".join(keys)

        self.conn.execute(
            f"INSERT INTO {self.table} ({columns}) VALUES ({placeholders})", values
        )
        self.conn.commit()
        logger.info(f"Inserted into {self.table}: {record}")
        return self.conn.execute("SELECT last_insert_rowid()").fetchone()[0]

    def bulk_insert(self, records: list[dict]) -> None:
        # needs testing
        if not records:
            return
        cleaned_records = list(filter(None, [self.validate_input(record) for record in records]))
        if not cleaned_records:
            return
        # import pdb; pdb.set_trace()
        #self.validate_input(records[0])  # assume all rows follow same structure

        keys = list(cleaned_records[0].keys())
        placeholders = ", ".join(["?"] * len(keys))
        columns = ", ".join(keys)

        values = [[r[k] for k in keys] for r in cleaned_records]
        self.conn.executemany(
            f"INSERT INTO {self.table} ({columns}) VALUES ({placeholders})", values
        )
        self.conn.commit()

    def get_by_id(self, record_id: int) -> Optional[Dict[str, str]]:
        result = self.conn.execute(
            f"SELECT * FROM {self.table} WHERE id = ?", (record_id,)
        ).fetchone()
        return dict(result) if result else None

    def delete_by_id(self, record_id: int) -> bool:
        result = self.conn.execute(
            f"DELETE FROM {self.table} WHERE id = ?", (record_id,)
        )
        self.conn.commit()
        return result.rowcount > 0


class Person(TableCRUD):

    TABLE = "person"
    REQ_COLS = ("ppl_Name_Last", "ppl_Name_First")

    def validate_input(self, record: dict):
        record = super().validate_input(record)
        last_name = record.get("ppl_Name_Last", "")
        first_name = record.get("ppl_Name_First", "")
        if not last_name:
            return None
        last_name = last_name.strip().lower()
        first_name = first_name.strip().lower()


        # Check for existing person (case-insensitive)
        result = self.conn.execute(
            "SELECT id FROM person WHERE LOWER(ppl_Name_Last) = ? AND LOWER(ppl_Name_First) = ? LIMIT 1",
            (last_name, first_name),
        ).fetchone()
        # import pdb; pdb.set_trace()

        if result:
            logger.info(
                f"Person with last name '{last_name}' already exists (ID {result[0]}). Skipping insert."
            )
            return None
            #return result[0]  # existing ID

        return record


class Project(TableCRUD):

    TABLE = "project"
    REQ_COLS = (
        "prj_ProjectTitle",
        "prj_ProjectBackground",
    )

    def validate_input(self, record: dict):
        # rewrite for project specifi
        # check if project title matches an existing project title in the database

        record = super().validate_input(record)

        proj_ProjectTitle = record.get("prj_ProjectTitle", "").strip().lower()
        proj_ProjectBackground = (
            record.get("proj_ProjectBackground", "").strip().lower()
        )
        ##

        # Check for existing person (case-insensitive)
        result = self.conn.execute(
            "SELECT id FROM project WHERE prj_ProjectTitle = ? LIMIT 1",
            (proj_ProjectTitle,),
        ).fetchone()

        if result:
            logger.info(
                f"Project with title '{proj_ProjectTitle}' already exists (ID {result[0]}). Skipping insert."
            )
            return None
            #return result[0]  # existing ID

        return record


class ProjectPerson(TableCRUD):
    TABLE = "project_person"
    REQ_COLS = ("person_id", "project_id")

    def validate_input(self, record: dict):
        super().validate_input(record)

        person_id = record.get("person_id")
        project_id = record.get("project_id")

        person_query = self.conn.execute(
            "SELECT id FROM person where id = ?", (person_id,)
        ).fetchone()

        logger.info(f"person query {person_query}")
        if person_query is None:
            logger.error(
                f"Tried to make a project person link with an invalid person id {person_id}"
            )
            raise ValueError(f"{person_id} not present in person table")

        project_query = self.conn.execute(
            "SELECT id FROM project where id = ?", (project_id,)
        ).fetchone()

        logger.info(f"project query {project_query}")
        if project_query is None:
            logger.error(
                f"Tried to make a project person link with an invalid project id {project_id}"
            )
            raise ValueError(f"{project_id} not present in projecttable")


# ADDING DATA TO TABLES.
def insert_df_to_table(conn, table_name, df, column_definitions):
    cursor = conn.execute(f"PRAGMA table_info({table_name})")
    table_cols = [row[1] for row in cursor]

    # checking for valid columns. uses primary key id to verify identity as a table.
    common_cols = [
        col
        for col in table_cols
        if col in df.columns
        and not (
            column_definitions[col].get("pk")
            and column_definitions[col].get("sqltype", "").startswith("INTEGER")
        )
    ]

    # return statement if it fails to find requested table/columns
    if not common_cols or df.empty:
        logger.warning("No insertable columns found or empty DataFrame.")
        return

    placeholders = ", ".join(["?"] * len(common_cols))
    q = f"INSERT INTO {table_name} ({', '.join(common_cols)}) VALUES ({placeholders})"
    values = df[common_cols].values.tolist()

    logger.info("Inserting into table %s", table_name)
    logger.info("Columns: %s", common_cols)
    logger.debug("Sample types: %s", [type(x) for x in values[0]])
    logger.debug("Sample row: %s", values[0])

    conn.executemany(q, values)


# returning data to user from the table/column/ probably not necessary.
def read_data_from_table(conn, table_name, column_definitions):
    cursor = conn.execute(
        f"PRAGMA table_info({table_name})"
    )  # sets up connection through cursor
    table_cols = [row[1] for row in cursor]

    # req_data = 0.1 #place holder data
    return req_data


# IDK if possible, but if necessary, removing data from db
def delete_data_from_table(conn, table_name, column_definitions):
    return


# replacing data w/ input from user into the table/column
def modify_data_to_table(conn, table_name, column_definitions, data_mod):
    return


# How do I test this?


# EXECUTE THIS EVERYTIME A NEW PROJECT OR PERSON IS ADDED
# INSERT DIFFERENT TABLE NAME BASED ON WHAT IS BEING ADDED, ex you're adding a person: table_name will be person
# Make sure the table name added
def connect_project_person(conn, table_name, column_definitions):
    def read_data_from_table(conn, table_name, column_definitions):
        cursor = conn.execute(
            f"PRAGMA table_info({table_name})"
        )  # sets up connection through cursor
        table_cols = [row[1] for row in cursor]

        if table_name == "project":
            # Make sure it is not empty
            # check for new project
            if colname == "prj_CreationTS":
                # find most newest row next
                explore_this_row = hypothetical_row

            # utilizing explore_this_row
            # find all people associated
            # make sure to form an ID between those people and the project
        if table_name == "person":
            pass
            # Make sure it is not empty
            # check for all the people person - potentially multiple IDS?
            # find projects associated with person.
        ## NEED TO USE AN ACTUAL FUNCTION to modify, not if -- statement
        # UPDATE people/project collums.


        return


# ADDING PEOPLE - + checking presence
# ADDING PROJECT - + checkin presence
# MATCHING FOREIGN KEYs people_project, updating tables.
