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

    def __init__(self, table_name: str = None, conn: sqlite3.Connection = None):
        self.table = table_name or TABLE
        self.conn = conn

    @cache
    def get_columns(self):
        result = self.conn.execute(f"PRAGMA table_info({self.table})")
        return [row[1] for row in result.fetchall()]

    def validate_input(record: dict):
        if hasattr(self.REQ_COLS, "__iter__"):
            for col in self.REQ_COLS:
                if col not in record.keys():
                    raise ValueError(f"{col} not in input records")

    def insert(self, record: Dict[str, str]) -> int:

        validate_input(record)

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
    REQ_COLS = ("ppl_Name_Last",)

    def insert(self, record: dict):

        validate_input(record)

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

        return super().insert(record)


class Project(TableCRUD):

    TABLE = "project"
    REQ_COLS = (
        "prj_ProjectTitle",
        "prj_ProjectBackground",
    )

    def insert(self, record: dict):
        # rewrite for project specifi
        # check if project title matches an existing project title in the database

        validate_input(record)

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

        return super().insert(record)


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
        if "project_person" == table_name:
            if "project_id" == colname or "person_id" == colname:
                # Insert ID?
                row[col] = hypothetical_id

        return


# ADDING PEOPLE - + checking presence
# ADDING PROJECT - + checkin presence
# MATCHING FOREIGN KEYs people_project, updating tables.
