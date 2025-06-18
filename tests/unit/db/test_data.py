import pytest
import tempfile
import sqlite3
import logging
from pathlib import Path
from faker import Faker
import pandas as pd
from ispec.db import init, connect  # Adjust path as needed
from functools import cache
import difflib

# Create logs directory if needed
Path("logs").mkdir(exist_ok=True)

# # Set up logging
# logging.basicConfig(
#     level=logging.DEBUG,
#     format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
#     handlers=[
#         logging.FileHandler("logs/test_debug.log"),
#         logging.StreamHandler(),  # optional: still logs to console
#     ]
# )
# 
# logger = logging.getLogger(__name__)

# logging long

logger = logging.getLogger("iSPEC")
logger.setLevel(logging.DEBUG)

# Clear any existing handlers added by pytest or elsewhere
if logger.hasHandlers():
    logger.handlers.clear()

# File handler
file_handler = logging.FileHandler("logs/test_debug.log")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter(
    "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
))

# Console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter(
    "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
))

# Add them to your logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# -------------------------- Schema Helpers -------------------------- #

def get_all_tables(db_path):
    with sqlite3.connect(db_path) as conn:
        tables = conn.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name NOT LIKE 'sqlite_%'
        """).fetchall()
    return [row[0] for row in tables]

def get_table_columns(db_path, table):
    with sqlite3.connect(db_path) as conn:  # should use ? holder
        cols = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return [row[1] for row in cols]

def get_column_defs_from_db(db_path, table_name):
    with sqlite3.connect(db_path) as conn:
        pragma = conn.execute(f"PRAGMA table_info({table_name})")

        columns = {}
        for row in pragma:
            cid, name, ctype, notnull, dflt_value, pk = row
            if name == "project_person":  # Skip known linker table/column
                continue
            columns[name] = {
                'type': guess_faker_type(name, ctype),
                'notnull': bool(notnull),
                'default': dflt_value,
                'pk': bool(pk),
                'sqltype': ctype.upper()
            }

    return columns

# -------------------------- Faker Helpers -------------------------- #


faker_instance = Faker()

FAKER_FIELDS = set()
for attr in dir(faker_instance):
    if attr.startswith("_") or attr in ( "seed", "generator"):
        continue
    try:
        if callable(getattr(faker_instance, attr)):
            FAKER_FIELDS.add(attr)
    except TypeError:
        continue

@cache
def guess_faker_type(colname, sqltype):

    # Optionally: whitelist only most useful fields
    name = colname.lower()
    sqltype = sqltype.upper()

    # First: check strong type hints from SQL
    if "REAL" in sqltype:
        return "pyfloat"
    if "INTEGER" in sqltype:
        return "random_int"
    if "name_first" in name:
        return "first_name"
    if "name_last" in name:
        return "last_name"
    if "addedby" in name or "pi" in name:
        return "name"
    if "phone" in name:
        return "phone_number"
    if "email" in name:
        return "email"
    # Second: try exact matches
    if name in FAKER_FIELDS:
        return name

    # Third: fuzzy match to known fields --- HELPS MISSepllings
    close_matches = difflib.get_close_matches(name, FAKER_FIELDS, n=1, cutoff=0.6)
    if close_matches:
        # logger.debug("found a close match %s for %s", (name, close_matches))
        return close_matches[0]

    # Fourth: fallback to "text"
    return "text"

faker_instance = Faker()
def generate_fake_data(n, column_definitions, seed=None):
    if seed is not None:
        Faker.seed(seed)

    f = faker_instance
    records = []

    for _ in range(n):
        row = {}

        for col, opts in column_definitions.items():
            ftype = opts.get('type', 'text')

            # Try to generate a value
            try:
                if hasattr(f, ftype):
                    val = getattr(f, ftype)()
                elif ftype == "random_int":
                    val = f.random_int(min=100000, max=999999)
                elif ftype == "pyfloat":
                    val = f.pyfloat(left_digits=3, right_digits=3)
                elif ftype == "boolean":
                    val = f.boolean()
                else:
                    val = f.text(max_nb_chars=20)
            except Exception as e:
                val = f"[error generating {ftype}]"

            # Ensure SQLite-safe types
            if isinstance(val, (list, tuple, dict)):
                val = str(val)
            elif hasattr(val, "__iter__") and not isinstance(val, (str, bytes)):
                val = str(list(val))  # handle unexpected generators

            row[col] = val

        records.append(row)

    return pd.DataFrame(records)



# -------------------------- Data Insertion -------------------------- #

def insert_df_to_table(conn, table_name, df, column_definitions):
    cursor = conn.execute(f"PRAGMA table_info({table_name})")
    table_cols = [row[1] for row in cursor]

    common_cols = [
        col for col in table_cols
        if col in df.columns and not (
            column_definitions[col].get('pk') and
            column_definitions[col].get('sqltype', '').startswith('INTEGER')
        )
    ]

    if not common_cols or df.empty:
        logger.warning("No insertable columns found or empty DataFrame.")
        return

    placeholders = ", ".join(["?"] * len(common_cols))
    q = f"INSERT INTO {table_name} ({', '.join(common_cols)}) VALUES ({placeholders})"
    logger.info("Query string: "+ q)
    values = df[common_cols].values.tolist()

    logger.info("Inserting into table %s", table_name)
    logger.info("Columns: %s", common_cols)
    logger.debug("Sample types: %s", [type(x) for x in values[0]])
    logger.debug("Sample row: %s", values[0])

    conn.executemany(q, values)

def get_primary_key_column(conn, table_name):
    rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    for row in rows:
        cid, name, col_type, notnull, dflt, pk = row
        if pk == 1:
            return name
    raise ValueError(f"No primary key found in table {table_name}")


# -------------------------- Test Setup -------------------------- #

with tempfile.NamedTemporaryFile(suffix=".db") as tmp_db:
    init.initialize_db(file_path=tmp_db.name)
    ALL_TABLES = get_all_tables(tmp_db.name)


# -------------------------- Tests -------------------------- #



@pytest.mark.parametrize("table_name", ALL_TABLES)
def test_fake_data_insert(tmp_path, table_name):
    dbfile = tmp_path / "test3.db"
    dbfile = Path("sandbox") / "test3.db"
    dbfile.parent.mkdir(exist_ok=True)
    init.initialize_db(file_path=dbfile)

    with connect.get_connection(dbfile) as conn:
        col_defs = get_column_defs_from_db(dbfile, table_name)

        if table_name == "project_person": # we should ignore this and have the database make this relationship itself
            # Insert into referenced tables first
            for ref_table in ("project", "person"):
                ref_defs = get_column_defs_from_db(dbfile, ref_table)
                df_ref = generate_fake_data(5, ref_defs)
                insert_df_to_table(conn, ref_table, df_ref, ref_defs)

            for fk in conn.execute("PRAGMA foreign_key_list(project_person)").fetchall():
                logger.debug("FK: %s", fk)
            logger.debug(conn.execute("PRAGMA table_info(project)").fetchall())


            # Fetch real IDs from DB
            project_pk = get_primary_key_column(conn, "project")
            person_pk = get_primary_key_column(conn, "person")

            project_ids = [row[0] for row in conn.execute(f"SELECT {project_pk} FROM project").fetchall()]
            person_ids = [row[0] for row in conn.execute(f"SELECT {person_pk} FROM person").fetchall()]


            import random
            data = [{
                'project_id': random.choice(project_ids),
                'person_id': random.choice(person_ids)
            } for _ in range(10)]

            df = pd.DataFrame(data)
            insert_df_to_table(conn, table_name, df, col_defs)
        else:
            df = generate_fake_data(10, col_defs)
            try:
                insert_df_to_table(conn, table_name, df, col_defs)
            except Exception as e:
                pytest.fail(f"Insertion failed for table {table_name}: {e}")

    # reconnect and see that it's there
    with connect.get_connection(dbfile) as conn:
        res = conn.execute(f"SELECT COUNT(*) FROM {table_name}")
        n = res.fetchone()[0]
        assert n >= 0

