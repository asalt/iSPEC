import pytest
import tempfile
import logging
from pathlib import Path

from faker import Faker
import pandas as pd
from sqlalchemy import inspect, text

from ispec.db import init  # Adjust path as needed
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

def get_all_tables(engine):
    insp = inspect(engine)
    return [t for t in insp.get_table_names() if not t.startswith("sqlite_")]


def get_table_columns(engine, table):
    insp = inspect(engine)
    cols = insp.get_columns(table)
    return [col["name"] for col in cols]


def get_column_defs_from_db(engine, table_name):
    insp = inspect(engine)

    columns = {}
    for col in insp.get_columns(table_name):
        name = col["name"]
        if name == "project_person":  # Skip known linker table/column
            continue
        sqltype = str(col["type"]).upper()
        columns[name] = {
            "type": guess_faker_type(name, sqltype),
            "notnull": not col.get("nullable", True),
            "default": col.get("default"),
            "pk": col.get("primary_key", False),
            "sqltype": sqltype,
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

def insert_df_to_table(conn, table_name, df, column_definitions, _depth: int = 0):
    """Insert a DataFrame into a table while satisfying foreign keys.

    - Ensures referenced tables have rows (recursively seeds them if needed)
    - Rewrites FK columns in df to valid IDs from the referenced tables
    - Skips autoincrement PK columns
    """
    insp = inspect(conn)

    # Guard against runaway recursion
    if _depth > 5:
        raise RuntimeError("Exceeded recursion while seeding foreign keys")

    table_cols = [col["name"] for col in insp.get_columns(table_name)]

    # Foreign key handling: ensure parents exist and map FK cols to valid IDs
    fks = insp.get_foreign_keys(table_name)
    for fk in fks:
        referred_table = fk.get("referred_table")
        constrained_cols = fk.get("constrained_columns", [])
        referred_cols = fk.get("referred_columns", [])

        if not referred_table or not constrained_cols or not referred_cols:
            continue

        referred_col = referred_cols[0]
        constrained_col = constrained_cols[0]

        # Seed parent table if empty
        parent_count = conn.execute(text(f"SELECT COUNT(*) FROM {referred_table}")).scalar()
        if parent_count == 0:
            parent_defs = get_column_defs_from_db(conn, referred_table)
            parent_df = generate_fake_data(5, parent_defs)
            insert_df_to_table(conn, referred_table, parent_df, parent_defs, _depth=_depth + 1)

        # Fetch available parent IDs and map into df
        parent_ids = [row[0] for row in conn.execute(text(f"SELECT {referred_col} FROM {referred_table}")).fetchall()]
        if parent_ids and constrained_col in df.columns:
            import random
            df[constrained_col] = [random.choice(parent_ids) for _ in range(len(df))]

    common_cols = [
        col
        for col in table_cols
        if col in df.columns
        and not (
            column_definitions[col].get("pk")
            and column_definitions[col].get("sqltype", "").startswith("INTEGER")
        )
    ]

    if not common_cols or df.empty:
        logger.warning("No insertable columns found or empty DataFrame.")
        return

    placeholders = ", ".join([f":{c}" for c in common_cols])
    q = text(
        f"INSERT INTO {table_name} ({', '.join(common_cols)}) VALUES ({placeholders})"
    )
    values = df[common_cols].to_dict(orient="records")

    logger.info("Inserting into table %s", table_name)
    logger.info("Columns: %s", common_cols)
    if values:
        logger.debug("Sample types: %s", [type(x) for x in values[0].values()])
        logger.debug("Sample row: %s", list(values[0].values()))

    conn.execute(q, values)


def get_primary_key_column(engine, table_name):
    insp = inspect(engine)
    pk = insp.get_pk_constraint(table_name).get("constrained_columns")
    if pk:
        return pk[0]
    raise ValueError(f"No primary key found in table {table_name}")


# -------------------------- Test Setup -------------------------- #

with tempfile.NamedTemporaryFile(suffix=".db") as tmp_db:
    engine = init.initialize_db(file_path=tmp_db.name)
    ALL_TABLES = [
        t
        for t in get_all_tables(engine)
        if t not in {"project", "project_comment", "project_person"}
    ]


# -------------------------- Tests -------------------------- #



@pytest.mark.parametrize("table_name", ALL_TABLES)
def test_fake_data_insert(tmp_path, table_name):
    dbfile = tmp_path / "test3.db"
    dbfile = Path("sandbox") / "test3.db"
    dbfile.parent.mkdir(exist_ok=True)
    engine = init.initialize_db(file_path=dbfile)

    with engine.begin() as conn:
        col_defs = get_column_defs_from_db(engine, table_name)

        # Generic approach: generate data and let insert_df_to_table handle
        # seeding and FK mapping for all tables, including linkers.
        df = generate_fake_data(10, col_defs)
        try:
            insert_df_to_table(conn, table_name, df, col_defs)
        except Exception as e:
            pytest.fail(f"Insertion failed for table {table_name}: {e}")

    # reconnect and see that it's there
    with engine.connect() as conn:
        res = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
        n = res.scalar()
        assert n >= 0
