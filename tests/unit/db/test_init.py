import os
import sqlite3
import shutil

import pytest
from sqlalchemy import inspect

from ispec.db import init


# Fixture to create a temporary SQL file and directory
# we are not using this yet
@pytest.fixture(autouse=True)
def setup_teardown():
    test_sql_file = "test_init.sql"
    test_sql_dir = "test_sql_dir"

    # Create the test file and directory
    with open(test_sql_file, "w") as f:
        f.write("CREATE TABLE test_table (id INTEGER);")
    os.makedirs(test_sql_dir, exist_ok=True)
    os.rename(test_sql_file, os.path.join(test_sql_dir, test_sql_file))

    yield  # Provide the test environment

    # Teardown: Remove the test file and directory
    shutil.rmtree(test_sql_dir, ignore_errors=True)


def test_get_sql_defs_success():

    from ispec.db import init  # Import inside the test function to use the fixture

    sql_dir_path = init.get_sql_code_dir(path="sqlite")
    assert os.path.exists(sql_dir_path)

    sql_file = init.get_sql_file(path="sqlite")
    assert os.path.exists(sql_file)


def test_get_sql_defs_file_not_found():
    from ispec.db import init

    with pytest.raises(ValueError) as excinfo:
        init.get_sql_file(path="nonexistent_dir")
    assert "sql script path" in str(excinfo.value)


def test_get_sql_file():
    """
    checks validity of file
    tests sqlite3.complete_statement
    """

    from ispec.db import init

    sql_code_file = init.get_sql_file()
    assert sql_code_file is not None

    with open(sql_code_file) as f:
        sql_code_str = f.read()

    assert sqlite3.complete_statement(sql_code_str)


def test_initialize_sqlite_db(tmp_path):
    tmp_db = tmp_path / "test.db"
    engine = init.initialize_db(file_path=tmp_db)

    inspector = inspect(engine)
    tables = inspector.get_table_names()

    assert "person" in tables
    assert "project" in tables
    assert "letter_of_support" in tables
    assert "project_person" in tables


