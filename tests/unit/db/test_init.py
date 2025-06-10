import sqlite3
import pytest
import os
from unittest.mock import patch
import shutil
from ispec.db import init, connect


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
    init.initialize_db(file_path=tmp_db)

    with connect.get_connection(tmp_db) as conn:
        # import pdb; pdb.set_trace()
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table';")
        res = cursor.fetchall()
        # list of tuples of table names
    flattened_res = [y for x in res for y in x]
    assert len(flattened_res) == 4
    assert "person" in flattened_res
    assert "project" in flattened_res
    assert "letter_of_support" in flattened_res
    assert "project_person" in flattened_res


@pytest.fixture
def mock_connection():
    from unittest.mock import Mock

    mock_conn = Mock()
    mock_cursor = mock_conn.cursor()
    return mock_conn, mock_cursor


@pytest.fixture
def mock_get_connection(mock_connection):
    from unittest.mock import patch

    with patch("ispec.db.connect.get_connection") as mock:
        mock.return_value = mock_connection[0]
        yield mock


# def test_initialize_db_success(mock_get_connection, mock_connection, mock_cursor):
#     from ispec.db import init
#
#     init.initialize_db()
#
#     mock_cursor.executescript.assert_called_once()
#
#     with open(os.path.join("test_sql_dir", "test_init.sql"), "r") as f:
#         sql_content = f.read()
#
#     mock_cursor.executescript.assert_called_with(sql_content)
