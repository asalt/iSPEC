from sqlalchemy import inspect, text

from ispec.db import init
from ispec.db.models import sqlite_engine, initialize_db


def test_initialize_sqlite_db(tmp_path):
    tmp_db = tmp_path / "test.db"
    engine = init.initialize_db(file_path=tmp_db)

    inspector = inspect(engine)
    tables = inspector.get_table_names()

    assert "person" in tables
    assert "project" in tables
    assert "letter_of_support" in tables
    assert "project_person" in tables


def test_initialize_db_repairs_missing_project_type_column(tmp_path):
    db_url = f"sqlite:///{tmp_path}/legacy.db"
    engine = sqlite_engine(db_url)

    # Simulate a legacy project table missing the enum-backed column.
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE project (id INTEGER PRIMARY KEY AUTOINCREMENT)"))

    inspector = inspect(engine)
    before_columns = {col["name"] for col in inspector.get_columns("project")}
    assert "prj_ProjectType" not in before_columns

    initialize_db(engine)

    inspector = inspect(engine)
    after_columns = {col["name"] for col in inspector.get_columns("project")}
    assert "prj_ProjectType" in after_columns
