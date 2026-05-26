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


def test_initialize_db_repairs_experiment_run_sample_columns(tmp_path):
    db_url = f"sqlite:///{tmp_path}/legacy-runs.db"
    engine = sqlite_engine(db_url)

    with engine.begin() as conn:
        conn.execute(
            text(
                """
                CREATE TABLE experiment_run (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id INTEGER NOT NULL,
                    run_no INTEGER NOT NULL DEFAULT 1,
                    search_no INTEGER NOT NULL DEFAULT 1,
                    label TEXT NOT NULL DEFAULT '0'
                )
                """
            )
        )
        conn.execute(
            text(
                """
                INSERT INTO experiment_run (experiment_id, run_no, search_no, label)
                VALUES (56774, 1, 7, 'labelnone')
                """
            )
        )

    initialize_db(engine)

    inspector = inspect(engine)
    after_columns = {col["name"] for col in inspector.get_columns("experiment_run")}
    assert "sample_name" in after_columns
    assert "sample_group" in after_columns
    assert "sample_metadata_json" in after_columns

    with engine.begin() as conn:
        sample_name = conn.execute(
            text("SELECT sample_name FROM experiment_run WHERE experiment_id = 56774")
        ).scalar_one()
    assert sample_name == "56774_1_7_0"
