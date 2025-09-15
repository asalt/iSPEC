import sys
from pathlib import Path

import pandas as pd
import pytest

# Ensure the src directory is on the Python path for direct imports
SRC_PATH = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(SRC_PATH))

from ispec.io import io_file
from ispec.db.connect import get_session
from ispec.db.models import Person, ProjectPerson
from ispec.db import operations


def test_import_person_csv_inserts_records(tmp_path):
    db_path = tmp_path / 'test.db'
    csv_path = tmp_path / 'people.csv'
    df = pd.DataFrame([
        {'ppl_Name_Last': 'Doe', 'ppl_Name_First': 'John', 'ppl_AddedBy': 'tester'}
    ])
    df.to_csv(csv_path, index=False)

    io_file.import_file(str(csv_path), 'person', db_file_path=str(db_path))

    with get_session(file_path=str(db_path)) as session:
        people = session.query(Person).all()
        assert len(people) == 1
        person = people[0]
        assert person.ppl_Name_Last == 'Doe'
        assert person.ppl_Name_First == 'John'

    csv_path.unlink()
    db_path.unlink()
    assert not csv_path.exists()
    assert not db_path.exists()


def test_operations_initialize_creates_db_file(tmp_path):
    db_path = tmp_path / 'cli.db'
    operations.initialize(file_path=str(db_path))
    assert db_path.exists()
    db_path.unlink()
    assert not db_path.exists()


def test_import_file_unsupported_extension_logs_error(tmp_path, caplog):
    db_path = tmp_path / 'test.db'
    bad_file = tmp_path / 'data.txt'
    bad_file.write_text('content')

    io_file.logger.addHandler(caplog.handler)
    with caplog.at_level("ERROR", logger=io_file.logger.name):
        io_file.import_file(str(bad_file), 'person', db_file_path=str(db_path))
    io_file.logger.removeHandler(caplog.handler)

    assert "Unsupported file extension" in caplog.text
    bad_file.unlink()
    if db_path.exists():
        db_path.unlink()
    assert not bad_file.exists()
    assert not db_path.exists()


@pytest.mark.parametrize(
    "suffix,writer",
    [
        (".tsv", lambda df, path: df.to_csv(path, sep="\t", index=False)),
        (".xlsx", lambda df, path: df.to_excel(path, index=False)),
    ],
)
def test_import_person_tsv_xlsx_inserts_records(tmp_path, suffix, writer):
    db_path = tmp_path / "test.db"
    file_path = tmp_path / f"people{suffix}"
    df = pd.DataFrame([
        {"ppl_Name_Last": "Doe", "ppl_Name_First": "Jane", "ppl_AddedBy": "tester"}
    ])
    writer(df, file_path)

    io_file.import_file(str(file_path), "person", db_file_path=str(db_path))

    with get_session(file_path=str(db_path)) as session:
        assert session.query(Person).count() == 1

    file_path.unlink()
    db_path.unlink()
    assert not file_path.exists()
    assert not db_path.exists()


def test_connect_project_person_populates_join_table(tmp_path):
    db_path = tmp_path / "test.db"
    project_file = tmp_path / "project.tsv"
    person_file = tmp_path / "person.tsv"

    project_df = pd.DataFrame([
        {
            "prj_AddedBy": "tester",
            "prj_ProjectTitle": "Title",
            "prj_ProjectBackground": "Bg",
            "prj_ProjectDescription": "Desc",
        }
    ])
    person_df = pd.DataFrame([
        {"ppl_Name_Last": "Doe", "ppl_Name_First": "John", "ppl_AddedBy": "tester"}
    ])
    project_df.to_csv(project_file, sep="\t", index=False)
    person_df.to_csv(person_file, sep="\t", index=False)

    io_file.import_file(str(project_file), "project", db_file_path=str(db_path))
    io_file.import_file(str(person_file), "person", db_file_path=str(db_path))

    with get_session(file_path=str(db_path)) as session:
        links = session.query(ProjectPerson).all()
        assert len(links) == 1
        assert links[0].project_id == 1
        assert links[0].person_id == 1

    project_file.unlink()
    person_file.unlink()
    db_path.unlink()
    assert not project_file.exists()
    assert not person_file.exists()
    assert not db_path.exists()


def test_import_file_unknown_table_raises_value_error(tmp_path):
    db_path = tmp_path / "test.db"
    csv_path = tmp_path / "data.csv"
    df = pd.DataFrame([
        {"ppl_Name_Last": "Doe", "ppl_Name_First": "John", "ppl_AddedBy": "tester"}
    ])
    df.to_csv(csv_path, index=False)

    with pytest.raises(ValueError):
        io_file.import_file(str(csv_path), "unknown", db_file_path=str(db_path))

    csv_path.unlink()
    if db_path.exists():
        db_path.unlink()
    assert not csv_path.exists()
    assert not db_path.exists()
