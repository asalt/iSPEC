import os
import sys
from pathlib import Path

import pandas as pd

# Ensure the src directory is on the Python path for direct imports
SRC_PATH = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(SRC_PATH))

from ispec.io import io_file
from ispec.db.connect import get_session
from ispec.db.models import Person
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


def test_operations_initialize_creates_db_file(tmp_path):
    db_path = tmp_path / 'cli.db'
    operations.initialize(file_path=str(db_path))
    assert db_path.exists()
