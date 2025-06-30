# test_crud.py

import sqlite3
import pytest


import pytest
from ispec.db.connect import get_session, sqlite_engine, initialize_db
from ispec.db.models import Person, Project, ProjectPerson
from ispec.db.crud import PersonCRUD, ProjectCRUD, ProjectPersonCRUD


@pytest.fixture(scope="function")
def db_session(tmp_path):
    db_url = f"sqlite:///{tmp_path}/test.db"
    engine = sqlite_engine(db_url)
    initialize_db(engine)
    with get_session(db_url) as session:
        yield session


def test_insert_and_get(db_session):
    person_crud = PersonCRUD()
    record = {
        "ppl_Name_Last": "Smith",
        "ppl_Name_First": "Jane",
        "ppl_AddedBy": "tester",
    }
    obj = person_crud.create(db_session, record)
    fetched = person_crud.get(db_session, obj.id)
    assert fetched.ppl_Name_Last == "Smith"
    assert fetched.ppl_Name_First == "Jane"
    assert fetched.id == obj.id


def test_insert_deduplicates_on_first_and_last_name(db_session):
    crud = PersonCRUD()
    rec1 = {"ppl_Name_Last": "Smith", "ppl_Name_First": "Jane", "ppl_AddedBy": "tester"}
    rec2 = {
        "ppl_Name_Last": "smith",
        "ppl_Name_First": "jane",
        "ppl_AddedBy": "tester",
    }  # different case

    obj1 = crud.create(db_session, rec1)
    obj2 = crud.create(db_session, rec2)

    assert obj1 is not None
    assert obj2 is None  # Deduplicated

    all_people = db_session.query(Person).all()
    assert len(all_people) == 1


def test_insert_missing_required_column(db_session):
    person_crud = PersonCRUD()
    record = {"ppl_Name_Last": "Smith"}  # missing first name
    with pytest.raises(Exception):
        person_crud.create(db_session, record)


def test_delete_by_id(db_session):
    person_crud = PersonCRUD()
    record = {
        "ppl_Name_Last": "Smith",
        "ppl_Name_First": "Jane",
        "ppl_AddedBy": "tester",
    }
    obj = person_crud.create(db_session, record)
    assert person_crud.delete(db_session, obj.id)
    assert person_crud.get(db_session, obj.id) is None


def test_delete_nonexistent_id(db_session):
    person_crud = PersonCRUD()
    assert not person_crud.delete(db_session, 99999)


def test_link_person_to_project(db_session):
    person_crud = PersonCRUD()
    project_crud = ProjectCRUD()
    link_crud = ProjectPersonCRUD()

    person = person_crud.create(
        session=db_session,
        record={
            "ppl_Name_Last": "Smith",
            "ppl_Name_First": "Jane",
            "ppl_AddedBy": "tester",
        },
    )
    project = project_crud.create(
        db_session,
        {
            "prj_ProjectTitle": "Moonbase Alpha",
            "prj_ProjectBackground": "This is a project background.",
        },
    )

    link = link_crud.create(
        db_session, {"person_id": person.id, "project_id": project.id}
    )
    assert link.person_id == person.id
    assert link.project_id == project.id


def test_link_person_to_project_invalid_person(db_session):
    project_crud = ProjectCRUD()
    link_crud = ProjectPersonCRUD()

    project = project_crud.create(
        db_session,
        {
            "prj_ProjectTitle": "Moonbase Alpha",
            "prj_ProjectBackground": "This is a project background.",
        },
    )

    with pytest.raises(ValueError):
        link_crud.create(db_session, {"person_id": 98, "project_id": project.id})


# old


# @pytest.fixture
# def conn():
#     c = sqlite3.connect(":memory:")
#     c.row_factory = sqlite3.Row

#     # Enable foreign key constraints
#     c.execute("PRAGMA foreign_keys = ON")

#     # Create person table
#     c.execute(
#         """
#         CREATE TABLE person (
#             id INTEGER PRIMARY KEY AUTOINCREMENT,
#             ppl_Name_Last TEXT NOT NULL,
#             ppl_Name_First TEXT NOT NULL
#         )
#     """
#     )

#     # Create project table
#     c.execute(
#         """
#         CREATE TABLE project (
#             id INTEGER PRIMARY KEY AUTOINCREMENT,
#             prj_ProjectTitle TEXT NOT NULL,
#             prj_ProjectBackground TEXT NOT NULL
#         )
#     """
#     )

#     # Create project_person join table
#     c.execute(
#         """
#         CREATE TABLE project_person (
#             id INTEGER PRIMARY KEY AUTOINCREMENT,
#             project_id INTEGER NOT NULL REFERENCES project(id),
#             person_id INTEGER NOT NULL REFERENCES person(id),
#             UNIQUE (person_id, project_id)
#         )
#     """
#     )

#     yield c
#     c.close()


# def test_insert_and_get(conn):
#     person = Person(conn)
#     record = {"ppl_Name_Last": "Smith", "ppl_Name_First": "Jane"}
#     new_id = person.insert(record)
#     row = person.get_by_id(new_id)
#     assert row["ppl_Name_Last"] == "Smith"
#     assert row["ppl_Name_First"] == "Jane"
#     assert row["id"] == new_id


# def test_insert_deduplicates_on_first_and_last_name(conn):
#     person = Person(conn)
#     rec1 = {"ppl_Name_Last": "Smith", "ppl_Name_First": "Jane"}
#     rec2 = {"ppl_Name_Last": "smith", "ppl_Name_First": "jane"}  # different case
#     id1 = person.insert(rec1)
#     id2 = person.insert(rec2)
#     assert id1 == id2  # Should deduplicate
#     all_rows = conn.execute("SELECT * FROM person").fetchall()
#     assert len(all_rows) == 1


# def test_insert_missing_required_column(conn):
#     person = Person(conn)
#     record = {"ppl_Name_Last": "Smith"}  # missing first name
#     with pytest.raises(ValueError):
#         person.insert(record)


# def test_delete_by_id(conn):
#     person = Person(conn)
#     record = {"ppl_Name_Last": "Smith", "ppl_Name_First": "Jane"}
#     new_id = person.insert(record)
#     assert person.delete_by_id(new_id)
#     assert person.get_by_id(new_id) is None


# def test_delete_nonexistent_id(conn):
#     person = Person(conn)
#     assert not person.delete_by_id(12345)


# def test_bulk_insert_projects(conn):
#     project = Project(conn)
#     records = [
#         {
#             "prj_ProjectTitle": "Project Alpha",
#             "prj_ProjectBackground": " This is a project background ",
#         },
#         {
#             "prj_ProjectTitle": "Project Beta",
#             "prj_ProjectBackground": " This is a project background ",
#         },
#         {
#             "prj_ProjectTitle": "Project Gamma",
#             "prj_ProjectBackground": " This is a project background ",
#         },
#     ]
#     project.bulk_insert(records)

#     rows = conn.execute("SELECT prj_ProjectTitle FROM project ORDER BY id").fetchall()
#     titles = [row["prj_ProjectTitle"] for row in rows]

#     assert titles == ["Project Alpha", "Project Beta", "Project Gamma"]


# def test_link_person_to_project(conn):
#     # this test is failing
#     person = Person(conn)
#     project = Project(conn)

#     person_id = person.insert({"ppl_Name_Last": "Lee", "ppl_Name_First": "Sara"})
#     project_id = project.insert(
#         {
#             "prj_ProjectTitle": "Moonbase Alpha",
#             "prj_ProjectBackground": "This is a project background.",
#         }
#     )

#     link = ProjectPerson(conn)
#     link_id = link.insert({"person_id": person_id, "project_id": project_id})

#     link = conn.execute(
#         "SELECT * FROM project_person WHERE id = ?", (link_id,)
#     ).fetchone()
#     assert link["person_id"] == person_id
#     assert link["project_id"] == project_id


# def test_link_person_to_project(conn):
#     # this test is failing
#     person = Person(conn)
#     project = Project(conn)

#     # person_id = person.insert({"ppl_Name_Last": "Lee", "ppl_Name_First": "Sara"})
#     project_id = project.insert(
#         {
#             "prj_ProjectTitle": "Moonbase Alpha",
#             "prj_ProjectBackground": "This is a project background.",
#         }
#     )

#     link = ProjectPerson(conn)
#     with pytest.raises(ValueError):
#         link_id = link.insert({"person_id": 98, "project_id": project_id})
