import pytest
from sqlalchemy import inspect
from sqlalchemy.orm import Session
from ispec.db.models import (
    Assay,
    Base,
    Experiment,
    Person,
    Project,
    ProjectComment,
    Reagent,
    sqlite_engine,
)


@pytest.fixture(scope="function")
def db_session(tmp_path):
    engine = sqlite_engine(f"sqlite:///{tmp_path}/test.sqlite")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        yield session


def test_create_all_tables(db_session):
    inspector = inspect(db_session.bind)
    tables = inspector.get_table_names()
    assert "person" in tables
    assert "project" in tables
    assert "project_comment" in tables
    assert "reagent" in tables
    assert "assay" in tables
    experiment_columns = {col["name"] for col in inspector.get_columns("experiment")}
    assert "assay_id" in experiment_columns


def test_assay_reagent_and_experiment_relationships(db_session):
    project = Project(prj_AddedBy="admin", prj_ProjectTitle="Cancer Study")
    reagent = Reagent(name="Trypsin", vendor="Promega", room_number="1420")
    assay = Assay(name="Digest LC-MS", primary_reagent=reagent)
    db_session.add_all([project, reagent, assay])
    db_session.flush()

    experiment = Experiment(project_id=project.id, assay=assay, record_no="EXP-001")
    db_session.add(experiment)
    db_session.commit()

    retrieved = db_session.query(Experiment).filter_by(record_no="EXP-001").one()
    assert retrieved.assay.name == "Digest LC-MS"
    assert retrieved.assay.primary_reagent.name == "Trypsin"


def test_insert_person(db_session):
    person = Person(
        ppl_AddedBy="admin",
        ppl_Name_First="Alice",
        ppl_Name_Last="Smith",
        ppl_Domain="example.org",
        ppl_Email="alice@example.org",
        ppl_Phone="123-456-7890",
        ppl_PI="Dr. Jones",
        ppl_Institution="ABC Institute",
        ppl_Center="Proteomics",
        ppl_Department="Biology",
        ppl_Status="active",
        ppl_Roles_PI="yes",
        ppl_Roles_CoreUser="no",
        ppl_Roles_CoreStaff="yes",
        ppl_Roles_Collaborator="yes",
        ppl_Roles="core_staff",
    )
    db_session.add(person)
    db_session.commit()

    found = db_session.query(Person).filter_by(ppl_Email="alice@example.org").first()
    assert found is not None
    assert found.ppl_Name_First == "Alice"
    assert found.ppl_CreationTS is not None


def test_project_comment_relationship(db_session):
    person = Person(ppl_AddedBy="admin", ppl_Name_First="Bob", ppl_Name_Last="Smith")
    project = Project(prj_AddedBy="admin", prj_ProjectTitle="Cancer Study")
    db_session.add_all([person, project])
    db_session.flush()

    comment = ProjectComment(
        project_id=project.id, person_id=person.id, com_Comment="Initial notes"
    )
    db_session.add(comment)
    db_session.commit()

    retrieved = db_session.query(ProjectComment).first()
    assert retrieved.project.prj_ProjectTitle == "Cancer Study"
    assert retrieved.person.ppl_Name_First == "Bob"
