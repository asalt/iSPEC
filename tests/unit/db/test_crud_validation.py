import pytest
from ispec.db.crud import CRUDBase, PersonCRUD, ProjectCRUD, ProjectCommentCRUD
from ispec.db.models import Person, Project


class PrefixedPersonCRUD(CRUDBase):
    prefix = "ppl_"

    def __init__(self):
        super().__init__(Person, req_cols=["ppl_Name_Last", "ppl_Name_First"])


def test_validate_input_requires_session():
    crud = CRUDBase(Person)
    record = {"ppl_Name_Last": "Smith"}
    with pytest.raises(ValueError):
        crud.validate_input(None, record)


def test_validate_input_removes_unknown_fields(db_session):
    crud = CRUDBase(Person)
    record = {
        "ppl_Name_Last": "Smith",
        "ppl_Name_First": "Jane",
        "ppl_AddedBy": "tester",
        "unknown": "value",
    }
    cleaned = crud.validate_input(db_session, record)
    assert "unknown" not in cleaned


def test_validate_input_enforces_required_columns(db_session):
    crud = CRUDBase(Person, req_cols=["ppl_Name_Last", "ppl_Name_First"])
    with pytest.raises(ValueError):
        crud.validate_input(db_session, {"ppl_Name_Last": "Smith"})


def test_validate_input_applies_prefix(db_session):
    crud = PrefixedPersonCRUD()
    record = {
        "Name_Last": "Smith",
        "Name_First": "Jane",
        "ppl_AddedBy": "tester",
    }
    cleaned = crud.validate_input(db_session, record)
    assert cleaned["ppl_Name_Last"] == "Smith"
    assert cleaned["ppl_Name_First"] == "Jane"
    assert "Name_Last" not in cleaned


def test_person_validate_input_blank_last_name(db_session):
    crud = PersonCRUD()
    record = {"ppl_Name_Last": "   ", "ppl_Name_First": "Jane", "ppl_AddedBy": "tester"}
    assert crud.validate_input(db_session, record) is None


def test_project_create_deduplicates_title(db_session):
    crud = ProjectCRUD()
    rec1 = {
        "prj_ProjectTitle": "Moonbase Alpha",
        "prj_ProjectBackground": "bg",
        "prj_AddedBy": "tester",
    }
    rec2 = {
        "prj_ProjectTitle": "moonbase alpha",
        "prj_ProjectBackground": "bg",
        "prj_AddedBy": "tester",
    }
    obj1 = crud.create(db_session, rec1)
    obj2 = crud.create(db_session, rec2)
    assert obj1 is not None
    assert obj2 is None
    assert db_session.query(Project).count() == 1


def test_projectcomment_validate_input_errors(db_session):
    person = PersonCRUD().create(
        db_session,
        {"ppl_Name_Last": "Smith", "ppl_Name_First": "Jane", "ppl_AddedBy": "tester"},
    )
    project = ProjectCRUD().create(
        db_session,
        {
            "prj_ProjectTitle": "Test",
            "prj_ProjectBackground": "bg",
            "prj_AddedBy": "tester",
        },
    )
    crud = ProjectCommentCRUD()
    with pytest.raises(ValueError):
        crud.validate_input(db_session, {"person_id": person.id})
    with pytest.raises(ValueError):
        crud.validate_input(db_session, {"person_id": person.id, "project_id": 9999})
    with pytest.raises(ValueError):
        crud.validate_input(db_session, {"project_id": project.id, "person_id": 9999})
    record = {"project_id": project.id, "person_id": person.id}
    assert crud.validate_input(db_session, record) == record
