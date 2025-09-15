from pydantic import BaseModel
from ispec.db.models import Person, Project, ProjectComment
from ispec.api.models.modelmaker import make_pydantic_model_from_sqlalchemy


def test_person_model_fields():
    """Basic field generation from a SQLAlchemy model."""

    PersonRead = make_pydantic_model_from_sqlalchemy(
        Person,
        name_suffix="Read",
        strip_prefix="ppl_",
        exclude_fields={"id"},
    )

    assert issubclass(PersonRead, BaseModel)
    model_fields = PersonRead.model_fields

    # Ensure key fields are present and properly renamed
    assert "Name_First" in model_fields
    assert "Name_Last" in model_fields
    assert "AddedBy" in model_fields

    # Ensure stripped prefix
    assert "ppl_Name_First" not in model_fields

    # Fields should be required by default
    assert model_fields["Name_First"].is_required() is True
    assert model_fields["Name_Last"].is_required() is True


def test_person_model_optional_fields_with_optional_all():
    """When optional_all is True, all fields become optional."""

    PersonCreate = make_pydantic_model_from_sqlalchemy(
        Person,
        name_suffix="Create",
        strip_prefix="ppl_",
        exclude_fields={"id"},
        optional_all=True,
    )

    model_fields = PersonCreate.model_fields
    assert model_fields["Name_First"].is_required() is False
    assert model_fields["Name_Last"].is_required() is False


def test_include_relationships_with_related_model_map():
    """Include relationship fields using related_model_map."""

    PersonRead = make_pydantic_model_from_sqlalchemy(
        Person,
        name_suffix="Read",
        strip_prefix="ppl_",
        exclude_fields={"id"},
    )
    ProjectRead = make_pydantic_model_from_sqlalchemy(
        Project,
        name_suffix="Read",
        strip_prefix="prj_",
        exclude_fields={"id"},
    )

    CommentRead = make_pydantic_model_from_sqlalchemy(
        ProjectComment,
        name_suffix="Read",
        include_relationships=True,
        related_model_map={
            "person": PersonRead,
            "project": ProjectRead,
        },
    )

    model_fields = CommentRead.model_fields
    assert "person" in model_fields
    assert model_fields["person"].annotation == PersonRead
    assert "project" in model_fields
    assert model_fields["project"].annotation == ProjectRead

