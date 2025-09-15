import pytest
from pydantic import BaseModel

from ispec.db.models import Person, Project, ProjectComment
from ispec.api.models.modelmaker import (
    make_pydantic_model_from_sqlalchemy,
    get_models,
)


def test_person_model_fields():
    """Basic field generation with prefix stripping."""
    PersonRead = make_pydantic_model_from_sqlalchemy(
        Person,
        name_suffix="Read",
        strip_prefix="ppl_",
        exclude_fields={"id"},
    )

    assert issubclass(PersonRead, BaseModel)
    model_fields = PersonRead.model_fields

    assert "Name_First" in model_fields
    assert "Name_Last" in model_fields
    assert "AddedBy" in model_fields

    assert "ppl_Name_First" not in model_fields

    assert model_fields["Name_First"].is_required() is True
    assert model_fields["Name_Last"].is_required() is True


def test_optional_all_allows_none():
    """optional_all=True should make fields optional with default None."""
    PersonOptional = make_pydantic_model_from_sqlalchemy(
        Person,
        name_suffix="Optional",
        strip_prefix="ppl_",
        exclude_fields={"id"},
        optional_all=True,
    )

    model_fields = PersonOptional.model_fields
    assert model_fields["Name_First"].is_required() is False
    assert model_fields["Name_Last"].is_required() is False
    assert model_fields["Name_First"].default is None
    assert model_fields["Name_Last"].default is None


def test_include_relationships_with_related_model_map():
    """Nested models should be included when include_relationships=True."""
    PersonRead = make_pydantic_model_from_sqlalchemy(Person, name_suffix="Read")
    ProjectRead = make_pydantic_model_from_sqlalchemy(Project, name_suffix="Read")
    ProjectCommentRead = make_pydantic_model_from_sqlalchemy(
        ProjectComment,
        name_suffix="Read",
        include_relationships=True,
        related_model_map={"person": PersonRead, "project": ProjectRead},
    )

    model_fields = ProjectCommentRead.model_fields
    assert "person" in model_fields
    assert model_fields["person"].annotation is PersonRead
    assert "project" in model_fields
    assert model_fields["project"].annotation is ProjectRead
    assert model_fields["person"].default is None
    assert model_fields["project"].default is None


def test_invalid_strip_prefix_and_missing_column():
    """Invalid strip_prefix or missing exclude field should keep names untouched."""
    PersonRead = make_pydantic_model_from_sqlalchemy(
        Person,
        name_suffix="Read",
        strip_prefix="invalid_",
        exclude_fields={"id", "nonexistent"},
    )

    model_fields = PersonRead.model_fields
    assert "ppl_Name_First" in model_fields
    assert "Name_First" not in model_fields
    assert "nonexistent" not in model_fields
    assert "id" not in model_fields


def test_get_models_returns_expected_model_names():
    models = get_models()
    assert {"PersonRead", "ProjectRead", "ProjectUpdate", "ProjectCommentRead"} \
        <= set(models.keys())
