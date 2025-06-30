import pytest
from pydantic import BaseModel
from ispec.db.models import Person
from ispec.api.models.modelmaker import make_pydantic_model_from_sqlalchemy


def test_person_model_fields():
    # Create a dynamic Pydantic model from SQLAlchemy
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

    # Check types or defaults
    assert (
        model_fields["Name_First"].is_required() is True
    )  # if nullable or optional_all

    assert (
        model_fields["Name_Last"].is_required() is True
    )  # if nullable or optional_all


def test_person_model_fields():
    # Create a dynamic Pydantic model from SQLAlchemy
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

    # Check types or defaults
    assert (
        model_fields["Name_First"].is_required() is True
    )  # if nullable or optional_all

    assert (
        model_fields["Name_Last"].is_required() is True
    )  # if nullable or optional_all
