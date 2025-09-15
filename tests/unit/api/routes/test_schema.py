import pytest

from ispec.db.models import Person, ProjectComment
from ispec.api.models.modelmaker import make_pydantic_model_from_sqlalchemy
from ispec.api.routes.schema import build_form_schema
from ispec.api.routes.utils.ui_meta import ui_from_column


def _get_create_model():
    return make_pydantic_model_from_sqlalchemy(Person, name_suffix="Create")


def test_ui_order_matches_model_column_order():
    PersonCreate = _get_create_model()
    schema = build_form_schema(Person, PersonCreate)

    expected_order = [
        c.name for c in Person.__table__.columns if c.name in PersonCreate.model_fields
    ]
    assert schema["ui"]["order"] == expected_order


def test_ui_metadata_injected_into_fields():
    PersonCreate = _get_create_model()
    schema = build_form_schema(Person, PersonCreate)

    for name, field in PersonCreate.model_fields.items():
        assert "ui" in field.json_schema_extra, f"missing ui for {name}"
        expected_ui = ui_from_column(Person.__table__.columns[name])
        assert field.json_schema_extra["ui"] == expected_ui
        assert schema["properties"][name]["ui"] == expected_ui


def test_foreign_key_field_uses_route_prefix_mapping():
    """Foreign key fields should use SelectAsync and mapped route prefix."""
    CommentCreate = make_pydantic_model_from_sqlalchemy(
        ProjectComment, name_suffix="Create"
    )

    mapping = {"project": "/projects"}

    def prefix_for_table(name: str) -> str:
        return mapping.get(name, f"/{name}")

    schema = build_form_schema(
        ProjectComment, CommentCreate, route_prefix_for_table=prefix_for_table
    )

    project_ui = schema["properties"]["project_id"]["ui"]
    assert project_ui["component"] == "SelectAsync"
    assert project_ui["optionsEndpoint"] == "/projects/options"
