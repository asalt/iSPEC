from typing import Any, get_args, get_origin

from sqlalchemy import JSON, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from ispec.api.models.modelmaker import make_pydantic_model_from_sqlalchemy


class _Base(DeclarativeBase):
    pass


class _Example(_Base):
    __tablename__ = "example"

    id: Mapped[int] = mapped_column(primary_key=True)
    meta: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    note: Mapped[str] = mapped_column(Text, nullable=True)
    required_name: Mapped[str | None] = mapped_column(Text, nullable=False)


def test_modelmaker_prefers_mapped_type_hints():
    Create = make_pydantic_model_from_sqlalchemy(_Example, name_suffix="Create")

    meta_ann = Create.model_fields["meta"].annotation
    assert get_origin(meta_ann) is dict
    assert get_args(meta_ann) == (str, Any)
    assert Create.model_fields["meta"].is_required()


def test_modelmaker_aligns_optional_with_nullability():
    Create = make_pydantic_model_from_sqlalchemy(_Example, name_suffix="Create")

    note_ann = Create.model_fields["note"].annotation
    assert type(None) in get_args(note_ann)
    assert not Create.model_fields["note"].is_required()

    required_ann = Create.model_fields["required_name"].annotation
    assert type(None) not in get_args(required_ann)
    assert Create.model_fields["required_name"].is_required()
