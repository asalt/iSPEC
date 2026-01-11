# ispec/api/models/modelmaker.py
from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, Optional, Type, get_args, get_origin, get_type_hints

from pydantic import BaseModel, ConfigDict, create_model
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapper,
    Mapped,
    ColumnProperty,
    RelationshipProperty,
    class_mapper,
)
from sqlalchemy.sql.schema import Column
from sqlalchemy.sql import sqltypes as T


_NONE_TYPE = type(None)


@lru_cache(maxsize=None)
def _model_type_hints(model_cls: Type[DeclarativeBase]) -> dict[str, Any]:
    try:
        return get_type_hints(model_cls, include_extras=True)  # type: ignore[call-arg]
    except TypeError:
        try:
            return get_type_hints(model_cls)
        except Exception:
            return {}
    except Exception:
        return {}


def _mapped_annotation_type(model_cls: Type[DeclarativeBase], field_name: str) -> Any | None:
    hints = _model_type_hints(model_cls)
    annotation = hints.get(field_name)
    if annotation is None or get_origin(annotation) is not Mapped:
        return None
    args = get_args(annotation)
    return args[0] if args else None


def _type_allows_none(tp: Any) -> bool:
    return _NONE_TYPE in get_args(tp)


def _ensure_optional(tp: Any) -> Any:
    if _type_allows_none(tp):
        return tp
    return Optional[tp]


def _strip_optional(tp: Any) -> Any:
    args = get_args(tp)
    if args and len(args) == 2 and _NONE_TYPE in args:
        return args[0] if args[1] is _NONE_TYPE else args[1]
    return tp



def make_pydantic_model_from_sqlalchemy(
    model_cls: Type[DeclarativeBase],
    *,
    name_suffix: str = "Create",
    exclude_fields: set[str] | None = None,
    optional_all: bool = False,
    include_relationships: bool = False,
    related_model_map: Optional[Dict[str, Type[BaseModel]]] = None,
    strip_prefix: str = "",
) -> Type[BaseModel]:
    mapper: Mapper = class_mapper(model_cls)
    fields: Dict[str, Any] = {}

    if exclude_fields is None:
        exclude_fields = set() if name_suffix.strip().lower() == "read" else {"id"}
    else:
        exclude_fields = set(exclude_fields)

    for prop in mapper.iterate_properties:
        # Column properties
        if isinstance(prop, ColumnProperty):
            col: Column = prop.columns[0]
            field_name = prop.key

            if field_name in exclude_fields:
                continue

            try:
                mapped_type = _mapped_annotation_type(model_cls, field_name)
                if mapped_type is not None:
                    python_type = mapped_type
                elif isinstance(col.type, T.Enum) and getattr(col.type, "enum_class", None):
                    python_type = col.type.enum_class
                else:
                    python_type = col.type.python_type
            except NotImplementedError:
                python_type = Any

            if strip_prefix and field_name.startswith(strip_prefix):
                field_name_out = field_name[len(strip_prefix) :]
            else:
                field_name_out = field_name

            is_optional = (
                optional_all
                or col.nullable
                or col.default is not None
                or col.server_default is not None
            )

            if is_optional:
                python_type = _ensure_optional(python_type)
                default = None
            else:
                python_type = _strip_optional(python_type)
                default = ...

            fields[field_name_out] = (python_type, default)

        # Relationship properties (only for read models)
        elif include_relationships and isinstance(prop, RelationshipProperty):
            rel_model_cls = prop.mapper.class_
            rel_name = prop.key

            if rel_name in related_model_map:
                fields[rel_name] = (related_model_map[rel_name], None)

    model_name = f"{model_cls.__name__}{name_suffix}"
    config = ConfigDict(from_attributes=True)
    return create_model(model_name, __config__=config, **fields)


#


def get_models():

    from ispec.db.models import Person, Project, ProjectComment

    PersonRead = make_pydantic_model_from_sqlalchemy(
        Person, name_suffix="Read", # strip_prefix="ppl_" #  not using the strip_prefix anymore
    )
    ProjectRead = make_pydantic_model_from_sqlalchemy(
        Project, name_suffix="Read", #strip_prefix="prj_"
    )
    ProjectUpdate = make_pydantic_model_from_sqlalchemy(
        Project, name_suffix="Update", #strip_prefix="prj_"
    )

    ProjectCommentRead = make_pydantic_model_from_sqlalchemy(
        ProjectComment,
        name_suffix="Read",
        #strip_prefix="com_",
        include_relationships=True,
        related_model_map={
            "person": PersonRead,
            "project": ProjectRead,
        },
    )

    return {
        "PersonRead": PersonRead,
        "ProjectRead": ProjectRead,
        "ProjectUpdate": ProjectUpdate,
        "ProjectCommentRead": ProjectCommentRead,
    }


if __name__ == "__main__":
    get_models()
