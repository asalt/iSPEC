# ispec/api/models/modelmaker.py
from pydantic import BaseModel, create_model
from typing import get_args, get_origin, Optional, Type, Dict, Any
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapper,
    ColumnProperty,
    RelationshipProperty,
    class_mapper,
)
from sqlalchemy.sql.schema import Column


def make_pydantic_model_from_sqlalchemy(
    model_cls: Type[DeclarativeBase],
    *,
    name_suffix: str = "Create",
    exclude_fields: set[str] = {"id"},
    optional_all: bool = False,
    include_relationships: bool = False,
    related_model_map: Optional[Dict[str, Type[BaseModel]]] = None,
    strip_prefix: str = "",
) -> Type[BaseModel]:
    mapper: Mapper = class_mapper(model_cls)
    fields: Dict[str, Any] = {}

    for prop in mapper.iterate_properties:
        # Column properties
        if isinstance(prop, ColumnProperty):
            col: Column = prop.columns[0]
            field_name = prop.key

            if field_name in exclude_fields:
                continue

            try:
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
                python_type = Optional[python_type]
                default = None
            else:
                default = ...

            fields[field_name_out] = (python_type, default)

        # Relationship properties (only for read models)
        elif include_relationships and isinstance(prop, RelationshipProperty):
            rel_model_cls = prop.mapper.class_
            rel_name = prop.key

            if rel_name in related_model_map:
                fields[rel_name] = (related_model_map[rel_name], None)

    model_name = f"{model_cls.__name__}{name_suffix}"
    return create_model(model_name, **fields)


#


def get_models():

    from ispec.db.models import Person, Project, ProjectComment

    PersonRead = make_pydantic_model_from_sqlalchemy(
        Person, name_suffix="Read", strip_prefix="ppl_"
    )
    ProjectRead = make_pydantic_model_from_sqlalchemy(
        Project, name_suffix="Read", strip_prefix="prj_"
    )
    ProjectUpdate = make_pydantic_model_from_sqlalchemy(
        Project, name_suffix="Update", strip_prefix="prj_"
    )

    ProjectCommentRead = make_pydantic_model_from_sqlalchemy(
        ProjectComment,
        name_suffix="Read",
        strip_prefix="com_",
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
