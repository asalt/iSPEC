# schemas.py
from typing import Any, Type
from functools import cache
from pydantic import BaseModel

from .utils.ui_meta import ui_from_column

#


# class PersonCreate(BaseModel):
#     ppl_Name_First: str
#     ppl_Name_Last: str
# 
# class PersonOut(PersonCreate):
#     id: int
# 
#     class Config:
#         orm_mode = True  # <-- enables use of SQLAlchemy ORM object directly


@cache
def build_form_schema(model: Type[Any], CreateModel: Type[BaseModel]) -> dict:
    schema = CreateModel.model_json_schema()
    props = schema.get("properties", {})
    columns = {c.name: c for c in model.__table__.columns}  # type: ignore

    # attach per-field UI
    for name, prop in props.items():
        if name in columns:
            prop["ui"] = ui_from_column(columns[name])

    # top-level UI affordances
    order = [c.name for c in model.__table__.columns if c.name in props]
    schema["ui"] = {
        "title": model.__name__,
        "order": order,
        "sections": getattr(model, "__ui__", {}).get("sections", []),  # optional
        # custom info from if the basemodel has a __ui__ attribute with a sections key
    }
    return schema

