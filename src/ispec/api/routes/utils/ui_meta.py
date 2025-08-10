# utils/ui_meta.py
from __future__ import annotations
from sqlalchemy import String, Integer, Boolean, DateTime, Enum, JSON
from sqlalchemy.orm import DeclarativeMeta, Column
from typing import Any

def ui_from_column(col: Column) -> dict[str, Any]:
    """

    TL;DR
    generates ui metainfo based on model schema info
    add a {self}/options endpoint

    automatically sets nullable fields as optional

    other things

    longer:

    Generate UI metadata for a SQLAlchemy model column.

    This function inspects the SQLAlchemy `Column` object and returns
    a dictionary describing how the column should be represented in a UI
    form or component. The output includes the suggested component type,
    optional validation constraints (e.g., max length), and hints for
    async select options if the column is a foreign key.

    Resolution order:
      1. **Manual override** – If `col.info` contains a `"ui"` entry, that
         dictionary is returned directly without further inspection.
      2. **Type-based defaults** – Maps SQLAlchemy column types to common
         UI components.
      3. **Foreign key handling** – Foreign key columns are represented
         as `"SelectAsync"`, with an `optionsEndpoint` pointing to the
         related table’s `/options` endpoint, and default `valueKey` /
         `labelKey` mappings.
      4. **Additional flags** –
         - `autofill=True` if a default or server_default is set
         - `optional=True` if the column is nullable

    Type → Component Mapping
    ------------------------
    | SQLAlchemy Type | Component   | Extra Fields                |
    |-----------------|-------------|-----------------------------|
    | String          | Text / Textarea | `maxLength` if defined, `Textarea` if length > 255 |
    | Integer         | Number      |                             |
    | Boolean         | Checkbox    |                             |
    | DateTime        | DateTime    |                             |
    | Enum            | Select      | `options` from `t.enums`    |
    | JSON            | Json        |                             |
    | Other           | Text        |                             |

    Parameters
    ----------
    col : sqlalchemy.orm.Column
        A SQLAlchemy column object whose type, constraints, and metadata
        will be inspected to produce the UI metadata.

    Returns
    -------
    dict[str, Any]
        A dictionary of UI metadata fields, suitable for driving dynamic
        form generation or API schema descriptions.

    Notes
    -----
    - Uses the assignment expression (walrus operator) to fetch and
      return the `col.info["ui"]` override in one step.
    - Intended to be paired with model schema definitions where
      `col.info` may hold additional UI-specific configuration.


    """
    # manual override first
    if (ui := (col.info or {}).get("ui")):
        return ui

    t = col.type
    ui: dict[str, Any] = {}
    if isinstance(t, String):
        ui["component"] = "Textarea" if (getattr(t, "length", 0) or 0) > 255 else "Text"
        if getattr(t, "length", None):
            ui["maxLength"] = t.length
    elif isinstance(t, Integer):
        ui["component"] = "Number"
    elif isinstance(t, Boolean):
        ui["component"] = "Checkbox"
    elif isinstance(t, DateTime):
        ui["component"] = "DateTime"
    elif isinstance(t, Enum):
        ui["component"] = "Select"
        ui["options"]  = list(t.enums)  # or [e.value for e in t.enums]
    elif isinstance(t, JSON):
        ui["component"] = "Json"
    else:
        ui["component"] = "Text"

    # foreign keys → async select
    if col.foreign_keys:
        fk = next(iter(col.foreign_keys))
        target = fk.column.table.name # e.g. "people", "project"
        ui["component"] = "SelectAsync"
        ui["optionsEndpoint"] = f"/{target}/options"
        ui["valueKey"] = "id"; ui["labelKey"] = "name"  # tweak per model
    if col.default is not None or col.server_default is not None:
        ui["autofill"] = True
    if col.nullable:
        ui["optional"] = True
    return ui

