# utils/ui_meta.py
from __future__ import annotations
from typing import Any, Callable
from sqlalchemy.orm import DeclarativeMeta
from sqlalchemy.sql.schema import Column as SAColumn
# from sqlalchemy import String, Integer, Boolean, DateTime, Enum, JSON
from sqlalchemy.sql import sqltypes as T  # canonical type classes


def ui_from_column(
    col: SAColumn,
    *,
    route_prefix_for_table: Callable[[str], str] | None = None,
) -> dict[str, Any]:
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
    # allow explicit overrides: make_column(..., info={"ui": {...}})
    if (col.info or {}).get("ui"):
        return dict(col.info["ui"])  # copy

    t = col.type
    ui: dict[str, Any] = {}

    # TEXT → textarea by default
    if isinstance(t, T.Text):
        ui["component"] = "Textarea"
    # strings with/without length
    elif isinstance(t, (T.String, T.Unicode)):
        if getattr(t, "length", None) and t.length <= 255:
            ui["component"] = "Text"
        else:
            ui["component"] = "Textarea"
    elif isinstance(t, (T.Integer, T.BigInteger, T.SmallInteger)):
        ui["component"] = "Number"
    elif isinstance(t, (T.Numeric, T.Float, T.REAL, T.DECIMAL)):
        ui["component"] = "Number"; ui["step"] = "any"
    elif isinstance(t, T.Boolean):
        ui["component"] = "Checkbox"
    elif isinstance(t, T.DateTime):
        ui["component"] = "DateTime"
    elif isinstance(t, T.Date):
        ui["component"] = "Date"
    elif isinstance(t, T.Enum):
        ui["component"] = "Select"
        if getattr(t, "enums", None):
            ui["options"] = list(t.enums)
        elif getattr(t, "enum_class", None):
            ui["options"] = [e.value for e in t.enum_class]  # python Enum
    elif isinstance(t, T.JSON):
        ui["component"] = "Json"
    else:
        ui["component"] = "Text"

    # Foreign keys → async select to the *target resource's* /options
    if col.foreign_keys:
        fk = next(iter(col.foreign_keys))
        target_table = fk.column.table.name
        endpoint = f"/{target_table}/options"
        if route_prefix_for_table:
            # prefer the actual router prefix you registered (handles plurals like /people)
            endpoint = f"{route_prefix_for_table(target_table)}/options"
        ui.update({
            "component": "SelectAsync",
            "optionsEndpoint": endpoint,
            "valueKey": "value",
            "labelKey": "label",
        })

    if col.nullable:
        ui["optional"] = True

    # mark obvious autofill-ish fields
    if col.default is not None or col.server_default is not None:
        ui["autofill"] = True

    return ui
