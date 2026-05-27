from __future__ import annotations

from sqlalchemy import Boolean, ForeignKey, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base, make_timestamp_mixin

ReagentTimestamp = make_timestamp_mixin("reagent")
AssayTimestamp = make_timestamp_mixin("assay")


class Reagent(ReagentTimestamp, Base):
    __tablename__ = "reagent"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        info={"ui": {"label": "Name"}},
    )
    reagent_type: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        info={"ui": {"label": "Type"}},
    )
    vendor: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        info={"ui": {"label": "Vendor"}},
    )
    catalog_no: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        info={"ui": {"label": "Catalog #"}},
    )
    lot_no: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        info={"ui": {"label": "Lot #"}},
    )
    room_number: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        info={"ui": {"label": "Room #"}},
    )
    storage_location: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        info={"ui": {"label": "Storage Location"}},
    )
    concentration: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        info={"ui": {"label": "Concentration"}},
    )
    notes: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        info={"ui": {"label": "Notes", "component": "Textarea"}},
    )
    active: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
        info={"ui": {"label": "Active"}},
    )

    primary_assays: Mapped[list["Assay"]] = relationship(
        "Assay",
        back_populates="primary_reagent",
    )


class Assay(AssayTimestamp, Base):
    __tablename__ = "assay"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        info={"ui": {"label": "Name"}},
    )
    assay_type: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        info={"ui": {"label": "Type"}},
    )
    primary_reagent_id: Mapped[int | None] = mapped_column(
        ForeignKey("reagent.id"),
        nullable=True,
        info={"ui": {"label": "Primary Reagent", "allowClear": True}},
    )
    description: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        info={"ui": {"label": "Description", "component": "Textarea"}},
    )
    notes: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        info={"ui": {"label": "Notes", "component": "Textarea"}},
    )
    active: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
        info={"ui": {"label": "Active"}},
    )

    primary_reagent: Mapped[Reagent | None] = relationship(
        "Reagent",
        back_populates="primary_assays",
    )
    experiments: Mapped[list["Experiment"]] = relationship(
        "Experiment",
        back_populates="assay",
    )
