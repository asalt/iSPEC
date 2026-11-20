"""Registry validation shared by the API, offline editor and pull CLI."""

import os
import re
from pathlib import Path, PureWindowsPath
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from sqlalchemy.orm import Session

from ispec.db.connect import get_db_dir
from ispec.db.models import InstrumentRegistry

Identifier = Annotated[str, Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}$")]
Alias = Literal["acquisition", "log", "tune"]
SourcePath = Annotated[str, Field(min_length=1, max_length=4096)]


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


def check_path(value, *, local):
    if re.search(r"[\x00-\x1f@]", value) or ".." in re.split(r"[/\\]", value):
        raise ValueError("Invalid source path")
    if local:
        base = Path(os.getenv("ISPEC_INSTRUMENT_MOUNT_ROOT", Path.home() / "windows"))
        if (
            not Path(value).is_absolute()
            or not Path(value).is_relative_to(base)
            or Path(value) == base
        ):
            raise ValueError(
                "Select a path below the operator-approved mount directory"
            )
    elif not (Path(value).is_absolute() or re.match(r"^[A-Za-z]:[\\/]", value)):
        raise ValueError("Expected an absolute source path")
    elif PureWindowsPath(value).drive.startswith("\\\\") or value.startswith("//"):
        raise ValueError("Use a local drive for a push source")


class Mount(Strict):
    host: Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9.-]{0,252}$")]
    share: Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.$ -]{0,79}$")]
    mount_path: SourcePath


class Settings(Strict):
    display_name: Annotated[str, Field(min_length=1, max_length=128)]
    plugin: Literal["thermo", "bruker"]
    mode: Literal["pull", "push"]
    enabled: bool = True
    expected_heartbeat_seconds: Annotated[int, Field(ge=30, le=86400)] = 3600
    collector_id: Identifier | None = None
    roots: dict[Alias, SourcePath | None]
    connections: dict[Alias, Mount] = Field(default_factory=dict)

    @model_validator(mode="after")
    def valid_sources(self):
        if not self.roots.get("acquisition"):
            raise ValueError("Acquisition root required")
        if self.mode == "push" and (not self.collector_id or self.connections):
            raise ValueError(
                "Push needs a collector ID and local roots, not backend mounts"
            )
        for alias, path in self.roots.items():
            if path:
                check_path(path, local=self.mode == "pull")
        for alias, mount in self.connections.items():
            check_path(mount.mount_path, local=True)
            if not self.roots.get(alias) or not Path(self.roots[alias]).is_relative_to(
                mount.mount_path
            ):
                raise ValueError("Source root must be under its corresponding mount")
        return self


class Entry(Settings):
    id: Identifier


def add_entry(db: Session, entry: Entry):
    row = InstrumentRegistry(id=entry.id, settings=entry.model_dump(exclude={"id"}))
    db.add(row)
    db.flush()
    return row


def change_entry(db, row, changes):
    updated = Settings.model_validate({**row.settings, **changes}).model_dump()
    if any(
        updated[key] != row.settings.get(key)
        for key in ("plugin", "mode", "collector_id", "roots", "connections")
    ):
        row.generation += 1  # Keep the old stream at its old path.
        row.archive_seen = False
    row.settings = updated
    db.flush()
    return row


def record(row):
    return {"id": row.id, **row.settings, "revision": row.revision}


def archive_path(row):
    base = (
        Path(os.environ["ISPEC_INSTRUMENT_ARCHIVES"])
        if "ISPEC_INSTRUMENT_ARCHIVES" in os.environ
        else get_db_dir() / "instruments"
    )
    return base / f"{row.id}.{row.generation}.sqlite"
