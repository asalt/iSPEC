from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any


_DEFAULT_DB_DIR = Path.home() / "ispec"
_DEFAULT_STATE_DIR = Path.home() / ".ispec"
_DEFAULT_LOG_DIR = Path.home() / ".ispec" / "logs"
_DEFAULT_CONFIG_DIR = Path.home() / ".ispec"

_DB_ENV_KEYS: dict[str, str] = {
    "core": "ISPEC_DB_PATH",
    "analysis": "ISPEC_ANALYSIS_DB_PATH",
    "psm": "ISPEC_PSM_DB_PATH",
    "assistant": "ISPEC_ASSISTANT_DB_PATH",
    "agent": "ISPEC_AGENT_DB_PATH",
    "agent_state": "ISPEC_AGENT_STATE_DB_PATH",
    "schedule": "ISPEC_SCHEDULE_DB_PATH",
}

_DB_DEFAULT_FILENAMES: dict[str, str] = {
    "core": "ispec.db",
    "analysis": "ispec-analysis.db",
    "psm": "ispec-psm.db",
    "assistant": "ispec-assistant.db",
    "agent": "ispec-agent.db",
    "agent_state": "ispec-agent-state.db",
    "schedule": "ispec-schedule.db",
}

_DB_ALIAS_ENV_KEYS: dict[str, tuple[str, ...]] = {
    "analysis": ("ISPEC_OMICS_DB_PATH",),
}


@dataclass(frozen=True)
class ResolvedLocation:
    name: str
    kind: str
    value: str
    path: str | None
    uri: str | None
    source: str
    env_var: str | None = None
    defaulted: bool = False
    deprecated_env_var: str | None = None
    notes: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "value": self.value,
            "path": self.path,
            "uri": self.uri,
            "source": self.source,
            "env_var": self.env_var,
            "defaulted": self.defaulted,
            "deprecated_env_var": self.deprecated_env_var,
            "notes": list(self.notes),
        }


def _env_value(key: str) -> str | None:
    raw = (os.getenv(key) or "").strip()
    return raw or None


def _sqlite_path(raw: str | os.PathLike[str] | None) -> Path | None:
    if raw is None:
        return None
    value = str(raw).strip()
    if not value:
        return None
    if value.startswith("sqlite:///"):
        return Path(value.removeprefix("sqlite:///")).expanduser()
    if value.startswith("sqlite://") and not value.startswith("sqlite:////"):
        return Path(value.removeprefix("sqlite://")).expanduser()
    if "://" in value:
        return None
    return Path(value).expanduser()


def _sqlite_uri(raw: str | os.PathLike[str]) -> str:
    value = str(raw).strip()
    if value.startswith("sqlite"):
        return value
    return "sqlite:///" + str(Path(value).expanduser())


def _resolved_path(
    *,
    name: str,
    kind: str,
    raw: str | os.PathLike[str],
    source: str,
    env_var: str | None = None,
    defaulted: bool = False,
    deprecated_env_var: str | None = None,
    notes: tuple[str, ...] = (),
) -> ResolvedLocation:
    path = Path(raw).expanduser() if not str(raw).strip().startswith("sqlite") else _sqlite_path(raw)
    value = str(path) if path is not None else str(raw).strip()
    return ResolvedLocation(
        name=name,
        kind=kind,
        value=value,
        path=str(path) if path is not None else None,
        uri=None,
        source=source,
        env_var=env_var,
        defaulted=defaulted,
        deprecated_env_var=deprecated_env_var,
        notes=notes,
    )


def _resolved_db(
    *,
    name: str,
    raw: str | os.PathLike[str],
    source: str,
    env_var: str | None = None,
    defaulted: bool = False,
    deprecated_env_var: str | None = None,
    notes: tuple[str, ...] = (),
) -> ResolvedLocation:
    path = _sqlite_path(raw)
    uri = _sqlite_uri(raw)
    value = str(path) if path is not None else str(raw).strip()
    return ResolvedLocation(
        name=name,
        kind="database",
        value=value,
        path=str(path) if path is not None else None,
        uri=uri,
        source=source,
        env_var=env_var,
        defaulted=defaulted,
        deprecated_env_var=deprecated_env_var,
        notes=notes,
    )


def resolve_db_dir() -> ResolvedLocation:
    env_value = _env_value("ISPEC_DB_DIR")
    if env_value is not None:
        return _resolved_path(
            name="db_dir",
            kind="directory",
            raw=env_value,
            source="env",
            env_var="ISPEC_DB_DIR",
        )
    return _resolved_path(
        name="db_dir",
        kind="directory",
        raw=_DEFAULT_DB_DIR,
        source="default",
        defaulted=True,
    )


def resolve_state_dir() -> ResolvedLocation:
    env_value = _env_value("ISPEC_STATE_DIR")
    if env_value is not None:
        return _resolved_path(
            name="state_dir",
            kind="directory",
            raw=env_value,
            source="env",
            env_var="ISPEC_STATE_DIR",
        )
    return _resolved_path(
        name="state_dir",
        kind="directory",
        raw=_DEFAULT_STATE_DIR,
        source="default",
        defaulted=True,
    )


def resolve_log_dir() -> ResolvedLocation:
    env_value = _env_value("ISPEC_LOG_DIR")
    if env_value is not None:
        return _resolved_path(
            name="log_dir",
            kind="directory",
            raw=env_value,
            source="env",
            env_var="ISPEC_LOG_DIR",
        )
    return _resolved_path(
        name="log_dir",
        kind="directory",
        raw=_DEFAULT_LOG_DIR,
        source="default",
        defaulted=True,
    )


def resolve_config_dir() -> ResolvedLocation:
    env_value = _env_value("ISPEC_CONFIG_DIR")
    if env_value is not None:
        return _resolved_path(
            name="config_dir",
            kind="directory",
            raw=env_value,
            source="env",
            env_var="ISPEC_CONFIG_DIR",
        )
    return _resolved_path(
        name="config_dir",
        kind="directory",
        raw=_DEFAULT_CONFIG_DIR,
        source="default",
        defaulted=True,
    )


def resolve_log_config_path() -> ResolvedLocation:
    env_value = _env_value("ISPEC_LOG_CONFIG")
    if env_value is not None:
        return _resolved_path(
            name="log_config",
            kind="file",
            raw=env_value,
            source="env",
            env_var="ISPEC_LOG_CONFIG",
        )
    config_dir = resolve_config_dir()
    base = Path(config_dir.path or _DEFAULT_CONFIG_DIR)
    return _resolved_path(
        name="log_config",
        kind="file",
        raw=base / "logging.json",
        source="default_from_config_dir",
        defaulted=True,
    )


def _core_parent_dir() -> Path:
    core = resolve_db_location("core")
    if core.path:
        return Path(core.path).parent
    db_dir = resolve_db_dir()
    return Path(db_dir.path or _DEFAULT_DB_DIR)


def resolve_db_location(
    logical_name: str,
    file: str | os.PathLike[str] | None = None,
) -> ResolvedLocation:
    name = logical_name.strip().lower() if logical_name else "core"
    if name == "primary":
        name = "analysis"

    if file is not None and str(file).strip():
        return _resolved_db(name=name, raw=file, source="explicit_arg")

    env_key = _DB_ENV_KEYS.get(name)
    if env_key is not None:
        env_value = _env_value(env_key)
        if env_value is not None:
            return _resolved_db(name=name, raw=env_value, source="env", env_var=env_key)

    for alias in _DB_ALIAS_ENV_KEYS.get(name, ()):
        alias_value = _env_value(alias)
        if alias_value is None:
            continue
        return _resolved_db(
            name=name,
            raw=alias_value,
            source="compat_env",
            env_var=alias,
            deprecated_env_var=alias,
            notes=(f"{alias} is deprecated; prefer {env_key}.",) if env_key else (),
        )

    filename = _DB_DEFAULT_FILENAMES.get(name, f"ispec-{name}.db")
    if name == "core":
        db_dir = resolve_db_dir()
        base = Path(db_dir.path or _DEFAULT_DB_DIR)
        return _resolved_db(
            name=name,
            raw=base / filename,
            source="default_from_db_dir",
            defaulted=True,
        )

    return _resolved_db(
        name=name,
        raw=_core_parent_dir() / filename,
        source="default_sibling",
        defaulted=True,
    )


def resolve_api_state_file() -> ResolvedLocation:
    env_value = _env_value("ISPEC_API_STATE_FILE")
    if env_value is not None:
        return _resolved_path(
            name="api_state_file",
            kind="file",
            raw=env_value,
            source="env",
            env_var="ISPEC_API_STATE_FILE",
        )
    state_dir = resolve_state_dir()
    return _resolved_path(
        name="api_state_file",
        kind="file",
        raw=Path(state_dir.path or _DEFAULT_STATE_DIR) / "api_server.json",
        source="default_from_state_dir",
        defaulted=True,
    )


def resolve_api_pid_file() -> ResolvedLocation:
    env_value = _env_value("ISPEC_API_PID_FILE")
    if env_value is not None:
        return _resolved_path(
            name="api_pid_file",
            kind="file",
            raw=env_value,
            source="env",
            env_var="ISPEC_API_PID_FILE",
        )

    state_override = _env_value("ISPEC_API_STATE_FILE")
    if state_override is not None:
        return _resolved_path(
            name="api_pid_file",
            kind="file",
            raw=Path(state_override).expanduser().parent / "api_server.pid",
            source="sibling_of_api_state_file",
            defaulted=True,
        )

    state_dir = resolve_state_dir()
    return _resolved_path(
        name="api_pid_file",
        kind="file",
        raw=Path(state_dir.path or _DEFAULT_STATE_DIR) / "api_server.pid",
        source="default_from_state_dir",
        defaulted=True,
    )


def resolve_supervisor_state_file() -> ResolvedLocation:
    env_value = _env_value("ISPEC_SUPERVISOR_STATE_FILE")
    if env_value is not None:
        return _resolved_path(
            name="supervisor_state_file",
            kind="file",
            raw=env_value,
            source="env",
            env_var="ISPEC_SUPERVISOR_STATE_FILE",
        )
    state_dir = resolve_state_dir()
    return _resolved_path(
        name="supervisor_state_file",
        kind="file",
        raw=Path(state_dir.path or _DEFAULT_STATE_DIR) / "supervisor.json",
        source="default_from_state_dir",
        defaulted=True,
    )


def resolve_supervisor_pid_file() -> ResolvedLocation:
    env_value = _env_value("ISPEC_SUPERVISOR_PID_FILE")
    if env_value is not None:
        return _resolved_path(
            name="supervisor_pid_file",
            kind="file",
            raw=env_value,
            source="env",
            env_var="ISPEC_SUPERVISOR_PID_FILE",
        )

    state_override = _env_value("ISPEC_SUPERVISOR_STATE_FILE")
    if state_override is not None:
        return _resolved_path(
            name="supervisor_pid_file",
            kind="file",
            raw=Path(state_override).expanduser().parent / "supervisor.pid",
            source="sibling_of_supervisor_state_file",
            defaulted=True,
        )

    state_dir = resolve_state_dir()
    return _resolved_path(
        name="supervisor_pid_file",
        kind="file",
        raw=Path(state_dir.path or _DEFAULT_STATE_DIR) / "supervisor.pid",
        source="default_from_state_dir",
        defaulted=True,
    )


def resolved_path_catalog() -> dict[str, dict[str, ResolvedLocation]]:
    return {
        "database": {
            "db_dir": resolve_db_dir(),
            "core": resolve_db_location("core"),
            "analysis": resolve_db_location("analysis"),
            "psm": resolve_db_location("psm"),
            "assistant": resolve_db_location("assistant"),
            "agent": resolve_db_location("agent"),
            "agent_state": resolve_db_location("agent_state"),
            "schedule": resolve_db_location("schedule"),
        },
        "state": {
            "state_dir": resolve_state_dir(),
            "api_state_file": resolve_api_state_file(),
            "api_pid_file": resolve_api_pid_file(),
            "supervisor_state_file": resolve_supervisor_state_file(),
            "supervisor_pid_file": resolve_supervisor_pid_file(),
        },
        "logging": {
            "log_dir": resolve_log_dir(),
            "config_dir": resolve_config_dir(),
            "log_config": resolve_log_config_path(),
        },
    }


__all__ = [
    "ResolvedLocation",
    "resolve_api_pid_file",
    "resolve_api_state_file",
    "resolve_config_dir",
    "resolve_db_dir",
    "resolve_db_location",
    "resolve_log_config_path",
    "resolve_log_dir",
    "resolve_state_dir",
    "resolve_supervisor_pid_file",
    "resolve_supervisor_state_file",
    "resolved_path_catalog",
]
