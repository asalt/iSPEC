from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class FileProbe:
    path: str
    exists: bool
    size_bytes: int | None
    mtime: float | None
    nonempty: bool


def _probe_file(path: str | Path | None) -> FileProbe | None:
    if path is None:
        return None
    text = str(path or "").strip()
    if not text:
        return None
    file_path = Path(text).expanduser()
    try:
        stat = file_path.stat()
    except OSError:
        return FileProbe(path=str(file_path), exists=False, size_bytes=None, mtime=None, nonempty=False)
    return FileProbe(
        path=str(file_path),
        exists=True,
        size_bytes=int(stat.st_size),
        mtime=float(stat.st_mtime),
        nonempty=stat.st_size > 0,
    )


def pid_is_alive(pid: int | None) -> bool | None:
    if pid is None or int(pid) <= 0:
        return None
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def inspect_long_task(
    *,
    pid: int | None = None,
    log_file: str | Path | None = None,
    output_paths: list[str | Path] | tuple[str | Path, ...] = (),
    sentinel_file: str | Path | None = None,
) -> dict[str, Any]:
    """Inspect durable state for a long-running local task.

    This intentionally does not infer semantics from logs. It only reports
    concrete state that a Codex/iSPEC caller can summarize or act on.
    """

    pid_alive = pid_is_alive(pid)
    log_probe = _probe_file(log_file)
    sentinel_probe = _probe_file(sentinel_file)
    output_probes = [probe for probe in (_probe_file(path) for path in output_paths) if probe is not None]

    outputs_ready = all(probe.exists and probe.nonempty for probe in output_probes) if output_probes else None
    complete = bool(sentinel_probe and sentinel_probe.exists)
    if pid_alive is False:
        complete = True
    if pid_alive is True:
        complete = False

    should_wake = bool(complete and (outputs_ready is not False))
    return {
        "pid": int(pid) if pid is not None else None,
        "pid_alive": pid_alive,
        "complete": complete,
        "outputs_ready": outputs_ready,
        "should_wake": should_wake,
        "log": log_probe.__dict__ if log_probe else None,
        "sentinel": sentinel_probe.__dict__ if sentinel_probe else None,
        "outputs": [probe.__dict__ for probe in output_probes],
    }
