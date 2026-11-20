"""Read-only Phase 1 archive adapter. Filesystem times are not run start/end."""

from contextlib import closing
import json
from pathlib import Path
import sqlite3
import subprocess
from urllib.parse import unquote

from .registry import archive_path


class ArchiveError(Exception):
    pass


class InvalidSnapshot(ValueError):
    pass


def mounted(root, connection=None):
    """Check the underlying filesystem and SMB identity, not just directory existence."""

    def run(*args):
        return subprocess.check_output(
            args, timeout=1, text=True, stderr=subprocess.DEVNULL
        ).strip()

    try:
        path = Path(run("realpath", "-e", "--", root))
        fs = json.loads(
            run("findmnt", "-J", "-T", str(path), "-o", "TARGET,FSTYPE,SOURCE")
        )["filesystems"][0]
        if fs["fstype"] == "fuse.gvfsd-fuse":
            share = path.relative_to(fs["target"]).parts[0]
            if not share.startswith("smb-share:"):
                return False
            parts = dict(pair.split("=", 1) for pair in share[10:].split(","))
            host, share = parts["server"], unquote(parts["share"])
        elif fs["fstype"] in {"cifs", "smb3"}:
            host, share = fs["source"].removeprefix("//").split("/", 1)
        else:
            return False
        return not connection or (host.lower(), share.lower()) == (
            connection["host"].lower(),
            connection["share"].lower(),
        )
    except (OSError, subprocess.SubprocessError, KeyError, ValueError, IndexError):
        return False


def _snapshot_end(db, stream, snapshot):
    """Pin pages to an event sequence in this archive generation."""
    high = db.execute("select coalesce(max(seq),0) from event").fetchone()[0]
    if snapshot is None:
        return high
    identity, _, seq = snapshot.rpartition(":")
    if identity == stream and seq.isascii() and seq.isdecimal() and int(seq) <= high:
        return int(seq)
    raise InvalidSnapshot("Invalid or retired run snapshot; reload the runs")


def runs_page(db, plugin, *, end, offset, limit):
    table = {"thermo": "qc_raw", "bruker": "bruker_run"}[plugin]
    rows = db.execute(
        f"""
        select * from (
          select e.*, detail, row_number() over w as n,
                 min(recorded_at) over (partition by root_alias,relative_path) as first_at
          from event e join {table} using(seq) where seq <= ?
          window w as (partition by root_alias,relative_path order by seq desc)
        ) where n=1 order by observed_at desc,seq desc limit ? offset ?
    """,
        (end, limit + 1, offset),
    ).fetchall()
    items = [
        {
            "run_id": r["root_alias"] + "/" + r["relative_path"],
            "name": Path(r["relative_path"]).name,
            "relative_path": r["relative_path"],
            "root_alias": r["root_alias"],
            "project_id": None,
            "first_observed_at": r["first_at"],
            "last_observed_at": r["recorded_at"],
            "source_modified_at": r["observed_at"],
            "size_bytes": json.loads(r["detail"])["bytes"],
            "evidence": table,
            "supported_state": None,
        }
        for r in rows[:limit]
    ]
    return dict(items=items, has_more=len(rows) > limit, limit=limit, offset=offset)


def read_archive(row, *, limit=1, offset=0, snapshot=None, runs=False):
    """Missing means never contacted. Reject non-files, linked files and other streams."""
    path = archive_path(row)
    empty = (
        dict(
            items=[],
            has_more=False,
            offset=offset,
            limit=limit,
            snapshot=None,
            truncated=False,
        )
        if runs
        else None
    )
    try:
        if path.is_symlink():
            raise ArchiveError("Archive link rejected")
        if not path.exists():
            if row.archive_seen:
                raise ArchiveError("Previously observed archive is missing")
            if snapshot is not None:
                raise InvalidSnapshot("Run snapshot unavailable; reload the runs")
            return empty
        if not path.is_file() or path.stat().st_nlink != 1:
            raise ArchiveError("Invalid archive file")
        with closing(
            sqlite3.connect(
                path.absolute().as_uri() + "?mode=ro", uri=True, timeout=0.2
            )
        ) as db:
            db.row_factory = sqlite3.Row
            db.execute("pragma query_only=on")
            db.execute("pragma trusted_schema=off")
            db.execute("begin")
            header = db.execute(
                "select instrument_id,plugin,version,stream_id from stream"
            ).fetchone()
            plugin = row.settings["plugin"]
            if not header or tuple(header[:3]) != (
                row.id,
                plugin,
                {"thermo": 1, "bruker": 2}[plugin],
            ):
                raise ArchiveError("Unsupported or mismatched archive")
            stream = header["stream_id"]
            end = _snapshot_end(db, stream, snapshot)
            page = runs_page(db, plugin, end=end, offset=offset, limit=limit)
            scan = db.execute(
                "select recorded_at,detail from event join scan using(seq) where seq <= ? order by seq desc limit 1",
                (end,),
            ).fetchone()
            info = json.loads(scan["detail"]) if scan else {}
            if runs:
                return {
                    **page,
                    "snapshot": f"{stream}:{end}",
                    "truncated": bool(info.get("scope", {}).get("truncated")),
                }
            if not scan:
                return None
            last_ok = db.execute(
                "select max(recorded_at) from event join scan using(seq) where json_extract(detail,'$.status')='readable'"
            ).fetchone()[0]
            event = db.execute(
                "select * from event where kind!='scan' order by seq desc limit 1"
            ).fetchone()
            latest = None
            if event:
                kind = event["kind"]
                detail = {}
                if kind in {"tng_record", "lc_transition"}:
                    detail = json.loads(
                        db.execute(
                            f"select detail from {kind} where seq=?", (event["seq"],)
                        ).fetchone()[0]
                    )
                latest = dict(
                    kind=detail.get("kind", kind),
                    description=detail.get("message")
                    or detail.get("procedure_state")
                    or event["relative_path"],
                    observed_at=event["observed_at"],
                    recorded_at=event["recorded_at"],
                    source_time_text=detail.get("source_time_local"),
                    source_time_zone=detail.get("source_timezone"),
                )
            scope = info.get("scope", {})
            coverage = info["coverage"]
            error = None
            if (
                "mount_unavailable" in coverage.values()
                or info["status"] == "unavailable"
            ):
                error = dict(
                    code="scan_unavailable",
                    message="Source unavailable at the last scan",
                    at=scan[0],
                )
            return dict(
                contract_version=2,
                last_collector_contact_at=scan[0],
                last_successful_scan_at=last_ok,
                last_mirror_receipt_at=None,
                reported_state=None,
                latest_source_event=latest,
                error=error,
                latest_run=next(iter(page["items"]), None),
                coverage=dict(
                    entries_examined=scope.get("entries_examined"),
                    truncated=scope.get("truncated"),
                    roots=[dict(alias=k, state=v) for k, v in coverage.items()],
                ),
            )
    except (sqlite3.Error, OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ArchiveError("Instrument archive is unavailable or invalid") from exc


def card(row):
    try:
        summary = read_archive(row)
    except ArchiveError as exc:
        return dict(
            summary=None,
            summary_error=dict(code="archive_unavailable", message=str(exc)),
        )
    error = None
    if row.settings["mode"] == "pull" and row.settings["enabled"]:
        missing = [
            alias
            for alias, root in row.settings["roots"].items()
            if root
            and not mounted(root, row.settings.get("connections", {}).get(alias))
        ]
        if missing:
            error = dict(
                code="source_not_mounted",
                message="Source not mounted or accessible: " + ", ".join(missing),
            )
    return dict(summary=summary, summary_error=error)
