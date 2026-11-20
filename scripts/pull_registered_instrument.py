"""Run one bounded, read-only vendor scan using this host's registry."""

import argparse
from contextlib import closing
from datetime import timedelta
import os
from pathlib import Path
import sqlite3

from instrument_pull_pilot import eclipse, tims
from ispec.db.connect import get_session
from ispec.db.models import InstrumentRegistry
from ispec.instruments.collector import CollectorConfig
from ispec.instruments.dashboard import mounted
from ispec.instruments.registry import archive_path
from ispec.timeutil import format_utc, utcnow
from ispec_observer.pilot_store import Archive
from ispec_observer.qc import load_qc_config
from ispec_observer.thermo_config import load_thermo_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("id")
    parser.add_argument(
        "--thermo", type=Path, help="Local Thermo reader bounds/fields template"
    )
    parser.add_argument("--qc", type=Path, help="Local QC selection/bounds template")
    args = parser.parse_args()
    with get_session() as core:
        row = core.get(InstrumentRegistry, args.id)
        if not row or not row.settings["enabled"] or row.settings["mode"] != "pull":
            parser.error("An enabled, registered pull instrument is required")
        path = archive_path(row)
        registry_generation = row.generation
        settings = row.settings
    plugin = settings["plugin"]
    roots = settings["roots"]
    connections = settings["connections"]

    def approved(root):
        return any(
            p == str(root) and mounted(p, connections.get(k))
            for k, p in roots.items()
            if p
        )

    if plugin == "thermo":
        if not (args.thermo and args.qc):
            parser.error("Thermo requires --thermo and --qc reader templates")
        cfg = load_thermo_config(args.thermo)
        qc = load_qc_config(args.qc)
        if cfg["sequence_files"]:
            parser.error("Additional sequence sources are outside registry scope")
        cfg.update(
            node_id=args.id,
            log_root=roots.get("log") or "/dev/null/absent",
            msi_root=roots.get("tune") or "/dev/null/absent",
            msi_source_path=roots.get("tune") or "",
            state_file=None,
            report_file=None,
            history_db=None,
        )
        qc.update(node_id=args.id, root=roots["acquisition"])

        def read(state):
            return eclipse(cfg, qc, state, source_ok=approved)
    else:
        cfg = CollectorConfig(
            collector_id=settings["collector_id"] or args.id,
            instrument_id=args.id,
            root=roots["acquisition"],
            max_depth=0,
            max_entries=500,
            timeout_seconds=8,
        )

        def read(state):
            return tims(cfg, state, lc_root=roots.get("log"), source_ok=approved)

    os.umask(0o077)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if path.is_symlink() or (path.exists() and path.stat().st_nlink != 1):
        parser.error("Refusing linked archive")
    with closing(sqlite3.connect(path)) as db:
        archive = Archive(db, args.id, plugin)
        state, generation = archive.checkpoint()
        scan, _ = read(state)
        now = utcnow()
        archive.apply(
            scan,
            generation=generation,
            recorded_at=format_utc(now),
            heartbeat_before=format_utc(
                now - timedelta(seconds=settings["expected_heartbeat_seconds"])
            ),
        )
        print(args.id, scan.status, scan.coverage, path)
    with get_session() as core:
        row = core.get(InstrumentRegistry, args.id)
        if row.generation == registry_generation and not row.archive_seen:
            row.archive_seen = True


if __name__ == "__main__":
    main()
