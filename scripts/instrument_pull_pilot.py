"""One bounded pull per instrument into two private scratch DBs on the Linux reader."""

import argparse
from contextlib import closing
from datetime import timedelta
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import tempfile

from ispec.instruments.collector import CollectorConfig, collect
from ispec.timeutil import format_utc, parse_utc, utcnow
from ispec_observer.bruker_lc import attach_lc, collect_lc
from ispec_observer.log_io import bounded_read
from ispec_observer.pilot_plugins import bruker_scan, thermo_scan
from ispec_observer.pilot_store import Archive
from ispec_observer.qc import collect_qc_bounded, load_qc_config
from ispec_observer.thermo import (
    collect_thermo_bounded,
    config_key,
    empty_state,
    expire_evidence,
)
from ispec_observer.thermo_config import load_thermo_config


def mounted(path):
    """Require an actual SMB filesystem or a GVFS SMB share, not its empty mountpoint."""
    try:
        resolved = subprocess.check_output(
            ["realpath", "-e", "--", path],
            timeout=4,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        mount = json.loads(
            subprocess.check_output(
                ["findmnt", "-J", "-o", "TARGET,FSTYPE", "-T", resolved],
                text=True,
                timeout=4,
            )
        )["filesystems"][0]
        if mount["fstype"] in {"smb3", "cifs"}:
            return True
        return mount["fstype"] == "fuse.gvfsd-fuse" and Path(resolved).relative_to(
            mount["target"]
        ).parts[0].startswith("smb-share:")
    except (subprocess.SubprocessError, ValueError, IndexError):
        return False


def stamp(value):
    return format_utc(parse_utc(value))


def eclipse(cfg, qc_cfg, previous, *, source_ok=mounted):
    tng = {"error": "mount_unavailable"}
    qc = {"error": "mount_unavailable", "files": []}
    log_mounted, msi_mounted = (source_ok(cfg[key]) for key in ("log_root", "msi_root"))
    if log_mounted or msi_mounted:
        state = previous.get("tng") or empty_state(cfg)
        if state["config_key"] != config_key(cfg):
            raise ValueError(
                "Source configuration changed; choose a fresh scratch directory"
            )
        # Do not scan an ordinary local directory left behind by a missing mount.
        inputs = dict(cfg)
        for key, available in (("log_root", log_mounted), ("msi_root", msi_mounted)):
            if not available:
                inputs[key] = "/dev/null/missing-mount"
        tng = collect_thermo_bounded(inputs, state)
    if source_ok(qc_cfg["root"]):
        qc = collect_qc_bounded(qc_cfg)
    scan = thermo_scan(previous, tng, qc, stamp)
    if not log_mounted:
        scan.coverage["tng"] = "mount_unavailable"
    if not msi_mounted:
        scan.coverage["msi"] = "mount_unavailable"
    if "tng" in scan.cursor:
        scan.cursor["tng"] = expire_evidence(
            scan.cursor["tng"], cfg.get("retention_seconds"), utcnow().isoformat()
        )
    return scan, {
        "qc_files": len(qc["files"]),
        "selected_logs": tng.get("state", {}).get("selected", {}),
        "log_reads": tng.get("reads", {}),
        "problems": tng.get("state", {}).get("problems", {}),
    }


def tims(cfg, previous, *, lc_root=None, source_ok=mounted):
    bound = previous.get("lc_root")
    if lc_root is not None and bound not in (None, str(lc_root)):
        raise ValueError("Archive already bound to a different Bruker LC log root")
    result = (
        collect(cfg)
        if source_ok(cfg.root)
        else {
            "source_state": "mount_unavailable",
            "recent_runs": [],
            "read_errors": ["mount_unavailable"],
        }
    )
    scan = bruker_scan(result, stamp)
    diagnostics = {
        k: result.get(k) for k in ("entries_examined", "truncated", "read_errors")
    }
    if lc_root is not None or bound is not None:
        lc_state = previous.get("lc", {})
        if lc_root is None:
            lc = {"error": "not_requested"}
        elif not source_ok(lc_root):
            lc = {"error": "mount_unavailable"}
        else:
            lc = bounded_read(
                collect_lc,
                (lc_root, lc_state, format_utc(utcnow())),
                timeout=cfg.timeout_seconds,
            )
        scan.cursor["lc_root"] = str(lc_root) if lc_root is not None else bound
        attach_lc(scan, lc_state, lc)
        diagnostics["lc_reads"] = lc.get("reads", {})
    return scan, diagnostics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("scratch", "bruker"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--thermo", type=Path)
    parser.add_argument("--qc", type=Path)
    parser.add_argument(
        "--bruker-only", action="store_true", help="Skip the Eclipse pull"
    )
    parser.add_argument(
        "--bruker-lc-root", type=Path, help="Read the mounted HyStar LogFiles directory"
    )
    args = parser.parse_args()
    if not args.bruker_only and not (args.thermo and args.qc):
        parser.error("Supply --thermo and --qc or select --bruker-only")
    if args.bruker_lc_root and not args.bruker_lc_root.is_absolute():
        parser.error("--bruker-lc-root must be an absolute mounted directory")
    scratch = args.scratch.resolve(strict=True)
    filesystem = subprocess.check_output(
        ["findmnt", "-n", "-T", str(scratch), "-o", "FSTYPE"], text=True, timeout=4
    ).strip()
    if (
        scratch.parent != Path(tempfile.gettempdir()).resolve()
        or args.scratch.is_symlink()
        or filesystem not in {"ext4", "xfs", "btrfs", "tmpfs", "overlay"}
        or scratch.stat().st_uid != os.getuid()
        or scratch.stat().st_mode & 0o077
    ):
        parser.error(
            "Use a private, owner-only scratch directory directly under the local temporary directory"
        )
    br = CollectorConfig(**json.loads(args.bruker.read_text()))
    os.umask(0o077)
    targets = []
    if not args.bruker_only:
        cfg = load_thermo_config(args.thermo)
        qc = load_qc_config(args.qc)
        if cfg["node_id"] != qc["node_id"] or cfg["sequence_files"]:
            parser.error(
                "Expected one Eclipse QC/TNG instrument and no optional sequence inputs"
            )
        targets.append(
            (
                "thermo",
                cfg["node_id"],
                [cfg["log_root"], cfg["msi_root"], qc["root"]],
                lambda state: eclipse(cfg, qc, state),
            )
        )
    targets.append(
        (
            "bruker",
            br.instrument_id,
            [br.root],
            lambda state: tims(br, state, lc_root=args.bruker_lc_root),
        )
    )
    for plugin, node, roots, reader in targets:
        path = scratch / (plugin + ".sqlite")
        if path.is_symlink() or (path.exists() and path.stat().st_nlink != 1):
            raise ValueError("Refusing linked pilot archive")
        with closing(sqlite3.connect(path)) as db:
            archive = Archive(db, node, plugin)
            state, generation = archive.checkpoint()
            source = {
                "roots": roots,
                "config": config_key(cfg)
                if plugin == "thermo"
                else [br.max_depth, br.max_entries],
                "qc_pattern": qc["pattern"] if plugin == "thermo" else None,
            }
            if state.get("source", source) != source:
                raise ValueError("Archive already bound to a different source scope")
            scan, coverage = reader(state)
            scan.cursor["source"] = source
            now = utcnow()
            archive.apply(
                scan,
                generation=generation,
                recorded_at=format_utc(now),
                heartbeat_before=format_utc(now - timedelta(hours=1)),
            )
            print(
                json.dumps(dict(archive.summary(3), diagnostics=coverage)), flush=True
            )


if __name__ == "__main__":
    main()
