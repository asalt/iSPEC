"""Portable Python 3.10+ metadata collector; safe to copy as a single file.

No iSPEC imports, vendor SDK, shell commands, raw-data reads or instrument writes.
Preview is the default. Network submission and foreground monitoring are opt-in.
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
import ipaddress
import json
import multiprocessing
import os
from pathlib import Path
import re
import stat
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid


@dataclass(frozen=True, kw_only=True)
class CollectorConfig:
    collector_id: str
    instrument_id: str
    root: str
    max_depth: int = 2
    max_entries: int = 2000
    timeout_seconds: int = 8

    def __post_init__(self):
        for value in (self.collector_id, self.instrument_id):
            if not re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", value):
                raise ValueError(
                    "IDs require letters, digits, dot, dash or underscore."
                )
        if (
            not isinstance(self.root, str)
            or not Path(self.root).is_absolute()
            or len(self.root) > 4096
        ):
            raise ValueError("root must be an absolute, operator-selected directory.")
        for name, low, high in (
            ("max_depth", 0, 4),
            ("max_entries", 1, 10000),
            ("timeout_seconds", 1, 30),
        ):
            value = getattr(self, name)
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or not low <= value <= high
            ):
                raise ValueError(f"Invalid {name}.")


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def blank_observation(cfg: CollectorConfig) -> dict:
    return dict(
        schema_version=1,
        collector_version="1",
        event_id=str(uuid.uuid4()),
        collector_id=cfg.collector_id,
        instrument_id=cfg.instrument_id,
        observed_at=utcnow(),
        source_kind="directory_metadata",
        source_root=cfg.root,
        resolved_root=None,
        source_state="unavailable",
        source_updated_at=None,
        entries_examined=0,
        truncated=False,
        recent_runs=[],
        read_errors=[],
        instrument_state="unknown",
        progress=None,
        instrument_error=None,
        instrument_heartbeat_at=None,
        pressure=None,
        flow=None,
    )


def is_link(metadata: os.stat_result) -> bool:
    return stat.S_ISLNK(metadata.st_mode) or bool(
        getattr(metadata, "st_file_attributes", 0) & 0x400
    )


def scan_directory(cfg: CollectorConfig) -> dict:
    result = blank_observation(cfg)
    errors = result["read_errors"]

    def error(code):
        if len(errors) < 20:
            errors.append(code)

    try:
        # The configured root can be an operator-managed mount symlink; never
        # follow symlinks/junctions discovered underneath that root.
        root = Path(cfg.root).resolve(strict=True)
        result["resolved_root"] = str(root)
        if not root.is_dir():
            raise NotADirectoryError()
    except OSError as exc:
        error(type(exc).__name__)
        return result
    todo = deque([(root, 0)])
    runs = []
    root_read = False
    while todo:
        directory, depth = todo.popleft()
        try:
            # Recheck queued directories in case they were replaced by links.
            if is_link(directory.lstat()):
                error("link_skipped")
                continue
            with os.scandir(directory) as entries:
                if directory == root:
                    root_read = True
                for entry in entries:
                    if result["entries_examined"] >= cfg.max_entries:
                        result["truncated"] = True
                        todo.clear()
                        break
                    result["entries_examined"] += 1
                    try:
                        metadata = entry.stat(follow_symlinks=False)
                        if is_link(metadata):
                            continue
                        if not stat.S_ISDIR(metadata.st_mode):
                            continue
                        path = Path(entry.path)
                        if path.suffix.lower() != ".d":
                            if depth < cfg.max_depth:
                                todo.append((path, depth + 1))
                            else:
                                result["truncated"] = True
                            continue
                        updated = metadata.st_mtime
                        total = 0
                        last_file = None
                        last_file_time = float("-inf")
                        for name in (
                            "analysis.tdf",
                            "analysis.tdf_bin",
                            "analysis.baf",
                            "SampleInfo.xml",
                            "HyStarMetadata.xml",
                        ):
                            try:
                                item = path / name
                                info = item.lstat()
                                if not is_link(info) and stat.S_ISREG(info.st_mode):
                                    total += info.st_size
                                    updated = max(updated, info.st_mtime)
                                    if info.st_mtime > last_file_time:
                                        last_file = item.relative_to(root).as_posix()
                                        last_file_time = info.st_mtime
                            except FileNotFoundError:
                                pass
                            except OSError as exc:
                                error(type(exc).__name__)
                        runs.append(
                            dict(
                                relative_path=path.relative_to(root).as_posix(),
                                modified_at=datetime.fromtimestamp(
                                    updated, timezone.utc
                                ).isoformat(),
                                observed_bytes=total,
                                last_file=last_file,
                            )
                        )
                        runs.sort(
                            key=lambda row: (row["modified_at"], row["relative_path"]),
                            reverse=True,
                        )
                        del runs[20:]
                    except (OSError, ValueError) as exc:
                        error(type(exc).__name__)
        except OSError as exc:
            error(type(exc).__name__)
    result["recent_runs"] = runs
    result["source_updated_at"] = runs[0]["modified_at"] if runs else None
    result["source_state"] = (
        "unavailable"
        if not root_read
        else ("partial" if errors or result["truncated"] else "readable")
    )
    return result


def scan_worker(cfg, connection):
    try:
        connection.send(scan_directory(cfg))
    except Exception:
        result = blank_observation(cfg)
        result["read_errors"] = ["collector_worker_failed"]
        connection.send(result)
    finally:
        connection.close()


def collect(cfg: CollectorConfig) -> dict:
    # A blocked SMB stat must not hang the caller indefinitely. The deadline
    # includes worker startup and the entire scan, not merely loop iterations.
    context = multiprocessing.get_context("spawn")
    reader, writer = context.Pipe(duplex=False)
    worker = context.Process(target=scan_worker, args=(cfg, writer), daemon=True)
    worker.start()
    writer.close()
    try:
        if reader.poll(cfg.timeout_seconds):
            try:
                return reader.recv()
            except EOFError:
                result = blank_observation(cfg)
                result["read_errors"] = ["collector_worker_failed"]
                return result
        result = blank_observation(cfg)
        result.update(source_state="timeout", read_errors=["scan_timeout"])
        return result
    finally:
        reader.close()
        worker.join(timeout=0.1)
        if worker.is_alive():
            worker.terminate()
            worker.join(timeout=1)
            if worker.is_alive():
                worker.kill()
                worker.join(timeout=1)


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError("Collector redirects are not allowed.")


def endpoint_url(server: str) -> str:
    url = urllib.parse.urlsplit(server)
    try:
        loopback = ipaddress.ip_address(url.hostname or "").is_loopback
    except ValueError:
        loopback = url.hostname == "localhost"
    if (
        (url.scheme != "https" and not (url.scheme == "http" and loopback))
        or not url.hostname
        or url.username
        or url.password
        or url.query
        or url.fragment
        or url.path not in {"", "/"}
    ):
        raise ValueError(
            "Use an HTTPS server origin, or loopback HTTP over an approved tunnel; no URL credentials or paths."
        )
    return server.rstrip("/") + "/api/instruments/observations"


def send_observation(server: str, observation: dict, token: str) -> dict:
    endpoint = endpoint_url(server)
    if len(token) < 32:
        raise ValueError("Dedicated collector token is missing or too short.")
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(observation).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer " + token,
        },
    )
    try:
        # Do not route local research metadata through ambient HTTP proxies.
        with urllib.request.build_opener(
            urllib.request.ProxyHandler({}), NoRedirect()
        ).open(request, timeout=10) as response:
            raw = response.read(65537)
        if len(raw) > 65536:
            raise ValueError("oversize receipt")
        receipt = json.loads(raw)
        if (
            receipt.get("event_id") != observation["event_id"]
            or receipt.get("stored") is not True
        ):
            raise ValueError("invalid receipt")
        return receipt
    except urllib.error.HTTPError as exc:
        raise ValueError(f"Ingestion HTTP {exc.code}; no confirmed receipt.") from None
    except (OSError, ValueError, AttributeError):
        raise ValueError(
            "Ingestion failed or receipt is invalid; delivery is unconfirmed."
        ) from None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        required=True,
        help="Local JSON file: collector_id, instrument_id, absolute root and optional scan limits.",
    )
    parser.add_argument(
        "--send",
        action="store_true",
        help="Explicitly submit to --server using ISPEC_INSTRUMENT_COLLECTOR_TOKEN.",
    )
    parser.add_argument(
        "--server", help="iSPEC HTTPS origin or approved loopback tunnel origin."
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Foreground observation loop; Ctrl-C stops it. No service is installed.",
    )
    parser.add_argument("--interval-seconds", type=int, default=60)
    args = parser.parse_args(argv)
    try:
        with Path(args.config).open("rb") as stream:
            raw = stream.read(32769)
        if len(raw) > 32768:
            raise ValueError("Configuration exceeds 32 KiB.")
        cfg = CollectorConfig(**json.loads(raw))
        if not 30 <= args.interval_seconds <= 86400:
            raise ValueError("Interval must be between 30 and 86400 seconds.")
        token = os.getenv("ISPEC_INSTRUMENT_COLLECTOR_TOKEN", "")
        if args.send:
            endpoint_url(args.server or "")
            if len(token) < 32:
                raise ValueError(
                    "Set a dedicated ISPEC_INSTRUMENT_COLLECTOR_TOKEN before sending."
                )
    except (OSError, ValueError, TypeError):
        print(
            json.dumps(
                {
                    "ok": False,
                    "error": "Invalid configuration, endpoint or missing dedicated token; nothing sent.",
                }
            )
        )
        return 2
    pending = None
    try:
        while True:
            if pending is None:
                pending = collect(cfg)
            try:
                output = {"ok": True, "sent": False, "observation": pending}
                if args.send:
                    output = {
                        "ok": True,
                        "sent": True,
                        "receipt": send_observation(args.server, pending, token),
                    }
                pending = None
            except ValueError as exc:
                output = {
                    "ok": False,
                    "sent": None,
                    "event_id": pending["event_id"],
                    "error": str(exc),
                }
            print(json.dumps(output, allow_nan=False), flush=True)
            if not args.watch:
                return 0 if output["ok"] else 1
            # Keep the same pending ID/evidence after uncertain delivery.
            # No persistent backlog and no fast retry loop.
            time.sleep(args.interval_seconds)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    multiprocessing.freeze_support()
    sys.exit(main())
