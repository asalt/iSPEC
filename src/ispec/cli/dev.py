"""Dev helpers for controlling local iSPEC services (tmux-based).

These are intentionally lightweight wrappers around the top-level Makefile +
tmux layout used by `scripts/dev-tmux.sh`.
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess

from ispec.logging import get_logger


logger = get_logger(__file__)


_DEFAULT_TMUX_SESSION = "ispecfull"

_SERVICE_TO_WINDOW = {
    "backend": "backend",
    "frontend": "frontend",
    "supervisor": "supervisor",
    "vllm": "vllm",
    "slack": "slack",
}

_SERVICE_TO_MAKE_TARGET = {
    "backend": "dev-backend",
    "frontend": "dev-frontend",
    "supervisor": "dev-supervisor",
    "vllm": "dev-vllm",
    "slack": "dev-slack",
}


def _find_make_root(start: Path | None = None) -> Path | None:
    here = (start or Path.cwd()).resolve()
    for parent in [here, *here.parents]:
        candidate = parent / "Makefile"
        if not candidate.is_file():
            continue
        # Cheap sniff test: only treat it as the root if it looks like the
        # ispec-full helper Makefile.
        try:
            text = candidate.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if "dev-tmux" in text and "dev-supervisor" in text and "dev-backend" in text:
            return parent
    return None


def _tmux(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["tmux", *args],
        text=True,
        capture_output=True,
    )


def _tmux_session_name(raw: str | None) -> str:
    value = (raw or "").strip()
    if value:
        return value
    value = (os.getenv("DEV_TMUX_SESSION") or "").strip()
    return value or _DEFAULT_TMUX_SESSION


def _tmux_has_session(session: str) -> bool:
    proc = _tmux("has-session", "-t", session)
    return proc.returncode == 0


def _tmux_first_pane_id(target_window: str) -> str | None:
    proc = _tmux("list-panes", "-t", target_window, "-F", "#{pane_id}")
    if proc.returncode != 0:
        return None
    for line in (proc.stdout or "").splitlines():
        pane_id = line.strip()
        if pane_id:
            return pane_id
    return None


def register_subcommands(subparsers) -> None:
    restart_parser = subparsers.add_parser(
        "restart",
        help="Restart dev services inside the dev tmux session (backend/supervisor/etc.)",
    )
    restart_parser.add_argument(
        "services",
        nargs="*",
        help="Services to restart (default: backend supervisor). Options: backend, supervisor, frontend, vllm, slack.",
    )
    restart_parser.add_argument(
        "--tmux-session",
        default=None,
        help="tmux session name (default: $DEV_TMUX_SESSION or ispecfull).",
    )
    restart_parser.add_argument(
        "--make-root",
        default=None,
        help="Path to the directory containing the top-level Makefile (auto-detected by default).",
    )


def dispatch(args) -> None:
    if args.subcommand != "restart":
        raise SystemExit(f"Unknown dev subcommand: {args.subcommand}")

    session = _tmux_session_name(getattr(args, "tmux_session", None))
    make_root_raw = (getattr(args, "make_root", None) or "").strip()
    make_root = Path(make_root_raw).expanduser().resolve() if make_root_raw else _find_make_root()
    if make_root is None:
        raise SystemExit("Unable to locate the top-level Makefile; pass --make-root.")

    services = list(getattr(args, "services", []) or [])
    if not services:
        services = ["backend", "supervisor"]

    unknown = [name for name in services if name not in _SERVICE_TO_WINDOW]
    if unknown:
        raise SystemExit(f"Unknown services: {unknown}. Known: {sorted(_SERVICE_TO_WINDOW)}")

    if not _tmux_has_session(session):
        raise SystemExit(f"tmux session not found: {session!r} (start it with `make dev-tmux`).")

    for service in services:
        window = _SERVICE_TO_WINDOW[service]
        make_target = _SERVICE_TO_MAKE_TARGET[service]
        target_window = f"{session}:{window}"
        pane_id = _tmux_first_pane_id(target_window)
        if not pane_id:
            raise SystemExit(f"Unable to find a pane for tmux window {target_window!r}.")

        cmd = f'cd "{make_root.as_posix()}" && make {make_target}'
        logger.info("Restarting %s via tmux pane=%s cmd=%s", service, pane_id, cmd)
        proc = _tmux("respawn-pane", "-k", "-t", pane_id, cmd)
        if proc.returncode != 0:
            stderr = (proc.stderr or "").strip()
            raise SystemExit(f"tmux respawn-pane failed for {service!r}: {stderr or 'unknown error'}")

