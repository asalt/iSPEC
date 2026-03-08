"""Command-line helpers for controlling the iSPEC API service.

This module exposes functions to register API-related subcommands on an
``argparse`` parser and to dispatch the parsed arguments to their respective
handlers.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import time

import requests

from ispec.config.paths import resolve_api_pid_file, resolve_api_state_file
from ispec.logging import get_logger

_STATUS_ENDPOINT = "/status"
_REQUEST_TIMEOUT = 2.0


def _state_file_path() -> Path:
    """Return the filesystem path used to persist API server state."""

    resolved = resolve_api_state_file()
    return Path(resolved.path or resolved.value)


def _pid_file_path() -> Path:
    """Return the filesystem path used to persist the API server PID."""

    resolved = resolve_api_pid_file()
    return Path(resolved.path or resolved.value)


def _write_pid_file(*, logger) -> Path | None:
    path = _pid_file_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{os.getpid()}\n")
        logger.debug("Recorded API server pid in %s", path)
        return path
    except OSError as exc:
        logger.warning("Unable to record API server pid in %s: %s", path, exc)
        return None


def _remove_pid_file(path: Path | None, *, logger) -> None:
    if path is None:
        path = _pid_file_path()
    try:
        path.unlink()
        logger.debug("Removed API server pid file %s", path)
    except FileNotFoundError:
        pass
    except OSError as exc:
        logger.warning("Unable to remove API server pid file %s: %s", path, exc)


def _write_state(host: str, port: int, *, logger) -> Path | None:
    """Persist the server's host/port so ``status`` can locate it later."""

    path = _state_file_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"host": host, "port": int(port), "pid": os.getpid()}
        path.write_text(json.dumps(payload))
        logger.debug("Recorded API server state in %s", path)
        return path
    except OSError as exc:
        logger.warning("Unable to record API server state in %s: %s", path, exc)
        return None


def _remove_state(path: Path | None, *, logger) -> None:
    """Delete the persisted state file, ignoring if it is already gone."""

    if path is None:
        path = _state_file_path()
    try:
        path.unlink()
        logger.debug("Removed API server state file %s", path)
    except FileNotFoundError:
        pass
    except OSError as exc:
        logger.warning("Unable to remove API server state file %s: %s", path, exc)


def _read_state(*, logger) -> tuple[str, int] | None:
    """Return the stored ``(host, port)`` pair if available and valid."""

    path = _state_file_path()
    try:
        raw = path.read_text()
    except FileNotFoundError:
        return None
    except OSError as exc:
        logger.debug("Unable to read API server state from %s: %s", path, exc)
        return None

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.warning("Ignoring corrupt API server state file %s: %s", path, exc)
        return None

    host = data.get("host")
    port = data.get("port")
    if not isinstance(host, str):
        logger.warning("State file %s missing 'host'; treating API as stopped", path)
        return None

    try:
        port_int = int(port)
    except (TypeError, ValueError):
        logger.warning("State file %s missing valid 'port'; treating API as stopped", path)
        return None

    return host, port_int


def _read_state_payload(*, logger) -> dict[str, object] | None:
    """Return the stored state payload if available and valid."""

    path = _state_file_path()
    try:
        raw = path.read_text()
    except FileNotFoundError:
        return None
    except OSError as exc:
        logger.debug("Unable to read API server state from %s: %s", path, exc)
        return None

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.warning("Ignoring corrupt API server state file %s: %s", path, exc)
        return None

    host = data.get("host")
    port = data.get("port")
    pid = data.get("pid")
    if not isinstance(host, str) or not host:
        logger.warning("State file %s missing 'host'; treating API as stopped", path)
        return None

    try:
        port_int = int(port)
    except (TypeError, ValueError):
        logger.warning("State file %s missing valid 'port'; treating API as stopped", path)
        return None

    payload: dict[str, object] = {"host": host, "port": port_int}
    try:
        if pid is not None:
            payload["pid"] = int(pid)
    except (TypeError, ValueError):
        pass
    return payload


def _probe_host(host: str) -> str:
    """Return the hostname to probe for status checks."""

    if host in {"0.0.0.0", "::"}:
        return "127.0.0.1"
    return host


def _is_local_bind_host(host: str) -> bool:
    return host in {"127.0.0.1", "localhost", "::1"}


def _is_server_running(host: str, port: int, *, logger) -> bool:
    """Return ``True`` if the FastAPI server responds to its status endpoint."""

    probe_host = _probe_host(host)
    url = f"http://{probe_host}:{port}{_STATUS_ENDPOINT}"
    try:
        response = requests.get(url, timeout=_REQUEST_TIMEOUT)
    except requests.RequestException as exc:
        logger.debug("Status probe failed for %s: %s", url, exc)
        return False

    if response.status_code != 200:
        logger.debug(
            "Status probe for %s returned unexpected status %s",
            url,
            response.status_code,
        )
        return False

    try:
        payload = response.json()
    except ValueError as exc:
        logger.debug("Status probe for %s returned invalid JSON: %s", url, exc)
        return False

    return bool(payload.get("ok"))


def register_subcommands(subparsers):
    """Register API subcommands on the provided ``argparse`` object.

    Parameters
    ----------
    subparsers : :class:`argparse._SubParsersAction`
        The ``argparse`` subparsers object to which API commands are added.

    Examples
    --------
    >>> import argparse
    >>> parser = argparse.ArgumentParser(prog="ispec api")
    >>> subparsers = parser.add_subparsers(dest="subcommand", required=True)
    >>> register_subcommands(subparsers)
    >>> parser.parse_args(["start", "--host", "0.0.0.0", "--port", "9000"])
    Namespace(subcommand='start', host='0.0.0.0', port=9000)

    """

    _ = subparsers.add_parser("status", help="Check whether the API is running")
    starter_parser = subparsers.add_parser("start", help="start the API server")
    starter_parser.add_argument(
        "--host", default="localhost", help="Host to run the API server "
    )
    starter_parser.add_argument(
        "--port", type=int, default=8000, help="Port to run the API server on"
    )
    stopper_parser = subparsers.add_parser("stop", help="Stop the API server")
    stopper_parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=5.0,
        help="Seconds to wait for shutdown after signaling (default: 5.0)",
    )


def dispatch(args):
    """Execute the API command associated with ``args.subcommand``.

    Parameters
    ----------
    args : :class:`argparse.Namespace`
        Parsed arguments containing a ``subcommand`` attribute and any
        additional options required by that subcommand.

    Examples
    --------
    >>> import types
    >>> args = types.SimpleNamespace(subcommand="status")
    >>> dispatch(args)  # doctest: +SKIP

    When starting the API server::

        >>> args = types.SimpleNamespace(subcommand="start", host="127.0.0.1", port=8000)
        >>> dispatch(args)  # doctest: +SKIP

    """

    logger = get_logger(__file__)

    if args.subcommand == "status":
        state = _read_state(logger=logger)
        if state is None:
            logger.info("API server is not running.")
            return

        host, port = state
        if _is_server_running(host, port, logger=logger):
            logger.info("API server is running at %s:%s", host, port)
        else:
            logger.info("API server is not running at %s:%s", host, port)
        return

    if args.subcommand == "stop":
        payload = _read_state_payload(logger=logger)
        if payload is None:
            logger.info("API server is not running.")
            return

        host = str(payload.get("host") or "")
        port = int(payload.get("port") or 0)
        pid = payload.get("pid")
        pid_int = int(pid) if isinstance(pid, int) and pid > 0 else None

        if pid_int is None:
            logger.warning("API state file did not include a pid; cannot stop cleanly.")
            return

        logger.info("Stopping API server pid=%s (host=%s port=%s)", pid_int, host, port)
        try:
            os.kill(pid_int, signal.SIGTERM)
        except ProcessLookupError:
            logger.info("API server process is already gone (pid=%s).", pid_int)
            _remove_state(None, logger=logger)
            _remove_pid_file(None, logger=logger)
            return
        except PermissionError as exc:
            logger.error("Permission denied signaling pid=%s: %s", pid_int, exc)
            raise SystemExit(2) from exc

        deadline = time.monotonic() + float(getattr(args, "timeout_seconds", 5.0) or 0.0)
        while time.monotonic() < deadline:
            if not _is_server_running(host, port, logger=logger):
                _remove_state(None, logger=logger)
                _remove_pid_file(None, logger=logger)
                logger.info("API server stopped.")
                return
            time.sleep(0.2)

        logger.warning("Timed out waiting for API shutdown (pid=%s).", pid_int)
        return

    if args.subcommand == "start":
        api_key = (os.environ.get("ISPEC_API_KEY") or "").strip()
        if not _is_local_bind_host(args.host) and not api_key:
            logger.error(
                "Refusing to start API bound to %s without ISPEC_API_KEY; "
                "set ISPEC_API_KEY or use --host 127.0.0.1 for local-only dev.",
                args.host,
            )
            raise SystemExit(2)

        from ispec.api.main import app
        import uvicorn

        logger.info("Starting API server at %s:%s", args.host, args.port)
        state_path = _write_state(args.host, args.port, logger=logger)
        pid_path = _write_pid_file(logger=logger)
        try:
            uvicorn.run(app, host=args.host, port=args.port)
        finally:
            _remove_state(state_path, logger=logger)
            _remove_pid_file(pid_path, logger=logger)
        return

    logger.error(f"No handler for subcommand: {args.subcommand}")
