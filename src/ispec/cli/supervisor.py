"""Command-line entry points for the internal supervisor loop."""

from __future__ import annotations

import os

from ispec.logging import get_logger
from ispec.supervisor.loop import SupervisorConfig, _default_agent_id, run_supervisor


def register_subcommands(subparsers) -> None:
    run_parser = subparsers.add_parser("run", help="Run the internal supervisor loop")
    run_parser.add_argument(
        "--agent-id",
        default=None,
        help="Agent identifier for emitted events (default: hostname)",
    )
    run_parser.add_argument(
        "--backend-base-url",
        default=None,
        help="Base URL for iSPEC backend (default: http://127.0.0.1:${ISPEC_PORT or 3001})",
    )
    run_parser.add_argument(
        "--frontend-url",
        default="http://127.0.0.1:3000/",
        help="Frontend URL to probe (default: http://127.0.0.1:3000/)",
    )
    run_parser.add_argument(
        "--interval-seconds",
        type=int,
        default=10,
        help="Seconds between steps (default: 10)",
    )
    run_parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=2.0,
        help="HTTP timeout seconds (default: 2.0)",
    )
    run_parser.add_argument("--once", action="store_true", help="Run one step then exit")


def dispatch(args) -> None:
    if args.subcommand != "run":
        raise SystemExit(f"Unknown supervisor subcommand: {args.subcommand}")

    logger = get_logger(__file__)

    agent_id = (args.agent_id or "").strip() or _default_agent_id()

    backend_base_url = (args.backend_base_url or "").strip()
    if not backend_base_url:
        port = (os.getenv("ISPEC_PORT") or "").strip() or "3001"
        backend_base_url = f"http://127.0.0.1:{port}"

    config = SupervisorConfig(
        agent_id=agent_id,
        backend_base_url=backend_base_url,
        frontend_url=str(args.frontend_url),
        interval_seconds=int(args.interval_seconds),
        timeout_seconds=float(args.timeout_seconds),
    )
    run_id = run_supervisor(config, once=bool(args.once))
    logger.info("Supervisor run_id=%s", run_id)

