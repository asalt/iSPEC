"""Command-line helpers for configuring iSPEC logging.

This module exposes functions to register logging-related subcommands on an
``argparse`` parser and to dispatch the parsed arguments to their respective
handlers.
"""

import logging

from ispec.logging import get_logger, reset_logger
from ispec.logging.logging import _resolve_log_file


def register_subcommands(subparsers):
    """Register logging subcommands on the provided ``argparse`` object.

    Parameters
    ----------
    subparsers : :class:`argparse._SubParsersAction`
        The ``argparse`` subparsers object to which logging commands are added.
    """

    set_level_parser = subparsers.add_parser(
        "set-level", help="Set the logging level"
    )
    set_level_parser.add_argument(
        "level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level to use",
    )

    subparsers.add_parser("show-path", help="Show the log file location")


def dispatch(args):
    """Execute the logging command associated with ``args.subcommand``."""

    if args.subcommand == "set-level":
        level = getattr(logging, args.level.upper())
        reset_logger()
        get_logger(level=level)
    elif args.subcommand == "show-path":
        path = _resolve_log_file().resolve()
        print(path)
    else:
        get_logger(__file__).error("No handler for subcommand: %s", args.subcommand)
