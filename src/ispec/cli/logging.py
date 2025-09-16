"""Command-line helpers for configuring iSPEC logging.

This module exposes functions to register logging-related subcommands on an
``argparse`` parser and to dispatch the parsed arguments to their respective
handlers.
"""

import logging

from ispec.logging import get_logger, reset_logger
from ispec.logging.logging import get_configured_level, _resolve_log_file
from ispec.logging.config import save_log_level



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
    subparsers.add_parser(
        "show-level", help="Show the configured logging level"
    )


def dispatch(args):
    """Execute the logging command associated with ``args.subcommand``."""

    if args.subcommand == "set-level":
        level_name = args.level.upper()
        level = getattr(logging, level_name)
        save_log_level(level_name)
        reset_logger()
        get_logger(level=level)
    elif args.subcommand == "show-path":
        path = _resolve_log_file().resolve()
        print(path)
    elif args.subcommand == "show-level":
        print(get_configured_level())
    else:
        get_logger(__file__).error("No handler for subcommand: %s", args.subcommand)
