"""Command-line helpers for interacting with the iSPEC database.

This module provides utilities to register database-related subcommands with an
``argparse`` parser and to dispatch parsed arguments to the appropriate
database operations.
"""

from ispec.db import operations
from ispec.logging import get_logger


def register_subcommands(subparsers):
    """Attach database subcommands to an ``argparse`` parser.

    Parameters
    ----------
    subparsers : :class:`argparse._SubParsersAction`
        The ``argparse`` subparsers object to which database commands will be
        registered.

    Examples
    --------
    >>> import argparse
    >>> parser = argparse.ArgumentParser(prog="ispec db")
    >>> subparsers = parser.add_subparsers(dest="subcommand", required=True)
    >>> register_subcommands(subparsers)
    >>> parser.parse_args(["init", "--file", "test.db"])
    Namespace(subcommand='init', file='test.db')

    """

    init_parser = subparsers.add_parser("init", help="initialize db")
    init_parser.add_argument("--file", required=False)

    _ = subparsers.add_parser("status", help="Check DB status")
    _ = subparsers.add_parser("show", help="Show tables")

    import_parser = subparsers.add_parser("import", help="Import file")
    import_parser.add_argument(
        "--table-name",
        required=True,
        choices=("person", "project", "comment", "letter"),
    )
    import_parser.add_argument("--file", required=True)


def dispatch(args):
    """Run the database operation associated with ``args.subcommand``.

    Parameters
    ----------
    args : :class:`argparse.Namespace`
        Parsed arguments containing the targeted ``subcommand`` and any
        additional options required by that subcommand.

    Examples
    --------
    >>> import types
    >>> args = types.SimpleNamespace(subcommand="status")
    >>> dispatch(args)  # doctest: +SKIP

    Importing data::

        >>> args = types.SimpleNamespace(subcommand="import", file="data.csv", table_name="person")
        >>> dispatch(args)  # doctest: +SKIP

    """

    logger = get_logger(__file__)

    if args.subcommand == "status":
        operations.check_status()
    elif args.subcommand == "show":
        operations.show_tables()
    elif args.subcommand == "import":
        operations.import_file(args.file, args.table_name)
    elif args.subcommand == "init":
        operations.initialize(file_path=args.file)
    else:
        logger.info("no dispatched function provided for %s", args.subcommand)
