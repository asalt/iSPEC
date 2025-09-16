"""Command-line helpers for interacting with the iSPEC database.

This module provides utilities to register database-related subcommands with an
``argparse`` parser and to dispatch parsed arguments to the appropriate
database operations.
"""

from collections.abc import Mapping, Sequence
from typing import Any

from rich.console import Console
from rich.table import Table

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

    Exporting data::

    >>> parser.parse_args(["export", "--table-name", "person", "--file", "out.json"])
    Namespace(subcommand='export', table_name='person', file='out.json')

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

    export_parser = subparsers.add_parser("export", help="Export table to CSV or JSON")
    export_parser.add_argument("--table-name", required=True, choices=("person", "project"))
    export_parser.add_argument("--file", required=True, help="Output file (CSV or JSON)")


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

    Exporting data::

        >>> args = types.SimpleNamespace(subcommand="export", table_name="person", file="out.json")
        >>> dispatch(args)  # doctest: +SKIP

    """

    logger = get_logger(__file__)

    if args.subcommand == "status":
        operations.check_status()
    elif args.subcommand == "show":
        table_definitions = operations.show_tables()
        _render_table_overview(table_definitions)
    elif args.subcommand == "import":
        operations.import_file(args.file, args.table_name)
    elif args.subcommand == "export":
        operations.export_table(args.table_name, args.file)
    elif args.subcommand == "init":
        operations.initialize(file_path=args.file)
    else:
        logger.info("no dispatched function provided for %s", args.subcommand)


def _render_table_overview(
    table_definitions: Mapping[str, Sequence[Mapping[str, Any]]],
    console: Console | None = None,
) -> None:
    """Pretty-print table metadata using ``rich``."""

    if console is None:
        console = Console()

    table = Table(title="iSPEC Database Schema", show_lines=True)
    table.add_column("Table", style="bold cyan")
    table.add_column("Column", style="magenta")
    table.add_column("Type", style="green")
    table.add_column("Nullable", justify="center", style="yellow")
    table.add_column("Default", style="bright_black")

    table_names = sorted(table_definitions)
    if not table_names:
        table.add_row("[dim]No tables found[/dim]", "", "", "", "")
        console.print(table)
        return

    for table_index, table_name in enumerate(table_names):
        columns = table_definitions[table_name]
        if not columns:
            table.add_row(table_name, "[dim]-[/dim]", "[dim]-[/dim]", "", "")
        else:
            for column_index, column in enumerate(columns):
                default = column.get("default")
                default_display = "" if default in (None, "") else str(default)
                table.add_row(
                    table_name if column_index == 0 else "",
                    str(column.get("name", "")),
                    str(column.get("type", "")),
                    "Yes" if column.get("nullable", True) else "No",
                    default_display,
                )
        if table_index < len(table_names) - 1:
            table.add_section()

    console.print(table)
