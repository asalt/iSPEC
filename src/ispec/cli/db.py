from ispec.logging import get_logger


def register_subcommands(subparsers):
    init_parser = subparsers.add_parser("init", help="initialize db")
    init_parser.add_argument("--file", required=False)

    subparsers.add_parser("status", help="Check DB status")
    subparsers.add_parser("show", help="Show tables")

    import_parser = subparsers.add_parser("import", help="Import file")
    import_parser.add_argument("--table-name", required=True, choices=("person", "project"))
    import_parser.add_argument("--file", required=True)


def dispatch(args):
    """Dispatch database CLI subcommands with minimal error handling."""
    logger = get_logger(__file__)

    import ispec.db.operations as operations

    commands = {
        "status": lambda: operations.check_status(),
        "show": lambda: operations.show_tables(),
        "import": lambda: operations.import_file(args.file),
        "init": lambda: operations.initialize(file_path=args.file),
    }

    try:
        handler = commands[args.subcommand]
    except KeyError as exc:
        message = f"No handler for DB subcommand: {args.subcommand}"
        logger.error(message)
        raise ValueError(message) from exc

    handler()
