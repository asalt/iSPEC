from ispec.db import operations
from ispec.logging import get_logger

def register_subcommands(subparsers):
    init_parser = subparsers.add_parser("init", help="initialize db")
    init_parser.add_argument("--file", required=False)

    _ = subparsers.add_parser("status", help="Check DB status")
    _ = subparsers.add_parser("show", help="Show tables")
    import_parser = subparsers.add_parser("import", help="Import file")
    import_parser.add_argument("--file", required=True)

def dispatch(args):
    
    logger = get_logger(__file__)
    # funcs = {
    #     "init" : operations.import_file,
    #     "status" : operations.check_status,
    #     "show" : operations.show_tables,
    #     "import" : operations.import_file,
    # }
    # subcommand = args.subcommand
    # func = funcs.get(subcommand)
    # if funcs is None:
    #     raise ValueError(f"subcommand {subcommand} is not configured")

    # todo figure out how to pass the args
    # unpack all the arguments?
    # func(args)


    if args.subcommand == "status":
        operations.check_status()
    elif args.subcommand == "show":
        operations.show_tables()
    elif args.subcommand == "import":
        operations.import_file(args.file)
    elif args.subcommand == "init":
        operations.initialize(file_path=args.file)
    else:
        logger.info("no dispatched function provided for %s", args.subcommand)
