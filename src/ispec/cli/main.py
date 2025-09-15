# ispec/cli/main.py
import argparse
from ispec.cli import api, db, logging as logging_cli


def main():

    parser = argparse.ArgumentParser(prog="ispec", description="iSPEC CLI toolkit")
    subparsers = parser.add_subparsers(dest="command", required=True)

    db_parser = subparsers.add_parser("db", help="Database operations")
    db_subparsers = db_parser.add_subparsers(dest="subcommand", required=True)
    db.register_subcommands(db_subparsers)

    api_parser = subparsers.add_parser("api", help="api control")
    api_subparsers = api_parser.add_subparsers(dest="subcommand", required=True)
    api.register_subcommands(api_subparsers)

    logging_parser = subparsers.add_parser("logging", help="Logging utilities")
    logging_subparsers = logging_parser.add_subparsers(
        dest="subcommand", required=True
    )
    logging_cli.register_subcommands(logging_subparsers)


    args = parser.parse_args()

    # can expand this later, can use a hashmap/dict lookup if becomes larger
    if args.command == "db":
        db.dispatch(args)
    elif args.command == "api":
        api.dispatch(args)
    elif args.command == "logging":
        logging_cli.dispatch(args)
