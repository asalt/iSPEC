# ispec/cli/main.py
import argparse
from ispec.cli import db
from ispec.logging import get_logger

def main():

    parser = argparse.ArgumentParser(prog="ispec", description="iSPEC CLI toolkit")
    subparsers = parser.add_subparsers(dest="command", required=True)

    db_parser = subparsers.add_parser("db", help="Database operations")
    db_subparsers = db_parser.add_subparsers(dest="subcommand", required=True)
    db.register_subcommands(db_subparsers)

    # se_parser = subparsers.add_parser("somethingelse", help="Other stuff")
    # se_subparsers = se_parser.add_subparsers(dest="subcommand", required=True)
    # somethingelse.register_subcommands(se_subparsers)

    args = parser.parse_args()

    # can expand this later, can use a hashmap/dict lookup if becomes larger
    if args.command == "db":
        db.dispatch(args)
    elif args.command == "somethingelse":
        somethingelse.dispatch(args)
