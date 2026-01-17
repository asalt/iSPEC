# ispec/cli/main.py
import argparse
import sys

from ispec.cli.env import extract_env_files, load_env_files


def main():

    env_files, argv = extract_env_files(sys.argv[1:])
    if env_files:
        load_env_files(env_files, override=True)

    from ispec.cli import (
        agent,
        api,
        auth,
        config as config_cli,
        db,
        logging as logging_cli,
        slack,
        supervisor,
    )

    parser = argparse.ArgumentParser(prog="ispec", description="iSPEC CLI toolkit")
    parser.add_argument(
        "--env-file",
        action="append",
        default=[],
        help=(
            "Load environment variables from a KEY=value file before running the command. "
            "Repeatable; later files override earlier ones. Can be placed anywhere in the command."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    db_parser = subparsers.add_parser("db", help="Database operations")
    db_subparsers = db_parser.add_subparsers(dest="subcommand", required=True)
    db.register_subcommands(db_subparsers)

    api_parser = subparsers.add_parser("api", help="api control")
    api_subparsers = api_parser.add_subparsers(dest="subcommand", required=True)
    api.register_subcommands(api_subparsers)

    auth_parser = subparsers.add_parser("auth", help="Authentication/user helpers")
    auth_subparsers = auth_parser.add_subparsers(dest="subcommand", required=True)
    auth.register_subcommands(auth_subparsers)

    logging_parser = subparsers.add_parser("logging", help="Logging utilities")
    logging_subparsers = logging_parser.add_subparsers(
        dest="subcommand", required=True
    )
    logging_cli.register_subcommands(logging_subparsers)

    agent_parser = subparsers.add_parser("agent", help="Local agent helpers")
    agent_subparsers = agent_parser.add_subparsers(dest="subcommand", required=True)
    agent.register_subcommands(agent_subparsers)

    slack_parser = subparsers.add_parser("slack", help="Slack bot helpers")
    slack_subparsers = slack_parser.add_subparsers(dest="subcommand", required=True)
    slack.register_subcommands(slack_subparsers)

    config_parser = subparsers.add_parser(
        "config", help="Config/env auditing and initialization helpers"
    )
    config_subparsers = config_parser.add_subparsers(dest="subcommand", required=True)
    config_cli.register_subcommands(config_subparsers)

    supervisor_parser = subparsers.add_parser("supervisor", help="Supervisor loop helpers")
    supervisor_subparsers = supervisor_parser.add_subparsers(dest="subcommand", required=True)
    supervisor.register_subcommands(supervisor_subparsers)

    args = parser.parse_args(argv)

    # can expand this later, can use a hashmap/dict lookup if becomes larger
    if args.command == "db":
        db.dispatch(args)
    elif args.command == "api":
        api.dispatch(args)
    elif args.command == "auth":
        auth.dispatch(args)
    elif args.command == "logging":
        logging_cli.dispatch(args)
    elif args.command == "agent":
        agent.dispatch(args)
    elif args.command == "slack":
        slack.dispatch(args)
    elif args.command == "config":
        config_cli.dispatch(args)
    elif args.command == "supervisor":
        supervisor.dispatch(args)
