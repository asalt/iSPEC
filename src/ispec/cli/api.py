# ispec/cli/api.py
from ispec.logging import get_logger


def register_subcommands(subparsers):

    status_parser_ = subparsers.add_parser("status", help="Check api status")
    starter_parser = subparsers.add_parser("start", help="start the API server")
    starter_parser.add_argument(
        "--host", default="localhost", help="Host to run the API server "
    )
    starter_parser.add_argument(
        "--port", type=int, default=8000, help="Port to run the API server on"
    )


def dispatch(args):

    logger = get_logger(__file__)

    from ispec.api.main import app
    import uvicorn

    if args.subcommand == "status":
        logger.info("run ispec api start to start the API server")
    elif args.subcommand == "start":
        logger.info(f"Starting API server at {args.host}:{args.port}")
        uvicorn.run(app, host=args.host, port=args.port)
    else:
        logger.error(f"No handler for subcommand: {args.subcommand}")
