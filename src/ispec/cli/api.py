# ispec/cli/api.py
from ispec.logging import get_logger


def register_subcommands(subparsers):
    subparsers.add_parser("status", help="Check api status")
    starter_parser = subparsers.add_parser("start", help="start the API server")
    starter_parser.add_argument("--host", default="localhost", help="Host to run the API server ")
    starter_parser.add_argument("--port", type=int, default=8000, help="Port to run the API server on")


def dispatch(args):
    """Dispatch API CLI subcommands using a simple lookup table.

    Errors from handlers are allowed to propagate so callers can see the
    underlying exception. Unknown subcommands raise ``ValueError`` with a clear
    message.
    """
    logger = get_logger(__file__)

    def _status() -> None:
        logger.info("run ispec api start to start the API server")

    def _start() -> None:
        from ispec.api.main import app
        import uvicorn

        logger.info(f"Starting API server at {args.host}:{args.port}")
        uvicorn.run(app, host=args.host, port=args.port)

    commands = {"status": _status, "start": _start}
    try:
        handler = commands[args.subcommand]
    except KeyError as exc:
        message = f"No handler for API subcommand: {args.subcommand}"
        logger.error(message)
        raise ValueError(message) from exc

    handler()
