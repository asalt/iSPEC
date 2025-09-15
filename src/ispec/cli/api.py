"""Command-line helpers for controlling the iSPEC API service.

This module exposes functions to register API-related subcommands on an
``argparse`` parser and to dispatch the parsed arguments to their respective
handlers.
"""

from ispec.logging import get_logger


def register_subcommands(subparsers):
    """Register API subcommands on the provided ``argparse`` object.

    Parameters
    ----------
    subparsers : :class:`argparse._SubParsersAction`
        The ``argparse`` subparsers object to which API commands are added.

    Examples
    --------
    >>> import argparse
    >>> parser = argparse.ArgumentParser(prog="ispec api")
    >>> subparsers = parser.add_subparsers(dest="subcommand", required=True)
    >>> register_subcommands(subparsers)
    >>> parser.parse_args(["start", "--host", "0.0.0.0", "--port", "9000"])
    Namespace(subcommand='start', host='0.0.0.0', port=9000)

    """

    _ = subparsers.add_parser("status", help="Check api status")
    starter_parser = subparsers.add_parser("start", help="start the API server")
    starter_parser.add_argument(
        "--host", default="localhost", help="Host to run the API server "
    )
    starter_parser.add_argument(
        "--port", type=int, default=8000, help="Port to run the API server on"
    )


def dispatch(args):
    """Execute the API command associated with ``args.subcommand``.

    Parameters
    ----------
    args : :class:`argparse.Namespace`
        Parsed arguments containing a ``subcommand`` attribute and any
        additional options required by that subcommand.

    Examples
    --------
    >>> import types
    >>> args = types.SimpleNamespace(subcommand="status")
    >>> dispatch(args)  # doctest: +SKIP

    When starting the API server::

        >>> args = types.SimpleNamespace(subcommand="start", host="127.0.0.1", port=8000)
        >>> dispatch(args)  # doctest: +SKIP

    """

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
