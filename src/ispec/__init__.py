"""Core package for the iSPEC project.

This top-level module exposes convenience helpers such as
the database :func:`get_session` function for interacting with
the project's persistence layer.
"""

from .db import get_session

__all__ = ["get_session"]

