"""Helpers for talking to the backend API."""

from __future__ import annotations

from typing import Any, Dict

import requests

from ispec.logging import get_logger

logger = get_logger(__file__)


def put_response(url: str, data: Dict[str, Any]) -> None:
    """Send ``data`` to ``url`` using an HTTP PUT request."""

    try:
        response = requests.put(url, json=data, timeout=5)
        response.raise_for_status()
    except requests.RequestException as exc:
        logger.error("Failed to PUT data to %s: %s", url, exc)
        raise
