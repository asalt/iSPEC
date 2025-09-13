"""Helpers for talking to the backend API."""

from __future__ import annotations

from typing import Any, Dict

import requests


def put_response(url: str, data: Dict[str, Any]) -> None:
    """Send ``data`` to ``url`` using an HTTP PUT request."""

    requests.put(url, json=data, timeout=5)
