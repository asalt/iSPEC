from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import secrets

from fastapi import Header, HTTPException


@dataclass(frozen=True, kw_only=True)
class CollectorIdentity:
    collector_id: str
    instrument_ids: frozenset[str]


def require_collector(
    authorization: str | None = Header(default=None),
) -> CollectorIdentity:
    """Dedicated ingest-only credentials. Never accept the general iSPEC API key."""
    location = os.getenv("ISPEC_INSTRUMENT_COLLECTORS_FILE", "").strip()
    if not location:
        raise HTTPException(503, "Instrument ingestion is not configured.")
    try:
        path = Path(location).expanduser()
        if not path.is_absolute():
            raise ValueError("absolute config required")
        with path.open("rb") as stream:
            raw = stream.read(65537)
        if len(raw) > 65536:
            raise ValueError("large config")
        config = json.loads(raw)
        if set(config) != {"collectors"} or not isinstance(config["collectors"], dict):
            raise ValueError("invalid config")
        identities = []
        for collector_id, entry in config["collectors"].items():
            if not re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", collector_id):
                raise ValueError("invalid collector")
            if set(entry) != {"token_env", "instrument_ids"}:
                raise ValueError("invalid entry")
            token_env = entry["token_env"]
            if not isinstance(token_env, str) or not token_env.startswith(
                "ISPEC_INSTRUMENT_TOKEN_"
            ):
                raise ValueError("invalid token variable")
            token = os.getenv(token_env, "")
            if len(token) < 32 or token == os.getenv("ISPEC_API_KEY"):
                raise ValueError("dedicated credential missing")
            ids = entry["instrument_ids"]
            if (
                not isinstance(ids, list)
                or not ids
                or not all(
                    isinstance(x, str) and re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", x)
                    for x in ids
                )
            ):
                raise ValueError("invalid instruments")
            identities.append(
                (
                    token,
                    CollectorIdentity(
                        collector_id=collector_id, instrument_ids=frozenset(ids)
                    ),
                )
            )
        if len({token for token, _ in identities}) != len(identities):
            raise ValueError("shared credentials are ambiguous")
    except (OSError, ValueError, TypeError, KeyError):
        raise HTTPException(
            503,
            "Instrument collector configuration is invalid or credentials are unavailable.",
        ) from None
    token = (
        authorization[7:]
        if authorization and authorization.startswith("Bearer ")
        else ""
    )
    for expected, identity in identities:
        if secrets.compare_digest(token.encode(), expected.encode()):
            return identity
    raise HTTPException(401, "Missing or invalid collector credential.")
