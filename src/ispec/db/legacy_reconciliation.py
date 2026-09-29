"""Field-level reconciliation; timestamps are bookkeeping, not edit ownership."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Mapping


def _value(value: Any) -> Any:
    if isinstance(value, datetime):
        if value.tzinfo is not None:
            value = value.astimezone(UTC).replace(tzinfo=None)
        return value.isoformat()
    return value


def _load(raw: str | None, *, source: str) -> dict[str, Any]:
    try:
        payload = json.loads(raw) if raw else None
    except (TypeError, ValueError):
        return {}
    if (
        not isinstance(payload, dict)
        or payload.get("version") != 1
        or payload.get("source") != source
        or not isinstance(payload.get("values"), dict)
    ):
        return {}
    return payload["values"]


def _dump(values: Mapping[str, Any], *, source: str) -> str:
    return json.dumps(
        {"version": 1, "source": source, "values": values},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


@dataclass(frozen=True, kw_only=True)
class SourceReconciliation:
    updates: dict[str, Any]
    baseline_json: str
    conflicts_json: str
    conflict_fields: tuple[str, ...]


def reconcile_source_fields(
    *,
    current: Mapping[str, Any],
    incoming: Mapping[str, Any],
    baseline_json: str | None,
    conflicts_json: str | None,
    source: str,
    backfill_missing: bool = False,
) -> SourceReconciliation:
    """Reconcile only observed fields and retain unresolved conflict evidence."""
    baseline = dict(_load(baseline_json, source=source))
    conflicts = dict(_load(conflicts_json, source=source))
    updates: dict[str, Any] = {}

    for field, incoming_value in incoming.items():
        local = _value(current.get(field))
        remote = _value(incoming_value)
        known = field in baseline
        previous = baseline.get(field)
        existing_conflict = conflicts.get(field)

        if local == remote:
            conflicts.pop(field, None)
        elif known and local == previous:
            updates[field] = incoming_value
            conflicts.pop(field, None)
        elif not known and backfill_missing and local in (None, ""):
            updates[field] = incoming_value
            conflicts.pop(field, None)
        elif known and remote == previous and not existing_conflict:
            pass  # Only the local value changed.
        else:
            # Advancing the source baseline must not erase an unresolved conflict
            # when an identical source payload arrives again on the next sync.
            evidence = (
                dict(existing_conflict)
                if isinstance(existing_conflict, dict)
                else {
                    "reason": "divergent_changes" if known else "unknown_baseline",
                    "previous_source_known": known,
                    "previous_source": previous,
                }
            )
            evidence.update(local=local, incoming_source=remote)
            conflicts[field] = evidence
        baseline[field] = remote

    return SourceReconciliation(
        updates=updates,
        baseline_json=_dump(baseline, source=source),
        conflicts_json=_dump(conflicts, source=source),
        conflict_fields=tuple(sorted(conflicts)),
    )
