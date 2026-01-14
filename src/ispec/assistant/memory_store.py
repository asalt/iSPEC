from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from sqlalchemy.orm import Session

from ispec.assistant.models import SupportMemory


def _load_json(value: str | None) -> Any:
    if not value:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    try:
        return json.loads(raw)
    except Exception:
        return raw


def _dump_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


@dataclass(frozen=True)
class MemoryContext:
    session: list[dict[str, Any]]
    user: list[dict[str, Any]]
    global_: list[dict[str, Any]]


def _rows_to_payload(rows: list[SupportMemory]) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for row in rows:
        payload.append(
            {
                "id": int(row.id),
                "session_pk": int(row.session_pk) if row.session_pk is not None else None,
                "user_id": int(row.user_id),
                "kind": row.kind,
                "key": row.key,
                "value": _load_json(row.value_json),
                "confidence": float(row.confidence),
                "importance": float(row.importance),
                "salience": float(row.salience),
                "updated_at": row.updated_at.isoformat() if row.updated_at else None,
            }
        )
    return payload


def fetch_memory_context(
    db: Session,
    *,
    session_pk: int | None,
    user_id: int | None,
    include_global: bool = True,
    limit_session: int = 20,
    limit_user: int = 20,
    limit_global: int = 20,
    kinds: set[str] | None = None,
) -> MemoryContext:
    """Fetch memory rows for prompt context (session > user > global)."""

    def query_session_scope(*, limit: int) -> list[SupportMemory]:
        if session_pk is None or limit <= 0:
            return []
        q = db.query(SupportMemory).filter(SupportMemory.session_pk == int(session_pk))
        if kinds:
            q = q.filter(SupportMemory.kind.in_(sorted(kinds)))
        return (
            q.order_by(
                SupportMemory.salience.desc(),
                SupportMemory.updated_at.desc(),
                SupportMemory.id.desc(),
            )
            .limit(int(limit))
            .all()
        )

    def query_user_scope(
        *,
        user_id_value: int,
        limit: int,
    ) -> list[SupportMemory]:
        if limit <= 0:
            return []
        q = db.query(SupportMemory)
        q = q.filter(SupportMemory.session_pk.is_(None))
        q = q.filter(SupportMemory.user_id == int(user_id_value))
        if kinds:
            q = q.filter(SupportMemory.kind.in_(sorted(kinds)))
        return (
            q.order_by(
                SupportMemory.salience.desc(),
                SupportMemory.updated_at.desc(),
                SupportMemory.id.desc(),
            )
            .limit(int(limit))
            .all()
        )

    session_rows = query_session_scope(limit=limit_session)

    user_rows: list[SupportMemory] = []
    if user_id is not None and int(user_id) > 0:
        user_rows = query_user_scope(user_id_value=int(user_id), limit=limit_user)

    global_rows: list[SupportMemory] = []
    if include_global:
        global_rows = query_user_scope(user_id_value=0, limit=limit_global)

    return MemoryContext(
        session=_rows_to_payload(session_rows),
        user=_rows_to_payload(user_rows),
        global_=_rows_to_payload(global_rows),
    )


def create_memory(
    db: Session,
    *,
    session_pk: int | None,
    user_id: int,
    kind: str,
    key: str | None,
    value: Any,
    confidence: float | None = None,
    importance: float | None = None,
    salience: float | None = None,
) -> SupportMemory:
    row = SupportMemory(
        session_pk=int(session_pk) if session_pk is not None else None,
        user_id=int(user_id),
        kind=str(kind),
        key=str(key) if key is not None else None,
        value_json=_dump_json(value),
    )
    if confidence is not None:
        row.confidence = float(confidence)
    if importance is not None:
        row.importance = float(importance)
    if salience is not None:
        row.salience = float(salience)
    db.add(row)
    db.flush()
    return row
