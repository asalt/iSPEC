from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
import sqlite3

from .bindings import discover_prompt_bindings_ast
from .connect import connect_prompts_db, ensure_prompts_schema
from .loader import resolve_prompt_root
from .models import PromptSource
from .parser import parse_prompt_file


@dataclass
class PromptSyncSummary:
    new_families: list[str] = field(default_factory=list)
    new_versions: list[tuple[str, int]] = field(default_factory=list)
    metadata_updates: list[str] = field(default_factory=list)
    binding_updates: int = 0
    missing_families: list[str] = field(default_factory=list)
    stale_bindings_removed: int = 0
    check_failed: bool = False


def _utcnow_iso() -> str:
    return datetime.now(UTC).isoformat()


def load_prompt_sources(*, prompt_root: str | Path | None = None) -> list[PromptSource]:
    root = Path(prompt_root).expanduser().resolve() if prompt_root is not None else resolve_prompt_root()
    sources = [parse_prompt_file(path) for path in sorted(root.glob("*.md"))]
    return sorted(sources, key=lambda item: item.family)


def _upsert_prompt_source(conn: sqlite3.Connection, source: PromptSource, *, check: bool, summary: PromptSyncSummary) -> int:
    now = _utcnow_iso()
    family_row = conn.execute(
        "SELECT id, title, notes, source_path FROM prompt_family WHERE family = ?",
        (source.family,),
    ).fetchone()
    if family_row is None:
        summary.new_families.append(source.family)
        if check:
            summary.check_failed = True
            return -1
        conn.execute(
            """
            INSERT INTO prompt_family (family, source_path, title, notes, active_version_id, created_at, updated_at)
            VALUES (?, ?, ?, ?, NULL, ?, ?)
            """,
            (source.family, source.source_path, source.title, source.notes, now, now),
        )
        family_id = int(conn.execute("SELECT id FROM prompt_family WHERE family = ?", (source.family,)).fetchone()["id"])
    else:
        family_id = int(family_row["id"])
        if (
            str(family_row["source_path"] or "") != source.source_path
            or (family_row["title"] != source.title)
            or (family_row["notes"] != source.notes)
        ):
            summary.metadata_updates.append(source.family)
            if check:
                summary.check_failed = True
            else:
                conn.execute(
                    """
                    UPDATE prompt_family
                    SET source_path = ?, title = ?, notes = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (source.source_path, source.title, source.notes, now, family_id),
                )

    version_row = conn.execute(
        "SELECT id, version_num FROM prompt_version WHERE family_id = ? AND body_sha256 = ?",
        (family_id, source.body_sha256),
    ).fetchone()
    if version_row is None:
        next_version = conn.execute(
            "SELECT COALESCE(MAX(version_num), 0) + 1 AS next_version FROM prompt_version WHERE family_id = ?",
            (family_id,),
        ).fetchone()
        version_num = int(next_version["next_version"] if next_version is not None else 1)
        summary.new_versions.append((source.family, version_num))
        if check:
            summary.check_failed = True
            version_id = -1
        else:
            conn.execute(
                """
                INSERT INTO prompt_version (family_id, version_num, body_sha256, body_text, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (family_id, version_num, source.body_sha256, source.body, now),
            )
            version_id = int(
                conn.execute(
                    "SELECT id FROM prompt_version WHERE family_id = ? AND body_sha256 = ?",
                    (family_id, source.body_sha256),
                ).fetchone()["id"]
            )
    else:
        version_id = int(version_row["id"])

    if not check and version_id > 0:
        conn.execute(
            "UPDATE prompt_family SET active_version_id = ?, updated_at = ? WHERE id = ?",
            (version_id, now, family_id),
        )
    return family_id


def _sync_bindings(
    conn: sqlite3.Connection,
    *,
    source_root: Path,
    family_ids: dict[str, int],
    check: bool,
    summary: PromptSyncSummary,
) -> None:
    now = _utcnow_iso()
    discovered = discover_prompt_bindings_ast(source_root=source_root)
    seen_keys: set[tuple[int, str, str]] = set()
    for binding in discovered:
        family_id = family_ids.get(binding.family)
        if family_id is None or family_id <= 0:
            if binding.family not in summary.missing_families:
                summary.missing_families.append(binding.family)
            summary.check_failed = True
            continue
        key = (family_id, binding.module, binding.qualname)
        seen_keys.add(key)
        existing = conn.execute(
            """
            SELECT id, source_file, source_line, binding_kind
            FROM prompt_binding
            WHERE family_id = ? AND module = ? AND qualname = ?
            """,
            key,
        ).fetchone()
        needs_update = (
            existing is None
            or str(existing["source_file"] or "") != str(binding.source_file or "")
            or (existing["source_line"] != binding.source_line)
            or str(existing["binding_kind"] or "") != binding.binding_kind
        )
        if not needs_update:
            continue
        summary.binding_updates += 1
        if check:
            summary.check_failed = True
            continue
        if existing is None:
            conn.execute(
                """
                INSERT INTO prompt_binding
                (family_id, module, qualname, source_file, source_line, binding_kind, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    family_id,
                    binding.module,
                    binding.qualname,
                    binding.source_file or "",
                    binding.source_line,
                    binding.binding_kind,
                    now,
                    now,
                ),
            )
        else:
            conn.execute(
                """
                UPDATE prompt_binding
                SET source_file = ?, source_line = ?, binding_kind = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    binding.source_file or "",
                    binding.source_line,
                    binding.binding_kind,
                    now,
                    int(existing["id"]),
                ),
            )

    existing_rows = conn.execute("SELECT id, family_id, module, qualname FROM prompt_binding").fetchall()
    stale_ids = [
        int(row["id"])
        for row in existing_rows
        if (int(row["family_id"]), str(row["module"]), str(row["qualname"])) not in seen_keys
    ]
    if stale_ids:
        summary.stale_bindings_removed = len(stale_ids)
        if check:
            summary.check_failed = True
        else:
            conn.executemany("DELETE FROM prompt_binding WHERE id = ?", [(item,) for item in stale_ids])


def sync_prompts(
    *,
    prompt_root: str | Path | None = None,
    source_root: str | Path | None = None,
    check: bool = False,
) -> PromptSyncSummary:
    root = Path(prompt_root).expanduser().resolve() if prompt_root is not None else resolve_prompt_root()
    code_root = Path(source_root).expanduser().resolve() if source_root is not None else Path(__file__).resolve().parents[1]
    summary = PromptSyncSummary()
    sources = load_prompt_sources(prompt_root=root)

    with connect_prompts_db() as conn:
        ensure_prompts_schema(conn)
        family_ids: dict[str, int] = {}
        for source in sources:
            family_id = _upsert_prompt_source(conn, source, check=check, summary=summary)
            family_ids[source.family] = family_id
        _sync_bindings(conn, source_root=code_root, family_ids=family_ids, check=check, summary=summary)
        if check and summary.check_failed:
            conn.rollback()
        else:
            conn.commit()
    return summary
