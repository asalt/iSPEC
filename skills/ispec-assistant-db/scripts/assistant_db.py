from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import quote


def _default_db_path() -> Path:
    ispec_dir = Path(__file__).resolve().parents[3]
    return ispec_dir / "data" / "ispec-assistant.db"


def _resolve_db_path(raw: str | None) -> Path:
    candidate = (raw or "").strip()
    if not candidate:
        candidate = (os.getenv("ISPEC_ASSISTANT_DB_PATH") or "").strip()
    if not candidate:
        return _default_db_path()

    if candidate.startswith("sqlite:///"):
        candidate = candidate.removeprefix("sqlite:///")
    elif candidate.startswith("sqlite://"):
        candidate = candidate.removeprefix("sqlite://")
        if candidate.startswith("/"):
            candidate = "/" + candidate.lstrip("/")
    return Path(candidate).expanduser()


def _open_db(path: Path) -> sqlite3.Connection:
    abs_path = path.expanduser().resolve()
    uri = f"file:{quote(str(abs_path), safe='/')}?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def _print_table(rows: Iterable[sqlite3.Row], columns: list[str]) -> None:
    items = list(rows)
    if not items:
        print("(no rows)")
        return

    widths = {col: len(col) for col in columns}
    for row in items:
        for col in columns:
            value = row[col]
            text = "" if value is None else str(value)
            widths[col] = max(widths[col], len(text))

    header = "  ".join(col.ljust(widths[col]) for col in columns)
    print(header)
    print("  ".join("-" * widths[col] for col in columns))
    for row in items:
        line_parts: list[str] = []
        for col in columns:
            value = row[col]
            text = "" if value is None else str(value)
            line_parts.append(text.ljust(widths[col]))
        print("  ".join(line_parts))


def cmd_sessions(connection: sqlite3.Connection, args: argparse.Namespace) -> int:
    limit = max(1, int(args.limit))
    query = (
        "select id, session_id, user_id, created_at, updated_at, state_json "
        "from support_session order by id desc limit ?"
    )
    rows = connection.execute(query, (limit,)).fetchall()
    if args.json:
        payload = [dict(row) for row in rows]
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0
    _print_table(rows, ["id", "session_id", "user_id", "created_at", "updated_at"])
    return 0


def _resolve_session_pk(connection: sqlite3.Connection, session_id: str) -> int | None:
    row = connection.execute(
        "select id from support_session where session_id = ?",
        (session_id,),
    ).fetchone()
    if row is None:
        return None
    return int(row["id"])


def cmd_messages(connection: sqlite3.Connection, args: argparse.Namespace) -> int:
    session_pk: int | None = None
    if args.session_pk is not None:
        session_pk = int(args.session_pk)
    elif args.session_id:
        session_pk = _resolve_session_pk(connection, args.session_id)

    if session_pk is None:
        print("Session not found. Provide --session-id or --session-pk.", file=sys.stderr)
        return 2

    limit = max(1, int(args.limit))
    rows = connection.execute(
        (
            "select id, session_pk, role, provider, model, created_at, content "
            "from support_message where session_pk = ? order by id desc limit ?"
        ),
        (session_pk, limit),
    ).fetchall()
    rows.reverse()

    payload: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        if not args.full:
            content = str(item.get("content") or "")
            item["content"] = content if len(content) <= 200 else content[:200] + "…"
        payload.append(item)

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    for item in payload:
        stamp = item.get("created_at") or ""
        role = item.get("role") or ""
        provider = item.get("provider") or ""
        model = item.get("model") or ""
        content = item.get("content") or ""
        print(f"[{stamp}] {role} ({provider} {model}): {content}")
    return 0


def _iter_sessions(connection: sqlite3.Connection, session_id: str | None) -> list[sqlite3.Row]:
    if session_id:
        rows = connection.execute(
            "select id, session_id, user_id, created_at, updated_at, state_json "
            "from support_session where session_id = ?",
            (session_id,),
        ).fetchall()
        return rows
    rows = connection.execute(
        "select id, session_id, user_id, created_at, updated_at, state_json "
        "from support_session order by id"
    ).fetchall()
    return rows


def _session_messages(connection: sqlite3.Connection, session_pk: int) -> list[dict[str, str]]:
    rows = connection.execute(
        (
            "select role, content from support_message "
            "where session_pk = ? and role in ('user','assistant','system') order by id"
        ),
        (session_pk,),
    ).fetchall()
    messages: list[dict[str, str]] = []
    for row in rows:
        role = str(row["role"] or "").strip()
        content = str(row["content"] or "")
        if not role or not content:
            continue
        messages.append({"role": role, "content": content})
    return messages


def cmd_export_jsonl(connection: sqlite3.Connection, args: argparse.Namespace) -> int:
    sessions = _iter_sessions(connection, args.session_id)
    out_path = (args.out or "").strip()
    out_fh = open(out_path, "w", encoding="utf-8") if out_path else sys.stdout
    try:
        for session in sessions:
            session_pk = int(session["id"])
            messages = _session_messages(connection, session_pk)
            record = {
                "session_id": session["session_id"],
                "user_id": session["user_id"],
                "created_at": session["created_at"],
                "updated_at": session["updated_at"],
                "state_json": session["state_json"],
                "messages": messages,
            }
            out_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    finally:
        if out_fh is not sys.stdout:
            out_fh.close()
    return 0


def _safe_json_load(raw: Any) -> Any:
    if not raw:
        return None
    try:
        return json.loads(raw)
    except Exception:
        return None


def cmd_export_feedback_jsonl(connection: sqlite3.Connection, args: argparse.Namespace) -> int:
    out_path = (args.out or "").strip()
    out_fh = open(out_path, "w", encoding="utf-8") if out_path else sys.stdout
    try:
        rows = connection.execute(
            (
                "select id, session_pk, role, content, created_at, provider, model, "
                "feedback, feedback_at, feedback_note, feedback_meta_json, meta_json "
                "from support_message "
                "where role = 'assistant' and feedback is not null order by id"
            )
        ).fetchall()

        for row in rows:
            session_pk = int(row["session_pk"])
            prior = connection.execute(
                (
                    "select role, content from support_message "
                    "where session_pk = ? and id < ? and role in ('user','assistant','system') "
                    "order by id"
                ),
                (session_pk, int(row["id"])),
            ).fetchall()
            prompt_messages = [
                {"role": str(item["role"] or ""), "content": str(item["content"] or "")}
                for item in prior
                if item["role"] and item["content"]
            ]

            record = {
                "assistant_message_id": int(row["id"]),
                "session_pk": session_pk,
                "rating": int(row["feedback"]),
                "rated_at": row["feedback_at"],
                "comment": row["feedback_note"],
                "provider": row["provider"],
                "model": row["model"],
                "prompt": prompt_messages,
                "response": row["content"],
                "assistant_meta": _safe_json_load(row["meta_json"]),
                "feedback_meta": _safe_json_load(row["feedback_meta_json"]),
            }
            out_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    finally:
        if out_fh is not sys.stdout:
            out_fh.close()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="assistant_db", add_help=True)
    parser.add_argument(
        "--db",
        default=None,
        help="Assistant DB file path (or sqlite:/// URI). Defaults to ISPEC_ASSISTANT_DB_PATH or iSPEC/data/ispec-assistant.db.",
    )

    sub = parser.add_subparsers(dest="cmd", required=True)

    sessions = sub.add_parser("sessions", help="List recent support sessions.")
    sessions.add_argument("--limit", type=int, default=20)
    sessions.add_argument("--json", action="store_true")

    messages = sub.add_parser("messages", help="List messages for a session.")
    messages_group = messages.add_mutually_exclusive_group(required=True)
    messages_group.add_argument("--session-id", dest="session_id")
    messages_group.add_argument("--session-pk", dest="session_pk", type=int)
    messages.add_argument("--limit", type=int, default=50)
    messages.add_argument("--full", action="store_true", help="Print full message bodies.")
    messages.add_argument("--json", action="store_true")

    export_jsonl = sub.add_parser("export-jsonl", help="Export conversations as JSONL.")
    export_jsonl.add_argument("--out", default="", help="Output file path (default stdout).")
    export_jsonl.add_argument("--session-id", default="", help="Export a single session.")

    export_feedback = sub.add_parser(
        "export-feedback-jsonl",
        help="Export rated assistant messages as JSONL (for training workflows).",
    )
    export_feedback.add_argument("--out", default="", help="Output file path (default stdout).")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    db_path = _resolve_db_path(args.db)
    if not db_path.exists():
        print(f"DB file not found: {db_path}", file=sys.stderr)
        return 2

    with _open_db(db_path) as connection:
        if args.cmd == "sessions":
            return cmd_sessions(connection, args)
        if args.cmd == "messages":
            return cmd_messages(connection, args)
        if args.cmd == "export-jsonl":
            session_id = (args.session_id or "").strip() or None
            args.session_id = session_id
            return cmd_export_jsonl(connection, args)
        if args.cmd == "export-feedback-jsonl":
            return cmd_export_feedback_jsonl(connection, args)

    print(f"Unknown command: {args.cmd}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
