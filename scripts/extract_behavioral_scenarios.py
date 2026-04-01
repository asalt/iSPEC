#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
import re
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = ROOT / 'src'
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from ispec.assistant.connect import get_assistant_db_uri, get_assistant_session
from ispec.assistant.models import SupportMessage, SupportSession


def _slugify(text: str) -> str:
    normalized = re.sub(r'[^a-z0-9]+', '-', text.strip().lower())
    normalized = normalized.strip('-')
    return normalized or 'case'


def _json_load(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _assistant_meta_subset(meta_json: str | None) -> dict[str, Any] | None:
    parsed = _json_load(meta_json)
    if not parsed:
        return None
    subset: dict[str, Any] = {}
    provider = parsed.get('provider')
    model = parsed.get('model')
    if isinstance(provider, str) and provider.strip():
        subset['provider'] = provider.strip()
    if isinstance(model, str) and model.strip():
        subset['model'] = model.strip()

    response_contract = parsed.get('response_contract')
    if isinstance(response_contract, dict):
        rc_subset = {
            key: response_contract.get(key)
            for key in (
                'configured_mode',
                'selected_contract',
                'shadow_candidate',
                'would_apply_if_live',
                'protection_reason',
            )
            if response_contract.get(key) not in (None, '')
        }
        if rc_subset:
            subset['response_contract'] = rc_subset

    reply_interpretation = parsed.get('reply_interpretation')
    if isinstance(reply_interpretation, dict):
        ri_subset = {
            key: reply_interpretation.get(key)
            for key in (
                'awaiting_state',
                'legacy_kind',
                'legacy_action',
                'classifier_kind',
                'classifier_action',
                'runtime_kind',
                'runtime_action',
                'applied',
            )
            if reply_interpretation.get(key) not in (None, '')
        }
        if ri_subset:
            subset['reply_interpretation'] = ri_subset

    tool_calls = parsed.get('tool_calls')
    if isinstance(tool_calls, list):
        tc_subset = []
        for item in tool_calls:
            if not isinstance(item, dict):
                continue
            name = str(item.get('name') or '').strip()
            if not name:
                continue
            tc_item = {'name': name}
            if 'ok' in item:
                tc_item['ok'] = bool(item.get('ok'))
            protocol = str(item.get('protocol') or '').strip()
            if protocol:
                tc_item['protocol'] = protocol
            error = str(item.get('error') or '').strip()
            if error:
                tc_item['error'] = error
            tc_subset.append(tc_item)
        if tc_subset:
            subset['tool_calls'] = tc_subset

    return subset or None


def _message_record(row: SupportMessage) -> dict[str, Any]:
    item: dict[str, Any] = {
        'id': int(row.id),
        'role': str(row.role or ''),
        'content': str(row.content or ''),
    }
    created_at = getattr(row, 'created_at', None)
    if created_at is not None:
        item['created_at'] = str(created_at)
    if str(row.role or '') == 'assistant':
        subset = _assistant_meta_subset(getattr(row, 'meta_json', None))
        if subset:
            item['assistant_meta'] = subset
    return item


def _load_sessions(
    *,
    assistant_db_path: str | Path | None,
    session_pks: list[int],
    session_ids: list[str],
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen: set[int] = set()
    with get_assistant_session(assistant_db_path) as db:
        for session_pk in session_pks:
            row = db.query(SupportSession).filter(SupportSession.id == int(session_pk)).first()
            if row is None:
                raise ValueError(f'session_pk {session_pk} not found')
            row_id = int(row.id)
            if row_id not in seen:
                selected.append({
                    'id': row_id,
                    'session_id': str(getattr(row, 'session_id', '') or ''),
                })
                seen.add(row_id)
        for session_id in session_ids:
            row = db.query(SupportSession).filter(SupportSession.session_id == str(session_id)).first()
            if row is None:
                raise ValueError(f'session_id {session_id!r} not found')
            row_id = int(row.id)
            if row_id not in seen:
                selected.append({
                    'id': row_id,
                    'session_id': str(getattr(row, 'session_id', '') or ''),
                })
                seen.add(row_id)
    return selected


def extract_behavioral_cases(
    *,
    assistant_db_path: str | Path | None = None,
    session_pks: list[int] | None = None,
    session_ids: list[str] | None = None,
    from_message_id: int | None = None,
    to_message_id: int | None = None,
    tags: list[str] | None = None,
    notes: str | None = None,
    label: str | None = None,
) -> list[dict[str, Any]]:
    session_pks = [int(item) for item in (session_pks or [])]
    session_ids = [str(item).strip() for item in (session_ids or []) if str(item).strip()]
    tags = [str(item).strip() for item in (tags or []) if str(item).strip()]
    if not session_pks and not session_ids:
        raise ValueError('at least one session selector is required')
    if label and (len(session_pks) + len(session_ids) > 1):
        raise ValueError('--label can only be used when extracting a single session')

    selected_sessions = _load_sessions(
        assistant_db_path=assistant_db_path,
        session_pks=session_pks,
        session_ids=session_ids,
    )
    assistant_db_display = get_assistant_db_uri(assistant_db_path)
    exported_at = datetime.now(UTC).isoformat()

    cases: list[dict[str, Any]] = []
    with get_assistant_session(assistant_db_path) as db:
        for session in selected_sessions:
            query = db.query(SupportMessage).filter(SupportMessage.session_pk == int(session['id'])).order_by(SupportMessage.id.asc())
            if isinstance(from_message_id, int) and from_message_id > 0:
                query = query.filter(SupportMessage.id >= int(from_message_id))
            if isinstance(to_message_id, int) and to_message_id > 0:
                query = query.filter(SupportMessage.id <= int(to_message_id))
            rows = query.all()
            if not rows:
                raise ValueError(f"no messages found for session {int(session['id'])} in the selected range")

            case_label = str(label or f"session_{int(session['id'])}")
            case = {
                'label': case_label,
                'tags': list(tags),
                'messages': [_message_record(row) for row in rows],
                'source': {
                    'assistant_db': assistant_db_display,
                    'session_pk': int(session['id']),
                    'session_id': str(session.get('session_id') or ''),
                    'from_message_id': from_message_id,
                    'to_message_id': to_message_id,
                    'exported_at': exported_at,
                },
            }
            if isinstance(notes, str) and notes.strip():
                case['notes'] = notes.strip()
            cases.append(case)
    return cases


def write_behavioral_cases(cases: list[dict[str, Any]], output_dir: str | Path) -> list[Path]:
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for case in cases:
        source = case.get('source') if isinstance(case.get('source'), dict) else {}
        session_pk = int(source.get('session_pk') or 0)
        label = str(case.get('label') or f'session_{session_pk}').strip()
        filename = f'{session_pk:04d}-{_slugify(label)}.json' if session_pk > 0 else f'{_slugify(label)}.json'
        path = root / filename
        path.write_text(json.dumps(case, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        written.append(path)
    return written


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Extract local behavioral scenarios from the assistant DB.')
    parser.add_argument('--assistant-db', help='Optional assistant DB path; defaults to resolved ISPEC assistant DB.')
    parser.add_argument('--output-dir', default=str(ROOT / 'tests' / 'behavioral' / 'local'))
    parser.add_argument('--session-pk', action='append', type=int, default=[])
    parser.add_argument('--session-id', action='append', default=[])
    parser.add_argument('--from-message-id', type=int)
    parser.add_argument('--to-message-id', type=int)
    parser.add_argument('--tag', action='append', default=[])
    parser.add_argument('--notes')
    parser.add_argument('--label')
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    cases = extract_behavioral_cases(
        assistant_db_path=args.assistant_db,
        session_pks=list(args.session_pk or []),
        session_ids=list(args.session_id or []),
        from_message_id=args.from_message_id,
        to_message_id=args.to_message_id,
        tags=list(args.tag or []),
        notes=args.notes,
        label=args.label,
    )
    written = write_behavioral_cases(cases, args.output_dir)
    for path in written:
        print(path)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
