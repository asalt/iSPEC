from __future__ import annotations

import difflib
import json
import sqlite3
from pathlib import Path

from ispec.prompt.connect import connect_prompts_db, get_prompts_db_path
from ispec.prompt.loader import load_prompt_source, resolve_prompt_root
from ispec.prompt.sync import sync_prompts


def register_subcommands(subparsers) -> None:
    sync_parser = subparsers.add_parser('sync', help='Sync prompt files and bindings into prompts.db')
    sync_parser.add_argument('--check', action='store_true', help='Validate prompt drift without mutating prompts.db')

    list_parser = subparsers.add_parser('list', help='List prompt families')
    list_parser.add_argument('--query', default='', help='Optional substring filter for family/title/notes')

    show_parser = subparsers.add_parser('show', help='Show the active current prompt file')
    show_parser.add_argument('family', help='Prompt family name')

    diff_parser = subparsers.add_parser('diff', help='Diff current prompt text against the previous synced version')
    diff_parser.add_argument('family', help='Prompt family name')

    search_parser = subparsers.add_parser('search', help='Search prompt families and notes')
    search_parser.add_argument('query', help='Substring query')


def _db_exists() -> bool:
    path = get_prompts_db_path()
    return bool(path is not None and path.exists())


def _family_rows(query: str = '') -> list[dict[str, object]]:
    query_text = str(query or '').strip().lower()
    if not _db_exists():
        rows: list[dict[str, object]] = []
        for path in sorted(resolve_prompt_root().glob('*.md')):
            source = load_prompt_source(path.stem)
            text = ' '.join(filter(None, [source.family, source.title or '', source.notes or ''])).lower()
            if query_text and query_text not in text:
                continue
            rows.append(
                {
                    'family': source.family,
                    'title': source.title,
                    'notes': source.notes,
                    'version_num': None,
                    'bindings': 0,
                }
            )
        return rows

    like = f'%{query_text}%'
    with connect_prompts_db() as conn:
        sql = """
        SELECT pf.family, pf.title, pf.notes, pv.version_num,
               COUNT(pb.id) AS binding_count
        FROM prompt_family pf
        LEFT JOIN prompt_version pv ON pv.id = pf.active_version_id
        LEFT JOIN prompt_binding pb ON pb.family_id = pf.id
        WHERE (? = '' OR LOWER(pf.family) LIKE ? OR LOWER(COALESCE(pf.title, '')) LIKE ? OR LOWER(COALESCE(pf.notes, '')) LIKE ?)
        GROUP BY pf.id, pf.family, pf.title, pf.notes, pv.version_num
        ORDER BY pf.family ASC
        """
        return [
            {
                'family': str(row['family']),
                'title': row['title'],
                'notes': row['notes'],
                'version_num': int(row['version_num']) if row['version_num'] is not None else None,
                'bindings': int(row['binding_count'] or 0),
            }
            for row in conn.execute(sql, (query_text, like, like, like)).fetchall()
        ]


def dispatch(args) -> None:
    if args.subcommand == 'sync':
        summary = sync_prompts(check=bool(args.check))
        payload = {
            'new_families': summary.new_families,
            'new_versions': summary.new_versions,
            'metadata_updates': summary.metadata_updates,
            'binding_updates': summary.binding_updates,
            'missing_families': summary.missing_families,
            'stale_bindings_removed': summary.stale_bindings_removed,
            'check_failed': summary.check_failed,
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        if args.check and summary.check_failed:
            raise SystemExit(1)
        return

    if args.subcommand == 'list':
        rows = _family_rows(args.query)
        for row in rows:
            version = f" v{row['version_num']}" if row['version_num'] is not None else ''
            title = f" :: {row['title']}" if row['title'] else ''
            notes = f" [{row['notes']}]" if row['notes'] else ''
            bindings = f" bindings={row['bindings']}" if row['bindings'] else ''
            print(f"{row['family']}{version}{bindings}{title}{notes}")
        return

    if args.subcommand == 'search':
        for row in _family_rows(args.query):
            version = f" v{row['version_num']}" if row['version_num'] is not None else ''
            title = f" :: {row['title']}" if row['title'] else ''
            notes = f" [{row['notes']}]" if row['notes'] else ''
            print(f"{row['family']}{version}{title}{notes}")
        return

    if args.subcommand == 'show':
        source = load_prompt_source(args.family)
        payload = {
            'family': source.family,
            'source_path': source.source_path,
            'title': source.title,
            'notes': source.notes,
            'body_sha256': source.body_sha256,
            'body': source.body,
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        return

    if args.subcommand == 'diff':
        source = load_prompt_source(args.family)
        if not _db_exists():
            raise SystemExit('prompts.db not found; run `ispec prompt sync` first')
        with connect_prompts_db() as conn:
            row = conn.execute(
                """
                SELECT pv.body_text, pv.version_num
                FROM prompt_version pv
                JOIN prompt_family pf ON pf.id = pv.family_id
                WHERE pf.family = ?
                ORDER BY pv.version_num DESC
                LIMIT 1 OFFSET 1
                """,
                (args.family,),
            ).fetchone()
        if row is None:
            raise SystemExit(f'No previous synced version found for {args.family}')
        previous = str(row['body_text'] or '').splitlines(keepends=True)
        current = source.body.splitlines(keepends=True)
        diff = difflib.unified_diff(
            previous,
            current,
            fromfile=f'{args.family}@v{int(row["version_num"])}',
            tofile=f'{args.family}@current',
        )
        print(''.join(diff), end='')
        return

    raise SystemExit(f'Unknown prompt subcommand: {args.subcommand}')
