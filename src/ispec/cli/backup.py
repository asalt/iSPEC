from __future__ import annotations

import json
import sys

from ispec.backup import create_backup_snapshot, load_backup_status


def register_subcommands(subparsers) -> None:
    subparsers.add_parser('snapshot', help='Create an operational backup snapshot')
    subparsers.add_parser('show-status', help='Show the latest local backup status JSON')


def dispatch(args) -> None:
    if args.subcommand == 'snapshot':
        payload = create_backup_snapshot()
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        if not payload.get('ok'):
            raise SystemExit(1)
        return
    if args.subcommand == 'show-status':
        payload = load_backup_status() or {}
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        return
    raise SystemExit(f'Unknown backup subcommand: {args.subcommand}')
