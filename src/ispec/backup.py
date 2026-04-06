from __future__ import annotations

import hashlib
import json
import os
import shutil
import socket
import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ispec.config.paths import resolve_state_dir, resolved_path_catalog
from ispec.logging import get_logger


logger = get_logger(__name__)

_DEFAULT_BACKUP_ROOT = Path('/media/alex/202603/ispec-backups')
_DEFAULT_BACKUP_TARGET_SENTINEL = '.ispec-backup-target'
_DEFAULT_BACKUP_RETENTION_COUNT = 30


def _workspace_root_default() -> Path:
    return Path(__file__).resolve().parents[3]


def resolve_backup_root() -> Path:
    raw = str(os.getenv('ISPEC_BACKUP_ROOT') or '').strip()
    if raw:
        return Path(raw).expanduser()
    return _DEFAULT_BACKUP_ROOT


def backup_target_sentinel_name() -> str:
    raw = str(os.getenv('ISPEC_BACKUP_TARGET_SENTINEL') or '').strip()
    return raw or _DEFAULT_BACKUP_TARGET_SENTINEL


def backup_retention_count() -> int:
    raw = str(os.getenv('ISPEC_BACKUP_RETENTION_COUNT') or '').strip()
    if not raw:
        return _DEFAULT_BACKUP_RETENTION_COUNT
    try:
        return max(1, int(raw))
    except ValueError:
        return _DEFAULT_BACKUP_RETENTION_COUNT


def resolve_backup_status_path() -> Path:
    state_dir = Path(resolve_state_dir().path or (Path.home() / '.ispec')).expanduser()
    return state_dir / 'backup-status.json'


def load_backup_status(*, path: Path | None = None) -> dict[str, Any] | None:
    target = Path(path or resolve_backup_status_path()).expanduser()
    try:
        payload = json.loads(target.read_text(encoding='utf-8'))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )
    tmp.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _file_entry(*, source: Path | None, dest: Path, root: Path) -> dict[str, Any]:
    stat = dest.stat()
    return {
        'path': dest.relative_to(root).as_posix(),
        'bytes': int(stat.st_size),
        'sha256': _sha256(dest),
        'source_path': str(source) if source is not None else None,
    }


def _copy_file(*, source: Path, destination: Path, root: Path) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return _file_entry(source=source, dest=destination, root=root)


def _copy_tree(*, source: Path, destination: Path, root: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for path in sorted(source.rglob('*')):
        if not path.is_file():
            continue
        rel = path.relative_to(source)
        entries.append(_copy_file(source=path, destination=destination / rel, root=root))
    return entries


def _sqlite_backup(*, source: Path, destination: Path, root: Path) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(source) as src_db, sqlite3.connect(destination) as dest_db:
        src_db.backup(dest_db)
    return _file_entry(source=source, dest=destination, root=root)


def _collect_config_files(workspace_root: Path) -> list[Path]:
    configs_dir = workspace_root / 'configs'
    if not configs_dir.exists():
        return []
    return sorted(path for path in configs_dir.glob('*.local.*') if path.is_file())


def _collect_state_files() -> list[Path]:
    state_dir = Path(resolve_state_dir().path or (Path.home() / '.ispec')).expanduser()
    if not state_dir.exists():
        return []
    allowed_suffixes = {'.json', '.jsonl', '.pid', '.txt', '.status', '.log'}
    return sorted(
        path
        for path in state_dir.iterdir()
        if path.is_file() and (path.suffix.lower() in allowed_suffixes or path.name.endswith('.jsonl'))
    )


def _existing_database_paths() -> dict[str, Path]:
    catalog = resolved_path_catalog().get('database', {})
    results: dict[str, Path] = {}
    for name, location in catalog.items():
        if not isinstance(name, str) or name == 'db_dir':
            continue
        path_str = getattr(location, 'path', None)
        if not isinstance(path_str, str) or not path_str:
            continue
        path = Path(path_str).expanduser()
        if path.exists() and path.is_file():
            results[name] = path
    return results


def _secrets_freshness(workspace_root: Path, secrets_root: Path) -> dict[str, Any]:
    tracked_root = secrets_root / 'tracked'
    if not tracked_root.exists():
        return {'ok': False, 'reason': 'tracked_missing', 'stale_count': 0, 'stale_items': []}

    stale: list[str] = []
    checked = 0
    for tracked_file in sorted(tracked_root.rglob('*')):
        if not tracked_file.is_file() or not tracked_file.name.startswith('.env'):
            continue
        checked += 1
        rel = tracked_file.relative_to(tracked_root)
        workspace_file = workspace_root / rel
        if not workspace_file.exists() or not workspace_file.is_file():
            continue
        try:
            if workspace_file.stat().st_mtime > tracked_file.stat().st_mtime + 1:
                stale.append(rel.as_posix())
        except Exception:
            continue

    return {
        'ok': not stale,
        'tracked_root': str(tracked_root),
        'checked_files': checked,
        'stale_count': len(stale),
        'stale_items': stale[:20],
    }


def _snapshot_success_dirs(root: Path) -> list[tuple[datetime, Path]]:
    items: list[tuple[datetime, Path]] = []
    if not root.exists():
        return items
    for manifest_path in root.rglob('manifest.json'):
        if '.incomplete' in manifest_path.parts:
            continue
        try:
            payload = json.loads(manifest_path.read_text(encoding='utf-8'))
        except Exception:
            continue
        if not isinstance(payload, dict) or payload.get('status') != 'ok':
            continue
        completed_at = payload.get('completed_at_utc')
        if not isinstance(completed_at, str) or not completed_at:
            continue
        try:
            dt = datetime.fromisoformat(completed_at)
        except ValueError:
            continue
        items.append((dt, manifest_path.parent))
    items.sort(key=lambda item: item[0], reverse=True)
    return items


def _prune_successful_snapshots(root: Path, *, keep: int) -> list[str]:
    pruned: list[str] = []
    for _, path in _snapshot_success_dirs(root)[max(keep, 0):]:
        try:
            shutil.rmtree(path)
            pruned.append(str(path))
        except Exception:
            logger.exception('Failed to prune old backup snapshot %s', path)
    return pruned


def _target_ready(root: Path) -> tuple[bool, str | None]:
    if not root.exists():
        return False, f'Backup root does not exist: {root}'
    if not root.is_dir():
        return False, f'Backup root is not a directory: {root}'
    sentinel = root / backup_target_sentinel_name()
    if not sentinel.exists():
        return False, f'Missing backup target sentinel: {sentinel}'
    return True, None


def _status_payload(
    *,
    previous: dict[str, Any] | None,
    ok: bool,
    attempted_at: str,
    target_root: Path,
    last_error: str | None,
    latest_snapshot_path: str | None = None,
    last_succeeded_at: str | None = None,
    manifest_path: str | None = None,
    incomplete_snapshot_path: str | None = None,
    pruned_paths: list[str] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        'ok': bool(ok),
        'status': 'ok' if ok else 'error',
        'target_root': str(target_root),
        'last_attempted_at': attempted_at,
        'last_error': last_error,
        'last_succeeded_at': last_succeeded_at if ok else (previous or {}).get('last_succeeded_at'),
        'latest_snapshot_path': latest_snapshot_path if ok else (previous or {}).get('latest_snapshot_path'),
        'manifest_path': manifest_path if ok else (previous or {}).get('manifest_path'),
    }
    if incomplete_snapshot_path:
        payload['incomplete_snapshot_path'] = incomplete_snapshot_path
    if pruned_paths:
        payload['pruned_paths'] = list(pruned_paths)
    return payload


def create_backup_snapshot(*, workspace_root: Path | None = None) -> dict[str, Any]:
    current_workspace = Path(workspace_root or _workspace_root_default()).expanduser()
    backup_root = resolve_backup_root()
    now = datetime.now(UTC)
    attempted_at = now.isoformat()
    status_path = resolve_backup_status_path()
    previous = load_backup_status(path=status_path)

    ready, reason = _target_ready(backup_root)
    if not ready:
        payload = _status_payload(
            previous=previous,
            ok=False,
            attempted_at=attempted_at,
            target_root=backup_root,
            last_error=reason,
        )
        _write_json(status_path, payload)
        return payload

    timestamp = now.strftime('%Y%m%dT%H%M%SZ')
    day_dir = backup_root / now.strftime('%Y') / now.strftime('%m') / now.strftime('%d')
    final_dir = day_dir / timestamp
    stage_dir = backup_root / '.incomplete' / f'{timestamp}-{uuid.uuid4().hex[:8]}'

    manifest: dict[str, Any] = {
        'status': 'ok',
        'started_at_utc': attempted_at,
        'completed_at_utc': None,
        'hostname': socket.gethostname(),
        'workspace_root': str(current_workspace),
        'target_root': str(backup_root),
        'snapshot_path': None,
        'databases': [],
        'configs': [],
        'state_files': [],
        'logs': [],
        'secrets': {'files': [], 'freshness': None},
        'summary': {'file_count': 0, 'bytes_total': 0},
    }

    try:
        stage_dir.mkdir(parents=True, exist_ok=True)

        db_root = stage_dir / 'databases'
        for name, source in _existing_database_paths().items():
            dest = db_root / f'{name}.db'
            entry = _sqlite_backup(source=source, destination=dest, root=stage_dir)
            manifest['databases'].append({'name': name, **entry})

        config_root = stage_dir / 'configs'
        for source in _collect_config_files(current_workspace):
            dest = config_root / source.name
            manifest['configs'].append(_copy_file(source=source, destination=dest, root=stage_dir))

        state_root = stage_dir / 'state'
        for source in _collect_state_files():
            dest = state_root / source.name
            manifest['state_files'].append(_copy_file(source=source, destination=dest, root=stage_dir))

        log_dir_location = resolved_path_catalog().get('logging', {}).get('log_dir')
        log_path = Path(log_dir_location.path).expanduser() if getattr(log_dir_location, 'path', None) else None
        if log_path is not None and log_path.exists() and log_path.is_dir():
            manifest['logs'] = _copy_tree(source=log_path, destination=stage_dir / 'logs', root=stage_dir)

        secrets_root = current_workspace / 'secrets'
        if secrets_root.exists() and secrets_root.is_dir():
            manifest['secrets']['files'] = _copy_tree(source=secrets_root, destination=stage_dir / 'secrets', root=stage_dir)
            manifest['secrets']['freshness'] = _secrets_freshness(current_workspace, secrets_root)
        else:
            manifest['secrets']['freshness'] = {'ok': False, 'reason': 'missing_secrets_repo'}

        all_entries: list[dict[str, Any]] = []
        for key in ('databases', 'configs', 'state_files', 'logs'):
            value = manifest.get(key)
            if isinstance(value, list):
                all_entries.extend(item for item in value if isinstance(item, dict))
        secrets_value = manifest.get('secrets')
        secrets_files = secrets_value.get('files') if isinstance(secrets_value, dict) else None
        if isinstance(secrets_files, list):
            all_entries.extend(item for item in secrets_files if isinstance(item, dict))
        manifest['summary'] = {
            'file_count': len(all_entries),
            'bytes_total': sum(int(item.get('bytes') or 0) for item in all_entries),
        }
        manifest['completed_at_utc'] = datetime.now(UTC).isoformat()

        _write_json(stage_dir / 'manifest.json', manifest)
        day_dir.mkdir(parents=True, exist_ok=True)
        if final_dir.exists():
            final_dir = day_dir / f'{timestamp}-{uuid.uuid4().hex[:6]}'
        stage_dir.rename(final_dir)

        manifest['snapshot_path'] = str(final_dir)
        _write_json(final_dir / 'manifest.json', manifest)
        pruned = _prune_successful_snapshots(backup_root, keep=backup_retention_count())
        payload = _status_payload(
            previous=previous,
            ok=True,
            attempted_at=attempted_at,
            target_root=backup_root,
            last_error=None,
            latest_snapshot_path=str(final_dir),
            last_succeeded_at=str(manifest['completed_at_utc']),
            manifest_path=str(final_dir / 'manifest.json'),
            pruned_paths=pruned,
        )
        _write_json(status_path, payload)
        return payload
    except Exception as exc:
        logger.exception('Backup snapshot failed')
        payload = _status_payload(
            previous=previous,
            ok=False,
            attempted_at=attempted_at,
            target_root=backup_root,
            last_error=f'{type(exc).__name__}: {exc}',
            incomplete_snapshot_path=str(stage_dir),
        )
        _write_json(status_path, payload)
        return payload
