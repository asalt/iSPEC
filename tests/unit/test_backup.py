from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from ispec.backup import create_backup_snapshot, load_backup_status


def _make_sqlite_db(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as db:
        db.execute('create table example (id integer primary key, name text)')
        db.execute("insert into example (name) values ('ok')")
        db.commit()


def test_create_backup_snapshot_copies_operational_data(tmp_path, monkeypatch):
    workspace_root = tmp_path / 'workspace'
    workspace_root.mkdir()
    (workspace_root / 'configs').mkdir()
    (workspace_root / 'configs' / 'assistant-schedules.local.json').write_text('{"jobs": []}\n', encoding='utf-8')

    secrets_root = workspace_root / 'secrets'
    (secrets_root / 'tracked' / 'iSPEC').mkdir(parents=True)
    (secrets_root / '.git').mkdir()
    (secrets_root / 'tracked' / '.env.local').write_text('A=1\n', encoding='utf-8')
    (workspace_root / '.env.local').write_text('A=1\n', encoding='utf-8')

    state_dir = tmp_path / 'state'
    state_dir.mkdir()
    (state_dir / 'supervisor.json').write_text('{"ok": true}\n', encoding='utf-8')

    log_dir = tmp_path / 'logs'
    (log_dir / 'vllm').mkdir(parents=True)
    (log_dir / 'vllm' / 'vllm-20260402.log').write_text('hello\n', encoding='utf-8')

    db_path = tmp_path / 'data' / 'core.db'
    _make_sqlite_db(db_path)

    backup_root = tmp_path / 'backup-target'
    backup_root.mkdir()
    (backup_root / '.ispec-backup-target').write_text('ok\n', encoding='utf-8')

    monkeypatch.setenv('ISPEC_DB_PATH', str(db_path))
    monkeypatch.setenv('ISPEC_STATE_DIR', str(state_dir))
    monkeypatch.setenv('ISPEC_LOG_DIR', str(log_dir))
    monkeypatch.setenv('ISPEC_BACKUP_ROOT', str(backup_root))
    monkeypatch.setenv('ISPEC_BACKUP_TARGET_SENTINEL', '.ispec-backup-target')
    monkeypatch.setenv('ISPEC_BACKUP_RETENTION_COUNT', '5')

    payload = create_backup_snapshot(workspace_root=workspace_root)

    assert payload['ok'] is True
    snapshot_path = Path(payload['latest_snapshot_path'])
    assert snapshot_path.exists()
    manifest = json.loads((snapshot_path / 'manifest.json').read_text(encoding='utf-8'))
    assert manifest['summary']['file_count'] >= 4
    assert any(item['name'] == 'core' for item in manifest['databases'])
    assert manifest['secrets']['freshness']['ok'] is True
    assert (snapshot_path / 'databases' / 'core.db').exists()
    assert (snapshot_path / 'configs' / 'assistant-schedules.local.json').exists()
    assert (snapshot_path / 'state' / 'supervisor.json').exists()
    assert (snapshot_path / 'logs' / 'vllm' / 'vllm-20260402.log').exists()

    status = load_backup_status(path=state_dir / 'backup-status.json')
    assert isinstance(status, dict)
    assert status['ok'] is True


def test_create_backup_snapshot_fails_closed_when_target_missing(tmp_path, monkeypatch):
    state_dir = tmp_path / 'state'
    state_dir.mkdir()
    backup_root = tmp_path / 'missing-target'
    monkeypatch.setenv('ISPEC_STATE_DIR', str(state_dir))
    monkeypatch.setenv('ISPEC_BACKUP_ROOT', str(backup_root))
    payload = create_backup_snapshot(workspace_root=tmp_path / 'workspace')
    assert payload['ok'] is False
    assert 'does not exist' in str(payload['last_error'])
    status = load_backup_status(path=state_dir / 'backup-status.json')
    assert isinstance(status, dict)
    assert status['ok'] is False
