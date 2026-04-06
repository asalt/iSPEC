from __future__ import annotations

import sqlite3
from textwrap import dedent

from ispec.prompt.sync import sync_prompts


def _write_prompt(path, *, title: str, notes: str, body: str) -> None:
    path.write_text(
        "+++\n"
        f'title = "{title}"\n'
        f'notes = "{notes}"\n'
        "+++\n"
        + body,
        encoding="utf-8",
    )


def test_sync_prompts_tracks_versions_and_binding_updates(tmp_path, monkeypatch):
    prompt_root = tmp_path / "prompts"
    source_root = tmp_path / "src"
    prompt_root.mkdir()
    (source_root / "pkg").mkdir(parents=True)
    prompt_path = prompt_root / "assistant.example.classifier.md"
    _write_prompt(prompt_path, title="Example", notes="Initial notes", body="Body v1\n")
    (source_root / "pkg" / "module.py").write_text(
        dedent(
            """
            from ispec.prompt import prompt_binding

            @prompt_binding("assistant.example.classifier")
            def example_prompt():
                return "unused"
            """
        ),
        encoding="utf-8",
    )
    db_path = tmp_path / "prompts.db"
    monkeypatch.setenv("ISPEC_PROMPTS_DB_PATH", str(db_path))

    summary = sync_prompts(prompt_root=prompt_root, source_root=source_root)
    assert summary.new_families == ["assistant.example.classifier"]
    assert summary.new_versions == [("assistant.example.classifier", 1)]
    assert summary.binding_updates == 1
    assert summary.check_failed is False

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    family = conn.execute("SELECT * FROM prompt_family WHERE family = ?", ("assistant.example.classifier",)).fetchone()
    assert family is not None
    assert family["title"] == "Example"
    version_rows = conn.execute(
        "SELECT version_num, body_text FROM prompt_version WHERE family_id = ? ORDER BY version_num",
        (family["id"],),
    ).fetchall()
    assert [(int(row["version_num"]), str(row["body_text"])) for row in version_rows] == [(1, "Body v1\n")]

    _write_prompt(prompt_path, title="Example", notes="Updated notes", body="Body v1\n")
    summary = sync_prompts(prompt_root=prompt_root, source_root=source_root)
    assert summary.new_versions == []
    assert summary.metadata_updates == ["assistant.example.classifier"]

    version_rows = conn.execute(
        "SELECT version_num, body_text FROM prompt_version WHERE family_id = ? ORDER BY version_num",
        (family["id"],),
    ).fetchall()
    assert [(int(row["version_num"]), str(row["body_text"])) for row in version_rows] == [(1, "Body v1\n")]

    _write_prompt(prompt_path, title="Example", notes="Updated notes", body="Body v2\n")
    summary = sync_prompts(prompt_root=prompt_root, source_root=source_root)
    assert summary.new_versions == [("assistant.example.classifier", 2)]

    version_rows = conn.execute(
        "SELECT version_num, body_text FROM prompt_version WHERE family_id = ? ORDER BY version_num",
        (family["id"],),
    ).fetchall()
    assert [(int(row["version_num"]), str(row["body_text"])) for row in version_rows] == [
        (1, "Body v1\n"),
        (2, "Body v2\n"),
    ]
    conn.close()


def test_sync_prompts_check_mode_detects_drift_without_mutating_db(tmp_path, monkeypatch):
    prompt_root = tmp_path / "prompts"
    source_root = tmp_path / "src"
    prompt_root.mkdir()
    source_root.mkdir()
    prompt_path = prompt_root / "assistant.check.example.md"
    _write_prompt(prompt_path, title="Check", notes="Initial", body="Body v1\n")
    db_path = tmp_path / "prompts.db"
    monkeypatch.setenv("ISPEC_PROMPTS_DB_PATH", str(db_path))

    sync_prompts(prompt_root=prompt_root, source_root=source_root)
    _write_prompt(prompt_path, title="Check", notes="Initial", body="Body v2\n")

    summary = sync_prompts(prompt_root=prompt_root, source_root=source_root, check=True)
    assert summary.check_failed is True
    assert summary.new_versions == [("assistant.check.example", 2)]

    conn = sqlite3.connect(str(db_path))
    count = conn.execute("SELECT COUNT(*) FROM prompt_version").fetchone()[0]
    conn.close()
    assert count == 1
