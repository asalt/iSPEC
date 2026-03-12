---
name: ispec-legacy-sync
description: Refresh and inspect the local iSPEC import database from legacy sources in this repo. Use when Codex needs to run `make` or `iSPEC/.venv/bin/ispec db` legacy sync commands, compare local project comments against legacy before push, choose between API sync and XLSX import, inspect `legacy_sync_state`, or plan targeted hydration of project comments and experiment runs without accidentally using the wrong DB home.
---

# iSPEC Legacy Sync

Use the repo root as the working directory. Prefer the top-level `Makefile` when possible because it loads the repo env files and pins the core DB to `~/.ispec/db/ispec.db`, which currently resolves through symlinks into `iSPEC/data/`.

If you run `iSPEC/.venv/bin/ispec db ...` directly, pass `--database ~/.ispec/db/ispec.db` unless you intentionally want a different DB.

## Quick Start

Bounded local refresh:

```bash
iSPEC/.venv/bin/ispec db sync-legacy-all --database ~/.ispec/db/ispec.db --no-backfill-missing
```

Targeted project verification:

```bash
make sync-legacy-projects LEGACY_PROJECT_ID=<PRJRecNo>
iSPEC/.venv/bin/ispec db sync-legacy-project-comments --database ~/.ispec/db/ispec.db --id <PRJRecNo>
```

Specific experiment runs:

```bash
make sync-legacy-experiments LEGACY_EXPERIMENT_ID=<EXPRecNo>
make sync-legacy-experiment-runs LEGACY_EXPERIMENT_ID=<EXPRecNo>
```

Offline full-history import from FileMaker XLSX exports:

```bash
iSPEC/.venv/bin/ispec db import-legacy --data-dir <dump_dir> --database ~/.ispec/db/ispec.db --mode overwrite
```

Safe writeback compare:

```bash
iSPEC/.venv/bin/ispec db sync-project-comments-to-legacy --database ~/.ispec/db/ispec.db --dry-run
```

## Choose the Path

- Use `sync-legacy-all --no-backfill-missing` for the normal non-destructive dev refresh. It syncs projects, people, and experiments, then hydrates only a small number of recently touched project comments and experiment runs.
- Use the `make sync-legacy-*` wrappers when you only need projects, experiments, or runs and want the wrapper to manage env loading.
- Use `sync-legacy-project-comments --id <PRJRecNo>` when a specific project's history matters now. There is no top-level Make target for this command.
- Use `sync-project-comments-to-legacy --dry-run` to compare local non-System comments against legacy `ProjectHistory` before any upstream insertion.
- Use `import-legacy` when you have FileMaker XLSX dumps and want the older one-shot import path, including project history.

## Workflow

1. Check the DB target and legacy connection.
- Confirm `iSPEC/.venv/bin/ispec` exists.
- Prefer `make` or pass `--database ~/.ispec/db/ispec.db`.
- Legacy connection resolves from `--legacy-url`, `ISPEC_LEGACY_API_URL`, `ISPEC_LEGACY_CONF` or `~/.ispec/ispec.conf`, then `iSPEC/data/ispec-legacy-schema.json`.

2. Refresh incrementally by default.
- Start with `iSPEC/.venv/bin/ispec db sync-legacy-all --database ~/.ispec/db/ispec.db --no-backfill-missing`.
- Do not reset unless the user explicitly asks for a rebuild.

3. Hydrate details selectively.
- For project notes/history, run `iSPEC/.venv/bin/ispec db sync-legacy-project-comments --database ~/.ispec/db/ispec.db --id <PRJRecNo>`.
- For experiment runs, run `make sync-legacy-experiment-runs LEGACY_EXPERIMENT_ID=<EXPRecNo>` or the equivalent direct CLI command.
- Do not blindly raise `--max-project-comments` or `--max-experiment-runs`. Those options fan out into one follow-up fetch per project or experiment and can turn into long jobs.

4. Inspect what landed.
- Check row counts:

```bash
sqlite3 ~/.ispec/db/ispec.db "select count(*) from project; select count(*) from person; select count(*) from experiment; select count(*) from project_comment;"
```

- Check incremental cursors:

```bash
sqlite3 ~/.ispec/db/ispec.db "select legacy_table, since, since_pk from legacy_sync_state order by legacy_table;"
```

5. Compare before any writeback.
- Run `iSPEC/.venv/bin/ispec db sync-project-comments-to-legacy --database ~/.ispec/db/ispec.db --dry-run`.
- This only considers local comments that are not attached to `System, System` and compares them against legacy `ProjectHistory` before any insert.

## Cautions

- Single-ID syncs are debug or verification operations and do not advance `legacy_sync_state`.
- `sync-legacy-all` backfills missing project, experiment, and run fields by default. The individual sync commands do not unless `--backfill-missing` is passed.
- Imported legacy project history becomes local `project_comment` rows with `person_id=0` and `com_LegacyImportTS` set.
- Local writeback uses compare-before-insert against legacy `iSPEC_ProjectHistory` and currently should be dry-run first until the upstream API deployment is confirmed.
- Syncing comments or runs before parent metadata can create placeholder project or experiment rows.
- The legacy API is finicky. Warnings about missing fields in `repeat` mode are expected when the client succeeds after falling back to `csv` or chunked field fetches.
- The standardized DB home is `~/.ispec/db/`. In this checkout it currently resolves through symlinks into `iSPEC/data/`.

## Writeback Reference

For local-to-legacy project-note insertion planning, read [references/project-comment-writeback.md](references/project-comment-writeback.md).
