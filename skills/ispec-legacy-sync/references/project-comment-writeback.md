# Project Comment Writeback

## Current State

- There is now a compare-before-insert CLI path in the repo: `iSPEC/.venv/bin/ispec db sync-project-comments-to-legacy`.
- The current importer still pulls legacy history from `/api/v2/legacy/tables/.../rows` into the local SQLAlchemy DB.
- Local writes exist in two places:
  - REST CRUD creates `project_comment` rows through the generic API router.
  - The assistant `create_project_comment` tool creates `ProjectComment(...)` rows directly and commits them.
- The live legacy schema uses `iSPEC_ProjectHistory`, not `ProjectComments`, so the upstream API repo must support that table family for round-trip note sync.

## Safest v1

- Keep the local insert as the source of truth.
- Use `sync-project-comments-to-legacy --dry-run` first to inspect the candidate set.
- The current compare key is `(project_id, normalized note text, creation timestamp rounded to seconds)`.
- Legacy writes are insertion-only and use compare-before-insert rather than local sync-state flags.

## Why Not ORM Events

- Inbound legacy imports and legacy API sync also insert `project_comment` rows locally.
- A global `after_insert` hook would risk re-pushing imported legacy rows back to legacy.

## Guardrails

- Skip imported rows. Imported comments are already distinguishable by `person_id == 0` and `com_LegacyImportTS`.
- Exclude `System, System` comments from writeback consideration even if the FK changes.
- Keep v1 insertion-only. Do not attempt overwrite or reconciliation.
- Prefer dry-run first, then explicit push after the upstream API deployment is confirmed.

## Key Files

- `iSPEC/src/ispec/assistant/tools.py`
- `iSPEC/src/ispec/api/routes/routes.py`
- `iSPEC/src/ispec/db/crud.py`
- `iSPEC/src/ispec/db/legacy_sync.py`
- `iSPEC/src/ispec/db/models/core.py`
- `/mnt/e/projects/bcmproteomics/bcmproteomics/api_v2/api.py`

## Tests To Extend

- `iSPEC/tests/unit/db/test_legacy_sync_project_comments.py`
- `iSPEC/tests/unit/db/test_legacy_push_project_comments.py`
- `/mnt/e/projects/bcmproteomics/bcmproteomics/tests/test_api_v2.py`
