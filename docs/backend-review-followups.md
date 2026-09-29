# Backend review follow-ups

## P1: ExperimentRun source reconciliation

Targeted and incremental run imports use the same field-level reconciler in
`ispec.db.legacy_reconciliation`. Project, person, and experiment import policies
are unchanged. This does not introduce another sync job or enable scheduling.

Two nullable, read-only JSON text columns use the existing core database and
migration/compatibility-init paths:

- `legacy_source_baseline_json`: latest observed, coerced source values.
- `legacy_source_conflicts_json`: unresolved field conflicts, including the
  original previous source, current local value, incoming value, and reason.

Both use `{"version":1,"source":"<legacy table URL>","values":{...}}`.
The URL identifies the source, not the current pagination cursor. A missing,
malformed, or differently sourced snapshot is not trusted. The baseline records
what was observed remotely, not a copy of whatever happened to be local at sync
time. Missing fields in a partial payload are not deletions; explicit nulls are.

For each observed field, compare previous source (B), current local (L), and
incoming source (S):

| Condition | Result |
| --- | --- |
| L = S | Accept agreement and clear the conflict |
| L = B, S differs | Import the source-only change |
| S = B, L differs, no unresolved conflict | Preserve the local-only change |
| Both changed differently | Preserve local and record a conflict |
| No trusted B and L differs from S | Preserve local; record `unknown_baseline` |

The existing explicit `backfill_missing` option can fill a null/empty local
field without a baseline. It never overwrites a nonempty value and does not
override a deliberate local deletion once a baseline exists. Without that
option, even empty preexisting values are treated conservatively.

Successful sync transactions advance the source snapshot even for conflicts.
Conflict evidence persists across identical replays, rather than disappearing
as a false local-only change. Agreement resolves a conflict; once local matches
the last observed source, subsequent source-only updates can be imported. A
conflict in one field does not block safe updates to other fields. The existing
row summary counts a row with unresolved conflicts as `conflicted`, even if
other fields were updated; inspect the JSON for field-level detail. Other
existing rows count as `updated`, including unchanged replays. The legacy
`backfilled` summary key remains for compatibility, but this reconciler does not
emit that separate row outcome.

Import/source-modification timestamps remain bookkeeping. Local modification
timestamps retain normal ORM behavior, including sync mutations, and are not
used as proof of local edits. Identity fields and bookkeeping are excluded from
reconciliation. Snapshot writes and imported values share the transaction;
dry-run rolls back both plus any cursor changes. Migration 0009 deliberately
does not invent a historical baseline for old rows. Existing normalization and
the generated sample-name fallback from the run identity are unchanged.

Tradeoffs: this stores the latest source and current unresolved conflicts, not
an append-only history or a new conflict-resolution UI. Old differing values
require review or eventual agreement; a changed source URL requires reseeding
through the same conservative rules. A source-mapped field absent on first
import has no baseline until observed, apart from the existing generated
sample-name fallback. Wider use for project/experiment sync is not part of this
P1 fix.

## P1: Shadow approval is not authorization

The project-comment approval evaluator still records classifier decisions,
shadow tickets, and actual write outcomes. Its outputs do not set live
`project_comment_save_authorized` or force `create_project_comment` tool choice.
Existing explicit-save, pending-confirmation, permission, and tool-guard paths
remain the live authority. A shadow result cannot promote a draft request into
a write or veto an independently authorized write.

Even the evaluator's currently accepted `own` setting produces only
`shadow_only` tickets with `live_applied=false`; it does not enable a ticket
actuator. This is separate from the existing live turn-decision controls.
Adding real ticket enforcement remains a separate reviewed change. Classifier
latency/availability still affects evaluation cost, but never write authority.

Regression tests exercise no-op and attempted-write replies under off/shadow/own
evaluation, explicit saves under approval/denial/invalid JSON/timeouts, actual
tool-result-derived write outcomes, and existing sandbox confirmation flows.

## P2: Short transactions around incremental sync pages

The project, person, experiment, and incremental run loops now end the initial
cursor read transaction before fetching. They retain the session/engine for the
run, but never retain a transaction across HTTP. `legacy_sync_transactions`
captures a plain cursor checkpoint, starts a short `BEGIN IMMEDIATE` transaction
after a page arrives, and commits each page's rows and cursor together. Local
rows are re-read after the fetch so intervening edits reach reconciliation.

The saved cursor is checked again under the write lock. If another sync changed
or created it during the fetch, the stale page is rejected with a retry error;
the winning cursor is not overwritten. Comparison includes the cursor's
modification timestamp, normalized to SQLite's naive UTC representation.

This intentionally changes failure recovery to page-level checkpoints: a failed
later page does not undo earlier committed pages, and retry resumes from the
last committed cursor. Reset/override values stay in memory until a page commits;
an empty response or fetch failure does not erase stored progress. Dry-runs
advance only their in-memory pagination and roll back each page. Existing schema
initialization remains unchanged and is not a dry-run schema guarantee.

Tests use real, independent SQLite connections to write during each fetch, with
new/existing cursors and real/dry-run syncs. They also cover HTTP/apply failure,
retry, competing cursor creation/update, empty responses, and local edits made
while a run's remote data is being fetched. This removes network-duration locks,
not SQLite's normal single-writer constraint during local page application.

No scheduler or service is enabled by these changes. Live database maintenance
is a separately backed-up, explicitly authorized operation; no legacy data
deletion or reimport is required for this fix.
