# Legacy Sync Operations

Automatic metadata sync is opt-in. It does not require the LLM or the exploratory
orchestrator loop. A running supervisor executes queued sync commands; starting
the API alone does not execute them. Leave automatic sync off while the legacy
bridge is unavailable.

## Inspect Without Syncing

From the deployment repository, run `make status-legacy-sync`, or use
`ispec db legacy-sync-status --agent-database /path/to/ispec-agent.db` directly.
The staff Operations page uses `GET /api/ops/legacy-sync/status`, with the same
role/username restriction as manual sync enqueueing. These reads do not contact
legacy or enqueue work.

Status distinguishes the last successful import, last failed/successful attempt,
active command, and recent commands (including previews/cancellations). It also
shows the configured interval and next due time. "Enabled" describes scheduling
configuration, not proof that the supervisor or bridge is healthy. Inspect
`make dev-status` separately. The Operations page refreshes sync status every
15 seconds while visible; its general snapshot/cursor display refreshes manually.

## Enable Later

After checking bridge connectivity, set these in the local service environment:

```dotenv
ISPEC_LEGACY_SYNC_ENABLED=1
ISPEC_LEGACY_SYNC_INTERVAL_SECONDS=86400
ISPEC_LEGACY_SYNC_MAX_ATTEMPTS=1
```

Inspect other supervisor schedules and queued work before starting/restarting the
supervisor: it can execute more than legacy imports. Do not blindly restart it
when obsolete Slack jobs remain queued. For the current local dev session,
backend and supervisor restarts are separate Make targets/options.

A real completed attempt, successful or failed, starts the daily interval.
Durable command receipts preserve this timing across supervisor restarts.
An existing queued/running sync prevents another scheduled enqueue. Dry-runs and
cancelled commands do not count as real attempts or postpone the next import.
The first enabled cycle with no previous real attempt is due immediately.
Manual enqueue remains possible with scheduling disabled, but still needs a worker.

## Evidence and Failures

Aggregate `sync-legacy-all` calls write compact daily JSONL files under
`<ISPEC_LOG_DIR>/legacy-sync/YYYY-MM-DD.jsonl`. Without `ISPEC_LOG_DIR`, the normal
log-directory resolver applies. The status command reports the resolved path.
New runs prune only date-named event files older than 30 days in this directory.
Events contain run/command IDs, phase status, timings, counts and errors, not raw
project rows or comment text. Credential-bearing HTTP URLs in exception messages
are replaced with `<legacy endpoint>`; this is not a general-purpose secret scrubber.

Queued runs also update their existing agent command receipt at phase boundaries.
No new database/table is required. Successful prior phases remain visible if a
later phase fails. Log/progress persistence failures are reported in `sync_run`
as `log_error`/`progress_error`; they do not turn a completed import into a retry.

Core sync still fetches outside write transactions and commits pages with verified
cursors. Completed pages survive a later failure. Counts for the interrupted
phase can be incomplete; a failed run does not mean all its imports rolled back.
Forward `since`/PK cursors use ascending modification-time/PK ordering. Newest-first
ordering with a forward cursor overlaps pages and can skip older eligible rows;
the pagination regression tests model the legacy API's actual ordering/filtering.
Existing historical cursors alone do not prove full backfill coverage.
A supervisor crash may leave a running receipt requiring existing queue recovery;
this status view does not independently judge a job dead or run recovery.
The existing sync executor remains synchronous; phase reporting does not add a
parallel worker or eliminate time spent waiting on legacy HTTP requests.

Direct aggregate CLI calls get JSONL events but no queued command receipt. Direct
single-table CLI syncs retain their existing logging/cursors and are not included
in this scheduler history. This is not a universal import ledger.

## Transport Choices

iSPEC currently fetches through the HTTP API. Separately, `bcmproteomics` can
provide the API's FileMaker connection using ODBC or JDBC. A bounded live replay
through a temporary localhost API backed by JDBC succeeded, including comments;
this did not install a native JDBC transport in iSPEC or establish a speedup.

The smallest operational change is a separately managed JDBC-backed API, selected
with `ISPEC_LEGACY_API_URL`. Keep that API URL distinct from the FileMaker host in
`~/.ispec/ispec.conf`. Restart/reload the relevant service after changing config.
Switch only between endpoints for the same database and equivalent cursor/field
semantics; changing the transport alone should not require resetting cursors.

A future native JDBC option should replace only page retrieval, sharing existing
mapping, reconciliation, checkpoints and receipts. Compare HTTP/JDBC results for
the same pages, including numeric IDs, NULLs, timestamp precision and composite
keys, before switching routinely. Use explicit selection, not silent mid-run
fallback. Direct access removes the HTTP hop but not database/driver failures or
the need for bounded query timeouts. Isolate driver work from the web process.
Keep legacy comment write-back on its tested path until separately validated.

Project intake need not wait for historical runs: use the existing separate
project/experiment sync commands and targeted run imports as appropriate. Reserve
a full run backfill for a deliberate resumable job. The aggregate daily command
still includes runs; narrower scheduled lanes would be a separate change.

## Retire Obsolete Jobs

Disable obsolete entries in the local assistant schedule file first, then preview
`ispec agent queue-cancel-stale` with a narrow command type/status/age filter.
Inspect the selected rows before rerunning with `--confirm` and a reason. Existing
queue cancellation retains the receipt as failed with `cancelled: true`; it does
not delete history or execute the queued message. Archive/export useful evidence
before cleanup. Do not use a broad cancel operation on current work.
