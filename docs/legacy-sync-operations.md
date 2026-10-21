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
The dispatcher runs inbound aggregate sync in a dedicated CLI child process.
The existing sync code remains sequential inside that child, but waiting on
legacy HTTP no longer blocks the dispatcher's other queued commands. Direct
CLI sync calls still run in the invoking process.

## Quiet Worker and Sync Children

Routine health probes are off by default (`ISPEC_SUPERVISOR_HEALTH_CHECKS_ENABLED=0`).
An idle supervisor does not create health-check steps/events or call a model.
It still checks configured schedules and the durable command queue. Queue polling
defaults to 30 seconds (`ISPEC_SUPERVISOR_COMMAND_POLL_SECONDS`), so new work may
wait that long. The 30-second heartbeat is only a liveness timestamp. If probes
are explicitly enabled, the CLI/Make default is one health step per 300 seconds.
The optional model-driven orchestrator remains separately opt-in.

For queued `legacy_sync_all` work, the dispatcher claims the existing receipt and
launches `python -m ispec supervisor sync-worker` with the command ID, run ID,
attempt, and a one-use launch token. This is an internal executor, not a second
scheduler or an arbitrary-shell entry point. It rejects unclaimed, mismatched,
and duplicate launches. Credentials and the import payload stay off argv; the
child inherits the configured environment and reads its own claimed receipt.

Each job has an append-only attempt log at
`<ISPEC_LOG_DIR>/supervisor-jobs/sync-<command>-attempt-<attempt>.log` in addition
to the daily sync JSONL. The receipt retains worker PID/start identity, log path,
phase progress, final import outcome, and observed exit code. These per-job logs
currently have no automatic pruning. A completed sync receipt remains authoritative
even if interpreter shutdown subsequently fails; the exit code is separate evidence.

Only one queued inbound sync runs at a time; other command types can proceed.
The child records its own final receipt, including if its parent has exited.
On restart, a still-live child is not retried. A verified dead child with an
unfinished receipt is marked failed during queue inspection, preserving partial
progress, rather than silently rerun. Normal supervisor shutdown terminates and
reaps its child; default timeout is six hours
(`ISPEC_SUPERVISOR_SYNC_TIMEOUT_SECONDS=21600`). Timeout enforcement belongs to
the parent dispatcher; an orphan can finish but is not independently timed out.
Committed pages survive interruption; inspect the receipt before retrying.

`supervisor status` (including `--json`) derives liveness from the process and
retains `recorded_status` separately. Operations marks a running record with an
expired heartbeat as `stale`. Neither read fabricates an exit time or rewrites
historical rows. SIGTERM follows normal shutdown cleanup. The queue is still a
single dispatcher: other synchronous handlers, including legacy note writeback,
can still delay dispatch and are not moved into children by this change.

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
