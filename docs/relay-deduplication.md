# Relay deduplication and retention

The local relay uses the existing agent database (`ispec-agent.db`). It does not
require Redis, another SQLite file, a master application writer, or a new loop.
Stage-only defaults and existing source/destination/confirmation policy remain.

## Identity and atomic enqueue

`relay_dedup_key` is created by the existing agent schema initialization. No
existing table columns or historical rows are rewritten. Each newly enqueued
relay has a source-scoped request-ID reservation and a payload reservation.
Schema initialization uses a short `BEGIN IMMEDIATE` transaction so independent
processes cannot race between checking for a table and creating it.
The first SQL statement reserves the request key with a unique constraint and
`INSERT ... ON CONFLICT DO NOTHING`. Both keys, the command, and its enqueue
event commit together. Failure rolls all of them back. Use a fresh short session
for enqueueing, not a transaction already holding unrelated work or a read snapshot.

Same request ID and payload return the original command. Reusing an ID with
different content fails (`relay_request_id_conflict`, HTTP 409). A new request
ID with the same protected payload also returns the original command, and its
request ID becomes an alias for that command. Duplicate calls do not create
another job or another enqueue event. API/CLI results include `deduplicated`,
`duplicate_match`, and `submitted_request_id`, alongside the canonical request.

Fingerprints include operation, mode, confirmation, source kind/ID, destination,
thread, Enter behavior, body, metadata, provenance, and attachment contents and
paths. Source cwd and request ID are excluded. This intentionally favors exact
duplicates, not fuzzy matching of equivalent destination aliases or prose.
Attachment paths become absolute at submission. Regular files up to 100 MiB
are hashed outside database transactions; larger/non-regular files are rejected
for this queued path. Missing/unreadable files can be staged but do not get a
content digest. Upload validation remains independent. Keep queued attachments
unchanged until delivery: a changed digest blocks execution. This is not an
immutable artifact store; concurrent file mutation during upload is not solved.

## Execution and uncertain outcomes

Queued live sends reserve a one-shot execution marker before invoking Slack or
tmux. The transaction closes before network/subprocess IO. Receipts are retained
in `agent_event` and mirrored on the command, so replay returns the recorded
outcome. A lost acknowledgement becomes `delivery_outcome=unknown`, `sent=null`,
not a claim that nothing happened. Partial file deliveries also stay protected.
A stale live relay command is finalized from its receipt, or marked unknown;
it is never blindly requeued. Stage-only recovery and other command types are
unchanged. This is local duplicate prevention, not exactly-once delivery.

Unknown/partial deliveries require operator investigation before any new send.
There is no automatic reconciliation or force-resend switch in this slice.
Direct calls to delivery adapters, generic queue insertions bypassing the relay
enqueue helper, old pre-upgrade jobs, and legacy Slack commands do not acquire
the new enqueue reservations. Queued relay jobs do have the execution marker.
No historical backfill or live send is performed by the schema upgrade.

## Retention and pruning

- Pending, unknown, partial, or missing-receipt operations do not expire.
- After a definite outcome, payload keys are eligible after 10 minutes and
  request-ID keys after 30 days. The command must also be terminal, with a
  definite outcome, before deletion is allowed.
- A new request can repeat an identical completed payload after its debounce
  window. An expired matching payload key is removed on enqueue, without
  waiting for a separate sweep.
- Request-ID protection lasts at least 30 days after the outcome, and longer
  until pruning. Do not automatically replay older requests after pruning;
  a deleted key no longer provides deduplication.
- Pruning deletes keys only. Jobs, receipt events, message bodies and remote IDs
  retain their existing audit retention. This command is not history compaction.

From the backend environment:

```sh
ispec agent relay-enqueue --kind slack_message --to alex --body 'Report ready' --request-id report-936-v1 --json
ispec agent relay-prune --json
ispec agent relay-prune --confirm --limit 1000 --json
```

The first command stages only. `relay-pdf` also accepts `--request-id`.
Pruning is dry-run-first, bounded, and manual for now; no frequent maintenance
tick or new environment variables were added.

## Verification

Unit tests cover identity conflicts, scope, changed files, rollback, expiry,
unknown/partial outcomes, no repeat execution, and receipt preservation.
Separate-process integration tests prove one command/event is created under
concurrent enqueue. API and CLI tests cover duplicate reporting and dry-run
pruning. All delivery tests use stubs, not external sends.
