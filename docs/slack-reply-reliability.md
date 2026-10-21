# Slack reply failures and retention

## Ordinary bot-created DM threads

Socket Mode delivers text to `cli/slack.py`'s `message` handler. Bot-authored
messages and message subtypes are ignored. A threaded human reply first checks
the specialized artifact-review endpoint. Only a matched receipt takes that path;
an ordinary bot-created DM thread falls through to `/api/support/chat`.

Unthreaded DMs use the daily `slack:TEAM:CHANNEL:dm24:DATE` bucket; threaded DMs
use their thread session. Parent and message timestamps are included in request
metadata, and output targets the same Slack thread. Bounded locally retained
outgoing context is described in [outbound context](slack-outbound-context.md);
arbitrary Slack history is not automatically imported. Reading that needs the
existing explicit history/reply tools. Normal replies do not notify Alex, dot, or another
agent automatically. Artifact review matching stores a review event; it is not
proof that a human or originating agent has seen it.

## What the fallback promises

- Connection error, unstructured HTTP error, or missing retention proof when
  no receiver-side inbox capture exists:
  `I'm unable to reply right now. Please try again later.`
- A transport failure after committed receiver-side inbox capture, or structured
  `responseStatus=unavailable` with a valid committed backend receipt:
  `I've received and saved your reply, but my response is delayed. Please try again later.`

The backend records a failed model turn and explicitly commits the assistant DB
before returning its `support_message` retention receipt. A failed commit cannot
produce that acknowledgement. A queue timeout includes an `agent_command` receipt
only after the existing enqueue transaction committed. Record IDs come from the
backend, not model text, guessed message strings, or the presence of a Slack ACK.
Provider error details stay out of the fallback text. `automaticDelivery=false`
is explicit: neither receipt promises autonomous retries, Slack delivery, or a
human notification. Retained text also does not prove any requested tool write.

## Current limits

- The [receiver inbox](slack-inbox.md) now commits selected human text before ACK
  and backend dispatch. It remains available when the API/model is offline.
  A request timeout still leaves the outcome of any requested write ambiguous.
- Captured messages use cross-restart atomic claims; other routes retain their
  process-local fallback (4096 keys, one-hour lifetime). Neither offers exactly-
  once processing or automatic retry after an ambiguous attempt.
- Pending compare choices are also in memory and do not survive receiver restart.
- Receiver retention needs its local database, not the API or an LLM. Backend
  turn retention additionally needs the API/database. With no fresh supervisor
  heartbeat, chat runs inline. If its model is unavailable, the failed turn is
  stored and acknowledged as above. It is not automatically regenerated later.
- Queued generation requires a working supervisor and configured model. The
  receiver currently waits for the HTTP result; if that wait ends, a later stored
  result has no automatic Slack outbox delivery. Do not blindly re-enqueue it.
- `file_deleted` and `reaction_removed`, like existing ignored file/reaction
  events, are acknowledged quietly with no response, deletion, or forwarding.

## Verification and activation

Unit/handler tests replace Slack clients and backend transport with mocks. They
exercise backend offline/recovery, ordinary DM threads, receipt-aware failures,
duplicate and bot-event suppression, commit failure, and queue preservation.
The focused and nearby regression suite passed 58 tests; lint for the new
module/tests and `git diff --check` also passed. No live Slack test messages were
sent.

Use the root Makefile and the actual `ispec-dev` service session. Do not start
`dev-tmux`, a second receiver, the full supervisor, or a model as a side effect.
Existing background processes must be stopped before respawning their panes.
Host process control for this repair was handed to waypoint; sandbox PID
visibility is not reliable. Verify PIDs again on the host before signalling them.

### Deployed October 6, 2026

Waypoint gracefully stopped the verified old API PID 2371784 and Slack PID
3533120. Their old service windows closed on exit, so `dev-restart` could not
respawn them. Waypoint recreated only `ispec-dev:1` (backend, pane `%29`) and
`ispec-dev:3` (slack, pane `%30`), using the existing `make dev-backend` and
`make dev-slack` launch targets. Frontend, supervisor, and model were not started
or restarted as part of this repair.

Verification at approximately 19:22 America/Chicago (October 7, 00:22 UTC):

- API PID 3537260 is recorded in `.pids/api_server.pid` and
  `.pids/api_server.json`. `/status` returns HTTP 200 with `ok=true`;
  `/openapi.json` now advertises `ChatResponse.responseStatus` and `retention`,
  confirming the deployed receipt contract. Pane `%29` reports application
  startup complete and Uvicorn running.
- Slack PID 3537256 is host-verified by waypoint, with one established TLS
  connection and no duplicate receiver. `iSPEC/logs/ispec.log` records
  `auth_test` success and Socket Mode startup at 19:19:49 local. Pane `%30`
  shows `Bolt app is running!`; the installed Slack Bolt handler emits this
  banner after its `connect()` call returns. No message delivery test was sent.
- Read-only SQL against `iSPEC/data/ispec-agent.db` reports queued=0, running=0,
  succeeded=11860, failed=5064. `/home/alex/.ispec/db/ispec-agent.db` resolves to
  that same database. Pending counts remain unchanged from pre-restart checks;
  latest command update remains September 29. No commands were cleared/requeued.
- Configured vLLM `127.0.0.1:8000` remains unreachable. `.pids/supervisor.json`
  remains the stale July record; its saved `status=running` is not live evidence.
  Waypoint confirms neither model nor supervisor was started. Retention and
  generation/delivery limitations above still apply.

There were no startup ERROR/CRITICAL entries in the inspected application log.
One WARNING at 19:19:50 requires separate attention: the existing dev-admin
startup hook reported `reset_password=True`. `api/main.py` therefore reapplied
the configured admin password on startup; whether its value changed was not
inspected. This behavior predates the Slack patch. No credentials, auth settings,
or process environments were inspected/changed during this verification. Review
that startup policy separately before future restarts rather than silently
changing it during a Slack repair.

These PIDs/counts/log markers are point-in-time evidence, not persistent
configuration or proof of a subsequent user's message delivery.
