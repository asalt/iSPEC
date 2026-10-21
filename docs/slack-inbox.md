# Durable Slack inbox

The Socket Mode receiver stores selected human text events in the existing
assistant SQLite database (`slack_inbox_message`) **before** acknowledging and
dispatching them. No polling loop, model call, supervisor job, Slack send, or
project-comment write is needed for capture. The normal DM/@mention handlers
continue to answer questions as before.

## Scope and configuration

DMs, group DMs, and app mentions are captured. Passive channel capture is opt-in:
copy `configs/slack-inbox.example.json` to an ignored local configuration and set
`ISPEC_SLACK_INBOX_CONFIG` to its absolute path. Each entry has an exact workspace
`team_id` and `channel_id`. This is an **inbound retention list**, separate from
outbound destination permissions and staff/project authorization.

Slack must already deliver the appropriate events to the bot. A configured ID
does not subscribe the app, add scopes, establish membership, or backfill missed
messages. Private channels need the corresponding existing event subscription.
No periodic history fetch is introduced; an explicitly requested bounded history
read remains a separate tool. Restart only the existing receiver after changing
its configuration. Do not start a second receiver or the entire dev stack.

Stored text is capped at 40,000 characters with an explicit truncation flag.
Attachment bytes, file-only messages, reactions, deletes, and bot/self messages
are excluded. Edits with a revision timestamp create a separate row; the original
evidence remains. Edits are retained for review, not rerouted as new instructions.

## State and deduplication

- `received_at`: text and workspace/channel/user/thread/message IDs committed.
- `processing_started_at`: one handler attempt claimed atomically in SQLite.
- `handled_at`: routing attempt ended, including an ignored passive message or
  failure. **Not** proof of a successful answer, Slack delivery, or tool write.
- `error_type`: bounded error category when known, not private exception details.
- `reviewed_at`, `reviewed_by`: explicit local review, independent of handling.
  Reviewing a message does not approve a write or send a response.

Uniqueness is workspace + channel + message timestamp + revision timestamp.
Duplicate events share a row, including mention/message deliveries. Claiming
survives restart. After an interrupted or ambiguous attempt, leave it for review
rather than automatically replaying a possibly completed write. A new user
message can still receive a normal answer. This is durable evidence and at-most-
one dispatch attempt, **not exactly-once processing or an automatic retry queue**.

Storage failure returns non-200 before dispatch; the Socket Mode adapter does
not ACK it. Slack's finite retry behavior is not a guaranteed recovery system.
Capture uses a one-second SQLite busy timeout without changing the pooled
connection's timeout for other callers. Failed outcome updates are logged by
type and leave incomplete evidence; they do not erase the original message.

## Local review

### Same-thread reply context

For a threaded DM or channel mention, the receiver now attaches the earlier
human messages already retained for that exact workspace/channel/thread. The
current receipt bounds the snapshot: later messages, later edits, and records
received after it are excluded. The newest eligible revision of each message is
used. Unthreaded DMs retain the existing session-history behavior.

This is a bounded local transcript, **not a claim to have fetched the full Slack
thread**. It keeps the root and recent replies, up to 24 messages, 2,000 characters
per message and an 8,000-character envelope. Missing roots and truncation are
explicit. Capture gaps, deletes, file contents, and messages from before capture
started are not recovered. Confirmed bot sends still use the separate outbound
receipt context. No additional Slack API call or model inference pass is added.

`meta.slack_thread_context` holds the snapshot. The backend validates its scope
against `meta.slack` and quotes it in the existing turn-decision, tool-router and
reply prompts. It never concatenates it into the latest user request, installs
it as an assistant confirmation, or fetches database contents on the basis of
caller-supplied metadata. Existing write/authorization checks remain separate.
The recent support exchange (especially a pending draft) gets budget priority,
then thread context, then older support history. `context.slack_thread_context`
and each `llm_trace[].slack_thread_context` record included inbox IDs, count,
parent presence, truncation and budget omission. The latter records actual
per-round inclusion, not just availability to the earlier router.

The synthetic `tests/behavioral/data/slack_reported_thread.json` replays the
1131 prof/BioID discussion without real user identities. Behavioral tests check
capture through routing and reply generation, including an old quoted approval
followed by a current social reply. Model responses and all writes are stubbed.
An optional local-model router check never dispatches tools or sends to Slack:

```sh
ISPEC_RUN_VLLM_TESTS=1 .venv/bin/python -m pytest \
  tests/integration/test_slack_thread_router_vllm_live.py -q
```

Receiver and backend processes need the updated source loaded; no supervisor,
new subscriptions, model change or permission change is required. This is
context repair, not the separate autonomous note-proposal workflow.

October 19 verification: a read-only lookup of the reported 1131 thread recovered
all seven prior human messages, including the project number and BioID discussion,
without truncation. Inference-only checks against the already loaded Tulu 3 8B
model initially reproduced the wrong assistant-tool selection despite context.
After clarifying the central router prompt, project-note and project-status
requests selected `projects`, while an explicit chat-history query selected
`assistant`. The full-catalog checks took 0.77-1.07 seconds each; this measures
router inference, not end-to-end reply/save latency. No tool was dispatched.

Partially activated October 19 at 19:35 America/Chicago by restarting only
`backend slack` in `ispecfull` using the established Make target. Before restarting, both
`ISPEC_DEV_DEFAULT_ADMIN` and `ISPEC_DEV_ADMIN_RESET_PASSWORD` were disabled in
the local deployment config, preventing bootstrap password/role changes. Tracked
templates now default both flags off; the recovery helper documents their
persistent, not one-shot, behavior. No credential values were changed.

The new API PID was 1268000 (`/status` returned 200), and Slack PID 1267987 passed
`auth_test`, started Socket Mode, and had an established TLS connection. Both
previous processes exited. The 18 inbox records and queue counts were unchanged:
no active commands in the agent DB; the assistant DB still had 19 old compaction
commands queued and one old orchestrator tick marked running. These historical
rows were not replayed or cleaned up. Supervisor, frontend and Tulu 3 8B were
left running unchanged. No test Slack messages or project writes were made;
real conversational behavior after activation still needs user validation.

Correction after the 19:52 queued turn: restarting those two services was not
sufficient. `_run_support_chat_turn` calls the support route inside the supervisor
process; its previously imported Python modules do not reload when the API restarts.
Command 76817 retained eight correct thread messages, but response 1351 lacked
thread-inclusion metadata and invented project 1234/marketing details. Its write
outcome was `not_attempted`. The supervisor had been running since October 14.
Reloading that worker is also necessary; do not mistake persisted context for
evidence that it reached generation.

The opt-in `tests/behavioral/test_slack_thread_queue_vllm_live.py` now exercises
that actual queue entrypoint in a fresh process with sandbox databases, a
synthetic/anonymized seven-message thread, and intercepted tools. Only local
model inference is allowed. Two initial cases passed the narrow context/target
checks (9.8 and 15.5 seconds), not a comprehensive semantic-quality benchmark.
The first drafted the 1131/BioID update for confirmation. The second proposed a
1131 write, intercepted without execution, but incorrectly phrased the project
number as "1131 sets" and gave a confused explanation after the intentional
block. Review remains required. No live supervisor/model restart or Slack send
was performed for this replay. It uses fresh conversation state, not the now
contaminated live conversation history; shadow response-contract/compaction
work is disabled to isolate the queue/context path.

### Inbox commands

From the deploy checkout, using its normal installed CLI and configuration:

```sh
ispec --env-file .env.local slack inbox list
ispec --env-file .env.local slack inbox list --channel CHANNEL_ID --before-id 100
ispec --env-file .env.local slack inbox show 42
ispec --env-file .env.local slack inbox review 42 --by alex
```

`list` defaults to unreviewed, newest first, at most 50 records; use `--all` or
`--limit` (1-200). `--before-id` pages older records. For an isolated database use
`slack inbox --database /absolute/path/assistant.db list`. These commands do not
need Slack credentials or network access. OS access to the database is the local
review boundary; `--by` is reviewer attribution, not authenticated staff identity.
Do not expose this CLI as an unauthenticated web endpoint.

There is no UI, automatic pruning, broad history import, or automatic project-note
creation in this slice. The table grows with captured messages, not idle ticks.
Keep it in the existing assistant-DB backup policy; choose retention explicitly
before expanding to many channels. Received/reviewed counts are not proposals,
approvals, or comment IDs. A later note workflow must bind the source revision,
actual authorized author/approver, project, draft, and successful write result.

## Verification

Unit tests cover commit failure, concurrent dedup/claim, edits, bounded capture,
safe outcome failures and explicit review. Real Bolt dispatch integration checks
commit-before-listener and non-200 storage failure with network calls forbidden.
Existing handler tests exercise backend offline/recovery and ordinary project
questions. The behavioral sandbox follows passive capture, backend failure,
restart, review and a fresh question. All tests use temporary databases and fake
transports; no Slack messages or project comments are created by these tests.

## Local activation, October 17, 2026

At 02:35 America/Chicago, only `ispecfull:slack` (pane `%6`) was restarted with
`make dev-restart DEV_TMUX_SESSION=ispecfull DEV_RESTART_SERVICES=slack`.
Startup reports the durable inbox ready with one passive channel, successful
bot authentication, and Socket Mode connected. Existing read-only Slack metadata
verified that the bot belongs to the private `proteomics_core` channel. Its exact
workspace/channel pair is in ignored `configs/slack-inbox.local.json`, selected
by the deploy `.env.local`. No auth, scopes, or subscriptions were changed.

Host process verification confirmed the prior receiver exited and the replacement
is PID 2519430. The table was created in `iSPEC/data/ispec-assistant.db`; initial
received/handled/reviewed counts were all zero. Agent queued/running counts stayed
zero. API `/status` remained HTTP 200, `ok=true`. These are point-in-time checks,
not proof of future delivery. No live test message was sent, so receipt of a new
ordinary channel event still needs observation. Backend, supervisor and model
were not restarted. Automatic note proposals/approvals remain unimplemented.
