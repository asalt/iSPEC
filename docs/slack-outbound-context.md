# Outgoing Slack receipts and reply context

## Scope

Ordinary human DMs and mentions already invoke `/api/support/chat` automatically.
Unmentioned channel replies remain quiet, and matched artifact-review threads keep
their existing record/acknowledge route. Bot events, file events and reactions do
not trigger assistant replies. There is no forwarding to Codex or tmux here.

Direct `ispec slack send/upload`, `ispec.cli.slack_history --send`, and the
top-level `slack-dm-notify` helper now record successful sends. This is not an
exactly-once outbound sender or a general Slack archive. Existing relay queue
receipts remain in the agent database; this slice does not change their executor.

## Storage and failures

`slack_outbound_message` lives in `ispec-assistant.db`, using the existing path
resolver (normally `~/.ispec/db/ispec-assistant.db`). Its fields include workspace,
channel, recipient when known, actual bot author when available, exact message and
parent timestamps, text/caption, attachment IDs/names/type/size, source, send time,
and receipt time. No attachment bytes, raw Slack responses, headers, upload URLs,
download URLs or credentials are retained. Timestamps remain strings for identity.

SQLite's unique workspace/channel/delivery-key constraint deduplicates concurrent
receipt retries atomically. File IDs identify uploads whose completion response
omits a share timestamp. Original records are preserved on duplicate ingestion.
Two actual sends with different Slack IDs are distinct, even if text is identical.

Backend CLIs write locally after Slack confirms success. The standalone helper
loads its existing canonical root/env configuration and posts the receipt to
`POST /api/agents/slack/outbound` using its existing service key. Both send and
logging outcomes are reported separately. A logging failure must not cause a
resend. No new credentials or grants are added. The new internal endpoints fail
closed if no service API key is configured.

There remains a crash window between Slack accepting a send and local logging.
Use the exact-message import in `slack-history-cli.md` to repair a known send;
there is no automatic history import. Missing workspace identity is reported as
unretained. Unknown author/recipient/share timestamps are not invented.

## Same-conversation context

Before normal chat submission, the receiver queries
`POST /api/agents/slack/outbound-context` with Slack's actual workspace, channel,
thread, and incoming message timestamp. Selection is deterministic:

- Explicit thread: only its parent and sent replies, within seven days.
- Unthreaded DM: only root messages in that exact DM, within 24 hours.
- No cross-workspace/channel/thread fallback, including empty or unknown threads.
- At most six receipts, 2,000 text characters each, about 6,000 JSON characters
  total; future sends relative to the incoming event are excluded.
- Attachment references are context, not evidence that the model read a file.

Threaded DMs now use their own existing thread-key session namespace. Root DMs
use daily UTC `dm24:DATE:root` sessions, deliberately not reusing older daily
sessions that mixed thread traffic. Existing history remains stored but is not
silently merged. This can reset pending conversational drafts at the transition.

Retrieved records and lookup status persist in the incoming support message's
`client_meta`. The prompt helper `assistant.slack.outbound_context.md` supplies
quoted user-role reference material under the normal prompt token budget, not
new system authority. Existing auth, tool selection and write guards remain
unchanged. These receipts are not approval inputs. Context lookup failure does
not prevent the normal reply path, but that reply may lack the earlier send.

The API must be reloaded for the new endpoints/prompt assembly, and the Slack
receiver for context lookup and thread-scoped sessions. Direct CLI invocations
use new code immediately. Neither supervisor nor model restart is required.
Do not activate overdue supervisor jobs to enable this feature.

## Local verification

Mocked tests cover concurrent/repeated receipts, text/file sends, missing logging
auth, logging failure after success, exact-message imports and peer verification,
workspace/channel/thread isolation, age/size bounds, API access rejection,
receiver fallback, and receipt context appearing in the normal model-call input.
They do not send Slack messages or exercise live model generation.

## Activation verification: 2026-10-09 UTC

Only backend and Slack were reloaded using their existing make entrypoints.
The old service windows closed on graceful termination; only `ispec-dev:1`
(backend) and `ispec-dev:3` (Slack) were recreated. API PID 2049041 returned
`/status` healthy; Slack PID 2054197 logged successful authentication/Socket Mode
startup and had an established TLS connection. vLLM PID 3669603 and its model
were not restarted. The supervisor stayed stopped; queued/running counts stayed
zero. No live test messages or model requests were sent.

The saved config enables dev-admin bootstrap/password reset. To avoid any auth
mutation during this reload, an ephemeral final env overlay
`/tmp/ispec-slack-reload.env` set both `ISPEC_DEV_DEFAULT_ADMIN=0` and
`ISPEC_DEV_ADMIN_RESET_PASSWORD=0`. No saved config or credential was changed.
The backend launch was:

```bash
make dev-backend ISPEC_ENV_FILES=".env .env.local .env.vllm .env.vllm.local .env.slack .env.slack.local /tmp/ispec-slack-reload.env"
```

Do not assume that temporary overlay survives a reboot, or that an unguarded
restart will avoid the preexisting bootstrap hook.

The separately authorized exact-message repair imported one bot-authored send
at `1791507704.187639` in `D0A9LQGJ1AM` as receipt 1. Live peer verification
identified Anna Malovannaya (`malovann`, `UC4QBPJTH`), not Alex. Its exact
message returned no file references; no neighboring file messages were fetched
or imported. This was a receipt repair, not another delivery.

Live context endpoint checks returned receipt 1 for that thread and its root DM,
and no records for a different thread or DM. Mocked normal-chat tests separately
prove that retrieved context reaches the generated model-call input.

The focused/adjacent suite passed 78 tests. A broader legacy staff-tool file has
eight failures reproduced on an isolated committed-HEAD snapshot: its fixtures
use viewer accounts for staff-only tools. Those permissions/tests were not
changed by this work.
