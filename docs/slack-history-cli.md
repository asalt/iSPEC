# Explicit Slack history and thread replies

Run from `/home/alex/tools/ispec-full/iSPEC`:

```bash
# Read recent channel history (default operation).
.venv/bin/python -m ispec.cli.slack_history --env-file ../.env.slack \
  --channel GC420B63V --limit 10

# Read the original 1660 thread, including replies.
.venv/bin/python -m ispec.cli.slack_history --env-file ../.env.slack \
  --channel GC420B63V --thread-ts 1791213934.178879

# Inspect a calendar day without posting or importing messages.
.venv/bin/python -m ispec.cli.slack_history --env-file ../.env.slack \
  --channel GC420B63V --date 2026-10-21 --timezone America/Chicago \
  --limit 100 --max-pages 5

# Preview exact reply text from stdin, without network access.
printf '%s' 'Your approved reply' | .venv/bin/python -m ispec.cli.slack_history \
  --channel GC420B63V --thread-ts 1791213934.178879 --stdin --dry-run
```

After approval, replace `--dry-run` with `--send` and include
`--env-file ../.env.slack` (or use the already configured bot environment).
Sending requires an explicit channel, parent timestamp, nonempty text, and
`--send`. Read the thread first to avoid duplicates, then read it back after
posting. The helper never retries a post automatically; a transport failure
may mean delivery is uncertain. Check history before retrying.

The route is **requesting parent → SSH/local iSPEC CLI helper → Slack Web API
as the existing iSPEC bot**. It does not call the iSPEC assistant or any other
agent, and does not use the ChatGPT Slack connector. The helper verifies and
returns the workspace, bot identity, and channel identity before a live read
or send. Dry runs show the proposed payload without verifying identity.

Configuration uses the existing iSPEC env-file loader and bot-token selection.
No credential arguments are accepted or printed. Known bot-token values and
Slack token-shaped strings are redacted in CLI output. The helper does not
install scopes, join channels, change access, or restart services.

Confirmed sends now return a separate `retention` result and write a deduplicated
`slack_outbound_message` receipt to the assistant database. `sent: true` with
`retention.retained: false` means delivery succeeded but local logging failed:
do not resend merely to repair logging. Reads still do not import history.

For an explicitly approved single-message receipt repair (not a send or backfill):

```bash
.venv/bin/python -m ispec.cli.slack_history --env-file ../.env.local --env-file ../.env.slack \
  --channel EXACT_CHANNEL_ID --thread-ts EXACT_MESSAGE_TS --record-outbound
```

This verifies bot/workspace/channel identity, verifies the actual DM peer when
applicable, fetches only the exact message, and refuses another author's message.
File references/captions returned on that message are recorded, not downloaded.
It does not retrieve neighboring file messages or infer the recipient from text.
Repeated imports deduplicate by workspace/channel plus message timestamp or file IDs.

Reads use `conversations.history` or `conversations.replies`. Both were
verified with the existing bot in private `proteomics_core` on October 5,
2026. Other channels and token types can have different access constraints.
Slack errors explain membership, scope, token-type, rate-limit, and thread
problems without exposing raw provider responses.

Pagination defaults to one page of ten messages. Increase `--max-pages`
and/or `--limit` (each bounded 1–100), or resume using the returned
`next_cursor` with `--cursor`. `has_more` signals incomplete history.
The helper stops if Slack repeats a cursor. Thread timestamps must be copied
exactly as strings, never rounded numbers.

`--date` selects local midnight through the next midnight in the selected IANA
timezone, including daylight-saving transitions. An explicitly supplied
`--timezone` on a valid read becomes the default for later reads, even from a
different working directory. It is stored in `~/.ispec/slack-history.json`
(or `$ISPEC_CONFIG_DIR/slack-history.json`); only the timezone is saved, not
messages or credentials. The initial default is `America/Chicago`. Pass another
`--timezone` to change it. Invalid arguments and send/dry-run operations do not
update this preference. A valid read saves the setting before network access,
so a later Slack outage does not lose the selection. Unreadable/invalid saved
settings require an explicit timezone rather than silently selecting another day.

A date scan covers channel history, **not all replies posted that day on older
threads**. Fetch selected threads separately with `--thread-ts`. Date mode cannot
send or import receipts, and does not generate project-note proposals itself.

Tests:
```bash
.venv/bin/python -m pytest tests/unit/cli/test_slack_history.py \
  tests/unit/cli/test_slack_module.py -q
```
Mutation tests use a fake API helper; no test sends to Slack.
