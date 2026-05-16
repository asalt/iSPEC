---
name: slack-tmux-bridge
description: Send review artifacts to Alex over Slack with provenance, then explicitly relay selected Slack replies back into allowlisted tmux/Codex panes.
---

# Slack Tmux Bridge

Use this skill when Alex explicitly asks Codex to send a report/artifact over Slack and keep a route back to the originating tmux pane.

## Mental Model

- iSPEC is the intermediary ledger/router.
- Codex is the actor that decides when to fetch pending Slack replies and when to relay one into tmux.
- Slack replies can create pending review events, but they should not automatically send keys.
- The bridge is intentionally split into a read-only fetch step and a confirmed write step.

## Expected Flow

1. Send the Slack message/file with artifact provenance recorded.
2. Alex replies in the Slack thread.
3. iSPEC records a pending Slack artifact reply if the thread matches a known artifact receipt.
4. Codex uses the read-only reply fetch tool to inspect pending replies.
5. Codex uses the tmux relay write tool only when Alex explicitly wants that reply forwarded.
6. The relay tool revalidates the concrete tmux pane target against the allowlist/blacklist before sending text.

## CLI Pattern

Prefer a root Slack text message first so the bridge has a stable thread id:

```bash
ispec slack send --to alex --text "Report is ready: <short context>"
```

Then upload the file into that thread and record the artifact receipt:

```bash
ispec slack upload \
  --to alex \
  --thread-ts "<message_ts_from_send>" \
  --file /abs/path/report.pdf \
  --title "Report" \
  --record-artifact-receipt \
  --origin-tmux-pane-id "%1" \
  --origin-tmux-target "936-1:fish" \
  --origin-tmux-allowlist-match "936-*" \
  --submit-allowed
```

The older one-way Slack DM helper is still fine for plain notifications, but a round trip needs a recorded receipt.

## Targeting Rules

- Discovery aliases may be friendly, such as `936-*`, `codex2-*`, or `936-1:fish`.
- The final send target should resolve to one concrete pane, preferably a tmux `pane_id` such as `%1`.
- Allowlist wildcard entries are for eligibility/discovery; they are not themselves final send targets.
- Blacklist entries win over allowlist entries.
- `press_enter=true` is allowed only as an explicit relay parameter, analogous to approving a project-comment save.

## Tool Split

- Read-only: `assistant_list_slack_artifact_replies`
- Write: `assistant_relay_slack_reply_to_tmux`

The write tool requires `confirm=true`. It sends literal text first, then sends Enter/C-m only when `press_enter=true`.

## Long-Running Task Pattern

For multi-minute jobs, avoid many tiny polling sleeps.

1. Start the job with durable state: log file, PID file when possible, known output paths, and optional sentinel file.
2. Use a tuned sleep interval before inspection. For multi-minute jobs, 120-300 seconds is usually better than repeated 10-second checks.
3. Inspect concrete state after sleeping: PID alive, log mtime/size, exit/sentinel state, and cheap output validation.
4. If complete, optionally relay a short wake message into the originating Codex pane using the tmux relay/write path.
5. If still running, summarize state and schedule/check again later rather than thrashing.

This is not a permission shortcut. The wake is just another explicit tmux write action with the same allowlist/blacklist and concrete-target rules.
