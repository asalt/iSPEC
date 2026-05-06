---
name: ispec-slack-wire
description: Send explicit one-off Slack text messages or one explicit local file upload from local iSPEC/Codex workflows using configured local Slack env vars. Use only when Alex explicitly asks Codex to send a Slack notification, upload a produced file, or test the Slack wire; not for autonomous notifications or background posting.
---

# iSPEC Slack Wire

Use this skill only for explicit user-requested Slack text sends or one explicit
file upload. Do not send Slack messages autonomously, on inferred intent, or as
part of routine status updates unless Alex asked for a Slack send in the current
task.

## Preferred Command

Use the bundled wrapper from the repo root or any working directory:

```bash
bash iSPEC/skills/ispec-slack-wire/scripts/send-to-alex.sh --text "message"
```

To upload one explicit local file:

```bash
bash iSPEC/skills/ispec-slack-wire/scripts/send-to-alex.sh --file report.pdf --text "Report is ready"
```

Use `--dry-run` first if recipient or file resolution is uncertain.

## Direct CLI

Prefer the iSPEC CLI:

```bash
iSPEC/.venv/bin/ispec --env-file .env.local --env-file .env.slack slack send --to alex --text "message"
```

Useful variants:

```bash
iSPEC/.venv/bin/ispec --env-file .env.local --env-file .env.slack slack send --channel "$ISPEC_ASSISTANT_STAFF_SLACK_CHANNEL" --text "message"
iSPEC/.venv/bin/ispec --env-file .env.local --env-file .env.slack slack send --user-id U0123456789 --text "message"
iSPEC/.venv/bin/ispec --env-file .env.local --env-file .env.slack slack send --email alex@example.org --text "message"
```

Use `--dry-run` first if recipient resolution is uncertain.

To upload one explicit local file:

```bash
iSPEC/.venv/bin/ispec --env-file .env.local --env-file .env.slack slack upload --to alex --file report.pdf --text "Report is ready"
```

Use upload only for files Alex explicitly asked to send or files produced as
the requested output of the current Codex task.

## Scope

- Text and one explicit local file path, normally to Alex.
- File upload uses Slack's external upload flow and requires `files:write` on
  the configured Slack bot token.
- Auth comes from local env files such as `.env.slack` or `.env.local`.
- Recipient aliases may be configured with `ISPEC_SLACK_RECIPIENTS_JSON` or
  `ISPEC_SLACK_DM_<ALIAS>_{CHANNEL,USER_ID,EMAIL}`.
- Assistant/supervisor-originated Slack notifications use a separate explicit
  destination allowlist via `ISPEC_ASSISTANT_SLACK_DESTINATIONS_JSON`. That
  allowlist can include named DM aliases and named channel aliases, but the
  assistant should choose aliases only, not arbitrary Slack IDs.

## Safety

- Treat this as a manual convenience wire, not a scheduler.
- Keep messages compact and factual.
- Do not upload env files, databases, secrets, or unrelated local files.
- Prefer generated reports/figures/docs from the current task: PDF, PNG, JPG,
  TXT, CSV, TSV, or similar harmless artifacts.
- If Slack returns an error, report it; do not retry in a loop.
