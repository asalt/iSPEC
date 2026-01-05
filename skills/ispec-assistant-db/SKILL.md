---
name: ispec-assistant-db
description: Inspect and export iSPEC assistant chat logs from the assistant SQLite database (support_session/support_message) for debugging and training data preparation.
---

# iSPEC Assistant DB

This skill helps inspect and export the assistant chat log SQLite DB used by the `/api/support/*` endpoints.

## DB location

Default resolution order matches the backend:

1) `ISPEC_ASSISTANT_DB_PATH` (file path or `sqlite:///...` URI)
2) `iSPEC/data/ispec-assistant.db` (repo-local default used by this skill script)

## Quick commands

List recent sessions:

`python iSPEC/skills/ispec-assistant-db/scripts/assistant_db.py sessions --limit 20`

Show recent messages for a session:

`python iSPEC/skills/ispec-assistant-db/scripts/assistant_db.py messages --session-id <session-id> --limit 50`

Export full conversations as JSONL (one session per line):

`python iSPEC/skills/ispec-assistant-db/scripts/assistant_db.py export-jsonl --out assistant-sessions.jsonl`

Export rated assistant replies (thumbs up/down) as JSONL for preference/SFT workflows:

`python iSPEC/skills/ispec-assistant-db/scripts/assistant_db.py export-feedback-jsonl --out assistant-feedback.jsonl`

## Notes

- The scripts open the DB read-only and do not write or migrate schema.
- Treat exported logs as potentially sensitive (PII / internal context).
