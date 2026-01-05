# Assistant training plan (WIP)

This doc outlines a lightweight path from “prompted local model” to an iSPEC-specific assistant that reliably follows iSPEC conventions, avoids hallucinating IDs/records, and gets better over time via logged feedback.

## Goals

- **Become iSPEC**: speak as the iSPEC support assistant (consistent persona + tone).
- **Be database-grounded**: treat the backend-provided `CONTEXT` JSON as authoritative.
- **Be operationally useful**: help staff navigate UI routes, explain statuses, and suggest next actions.
- **Reduce hallucinations**: never invent project IDs, titles, people, dates, or outcomes.

## Phase 0: prompt + logging hygiene (now)

1) Iterate the system prompt via env:
   - `ISPEC_ASSISTANT_SYSTEM_PROMPT` (full override)
   - `ISPEC_ASSISTANT_SYSTEM_PROMPT_EXTRA` (append-only)
2) Ensure we capture:
   - chat transcripts (`support_message`)
   - per-session state (`support_session.state_json`)
   - thumbs up/down + optional comment (`feedback`, `feedback_note`, `feedback_meta_json`)
3) Keep prompts stable for evaluation runs (version prompts, keep notes when changing them).

## Phase 1: data export + curation

Primary data source is the assistant SQLite DB (example: `iSPEC/data/ispec-assistant.db`).

- Export conversations as JSONL:
  - `python iSPEC/skills/ispec-assistant-db/scripts/assistant_db.py export-jsonl --out assistant-sessions.jsonl`
- Export rated assistant replies as JSONL:
  - `python iSPEC/skills/ispec-assistant-db/scripts/assistant_db.py export-feedback-jsonl --out assistant-feedback.jsonl`

Curation steps (recommended):

- Remove/blur sensitive fields (PII, internal notes, any secrets).
- Drop sessions with broken context / errors / empty messages.
- Add a small “golden set” of ~50–200 iSPEC tasks you care about (for regression testing).

## Phase 2: supervised fine-tuning (SFT)

Start with SFT on curated conversations:

- Format: chat-style `messages` (system/user/assistant), where the assistant outputs only the final answer.
- Include iSPEC “style rules” and non-hallucination constraints in the system message.
- Prefer high-signal examples (staff workflows, common UI questions, correct use of IDs/titles).

## Phase 3: preference tuning (feedback → DPO/RLAIF)

Thumbs up/down is not a full preference pair by itself, but it’s still useful:

- **Create pairs** by rewriting downvoted answers into a better “chosen” answer (human or LLM-assisted), then train with DPO (chosen vs rejected).
- Alternatively, use RLAIF-style pipelines where you generate multiple candidates and pick the best with rules/graders.

Keep a strict eval gate: new checkpoints must not regress on the golden set.

## Phase 4: evaluation + rollout

- Offline eval:
  - golden set task accuracy + “no hallucinated IDs” checks
  - response length + clarity
  - safety constraints (no secrets, no unsafe instructions)
- Rollout:
  - track model + prompt version
  - monitor thumbs down rate and common failure tags

## Notes on base model

If serving `allenai/Llama-3.1-Tulu-3-8B`, the default “Tulu 3” system prompt from demos is not required; iSPEC should provide its own system prompt and (later) fine-tune to reduce reliance on prompt steering.
