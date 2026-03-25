---
name: ispec-support-chat
description: Replay or script messages against the local iSPEC `/api/support/chat` endpoint from the command line with the repo env files and API key resolved automatically. Use when Codex or a human wants to verify live assistant behavior after a restart, test tool-calling flows like project-note creation, or send support-chat prompts without using the GUI.
---

# iSPEC Support Chat

Use the repo root as the working directory.

Prefer the wrapper script:

```bash
python3 scripts/support_chat.py "how many current projects do we have?"
```

For humans, the Make target is the shortest entry point:

```bash
make support-chat SUPPORT_CHAT_MESSAGE='how many current projects do we have?'
```

## Useful Flags

Print the full response JSON:

```bash
python3 scripts/support_chat.py --json "tell me about project 1531"
```

Reuse a session:

```bash
python3 scripts/support_chat.py --session-id debug:notes "make a note on project 1531 that data is regrouped"
```

Bypass the queued supervisor path and force inline handling:

```bash
python3 scripts/support_chat.py --inline "tell me about project 1531"
```

Read the prompt from stdin for long messages:

```bash
printf '%s\n' 'make a note on project 1531 that data is regrouped' | python3 scripts/support_chat.py
```

## Notes

- The script reads `.env`, `.env.local`, `.env.vllm`, `.env.vllm.local`, `.env.slack`, and `.env.slack.local`, then lets current shell env vars override them.
- API URL resolution defaults to `ISPEC_API_URL` when set, otherwise `http://127.0.0.1:$ISPEC_PORT`, then appends `/api/support/chat`.
- API key resolution prefers `--api-key`, then `ISPEC_API_KEY`, then `ISPEC_SLACK_API_KEY`.
- Writes are real. For note/comment flows, use a throwaway project or be intentional about the target project number.
- When debugging a live turn, pair this skill with `ispec-assistant-db` to inspect `support_session` / `support_message` rows after the request.
