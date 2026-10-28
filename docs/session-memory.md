# Conversation memory: current implementation and activation boundary

The existing system already persists messages and conversation state in the
assistant SQLite database. Reuse it rather than introducing a second memory store.

## What exists

- `SupportSession` owns a conversation and, for authenticated web users, a user ID.
  The support route rejects another user's session. Slack conversation/thread
  identity remains a separate routing concern, not a shared user-wide profile.
- `api/routes/support.py` sends selected state as `CONTEXT v1`, including normalized
  `conversation_memory` and its last covered message ID. A newer rolling text
  summary can supplement it; an older covered summary is omitted.
- Compaction is queued after eight new older messages by default, retaining the
  latest six uncompacted. The requested message checkpoint suppresses repeated
  enqueue of that same range. This is persisted cache state, not a timer that
  summarizes unchanged conversations on every turn.
- `assistant/compaction.py` uses the tracked `assistant.compaction.distiller`
  prompt. The supervisor distills batches outside the database transaction and
  persists the summary, version, checkpoint and `SupportMemoryEvidence` links.
- `AuthUser.assistant_brief` is a separate explicit user preference/context field;
  it is not automatically populated by merging all of that user's chats.

## Not simply an enable switch

Compaction currently requires the vLLM provider and a worker executing the queued
`assistant_compact_session_memory_v1` command. API persistence and normal inline replies do
not require that worker. The staged Mariana read-only activation explicitly sets
`ISPEC_ASSISTANT_COMPACTION_ENABLED=0`; this review changes no live setting.
Starting the general supervisor may also activate schedules and writeback, so it
is not an appropriate implicit step to enable memory alone.

The schema bounds individual fields but permits many facts/tasks/entities; it is
not yet a deliberately tiny 400-800-token summary. The worker also needs stronger
concurrent/stale-result checks and recovery of abandoned requested checkpoints
before continuous use. A summary is fallible context, not an authority grant or
proof that a tool write succeeded. Current permissions and write receipts remain
the source of authorization and outcome truth.

## Next bounded slice

1. Keep memory keyed by owned conversation/thread. Do not silently combine DMs,
   channel threads or different users. Cross-chat preferences remain explicit.
2. Reuse the existing schema/checkpoints and prompt registry, with a measured
   total context budget. Put the small memory block before recent messages;
   preserve the latest user request at the end rather than repeating memory.
3. Run compaction through a dedicated command-only worker or equivalently scoped
   dispatcher, not by enabling unrelated schedules. Leave the last good cache
   available on failure; reject stale results and recover abandoned jobs.
4. Test restart persistence, ownership, stale completion, summarizer failure,
   correction of earlier mistaken facts and no approval carried between turns.
   Compare cache age, tokens and latency before separately activating production.

`test_session_memory_boundaries.py` exercises persisted enqueue checkpoints,
session-scoped selection, provider/disable gates and prompt-cache precedence.
Existing `test_support_chat_state.py` tests the full mocked chat context path.
These tests do not constitute a live compaction quality or concurrency benchmark.
