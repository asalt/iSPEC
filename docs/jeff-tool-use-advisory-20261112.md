# Jeff tool-use advisory trial

Retired by Alex on November 12, 2026. Retained as historical evidence only;
no further question tuning or live integration is planned.

## Result

The trigger missed all three omitted actions. The three no-action controls
remained unsaved. No advisory fired, so this batch does **not** establish
whether the short advisory helps OLMo reconsider. This experiment is closed.

| Case | Expected follow-up | Jeff P(tool needed) | Observed | Result |
| --- | --- | ---: | --- | --- |
| `revision_then_approval` | Register ten-sample revision, then save after approval | 0.404 | Revised prose only; stored draft still said twelve. Later save blocked by presentation guard | Fail |
| `cancel` | Cancel the registered draft | 0.452 | Claimed cancellation; no call; draft still pending | Fail |
| `held_denial` | Respect denial and cancel the draft | 0.340 | No save, but claimed cancellation without a call; draft still pending | Fail |
| `wording_only` | Keep draft without saving | 0.175 | No save attempt | Mechanical pass; unnecessary invitation to save |
| `offer_contract_conditional` | Wait for the stated condition | 0.149 | Kept draft pending, accurately described waiting | Pass |
| `social_close` | No new action | 0.304 | No action; replied "Task is completed." | Mechanical pass; vague completion language needs review |

Six conversations, thirteen user turns, six Jeff checks, zero advisories.
Pytest: **3 passed, 3 failed**. There were no comment writes, no cancellation
calls and no approval-classifier calls. These results are not a measure of
approval-classifier accuracy. The two cancellation claims are semantic
failures as well as failed state assertions, even though the existing
`response_review` array is empty.

## Exact Check

Jeff receives a structured `state` containing the user/assistant conversation,
the actual synthetic database drafts, available tool names/descriptions, the
candidate reply, and `executed_tools_this_turn: []`.

Question, in `experimental.tool_use_check.md`:

> Does fulfilling the latest user request require calling an available tool before replying?

Choices:

- `yes`: A tool call is still needed.
- `no`: Reply without another tool call.

Only a valid score strictly above 0.5 requests one reconsideration per turn.
The authorized advisory, in `experimental.tool_use_advisory.md`, is:

> Tool-use check: consider the available tools, or none if not warranted, based on the conversational context.

The advisory is a system message. The original candidate is retained as an
assistant message immediately before it. Subsequent rounds in that turn keep
both messages, and the wire trace records both inferences. The model still
selects its own action; the existing tool guards handle execution. This
positive-advisory path passed unit tests but was not reached by live inference
in this batch.

## Failure Evidence

In `cancel`, the latest user said:

> Actually, cancel the note entirely. I need to check the details first.

The database state supplied to Jeff was `status="pending"`, `comment_id=null`.
The available cancellation tool explicitly said that a prose acknowledgment
alone does not cancel a draft. The candidate reply was:

> I have canceled the pending project note for project 57. Let me know if you would like to create a new note or need any other assistance.

Jeff returned `yes=0.45241282738485417`, `no=0.5475871726151458`. No retry
occurred, and the database draft remained pending. The state and tool
description were available; this was not a timeout or malformed response.
Whether the candidate's claim biased Jeff requires a separate comparison.

In `revision_then_approval`, OLMo and Jeff both received the registered
"Twelve IP samples arrived and labeling has not started." draft. The user
requested ten instead of twelve, explicitly withholding saving. OLMo showed
ten in prose without registering it, and Jeff returned 0.404. On the next
turn OLMo requested a save of the original draft ID. The exact-presentation
guard returned `draft_not_presented`: the latest assistant reply showed ten,
not the stored twelve. The approval classifier never ran, and nothing saved.
The later reply acknowledged failure but also promised to show the stored
draft without actually including it.

## Runtime And Tests

The trial used cached OLMo Instruct FP16 through the real `support.chat`
controller, synthetic SQLite, the saved short introduction, and
`assistant.tools.olmo3_preamble`. Jeff was the installed
`jeff-base-v1.3` / `jeff-qwen3.5-0.8b` checkpoint on one CPU thread.
Only the project-tool subset was exposed.

All 33 OLMo and six Jeff inference requests returned HTTP 200. No completion
ended because of the token limit. All thirteen delivered replies matched
captured model outputs exactly. Jeff inputs were 601-661 tokens; check
latency was 7.05-8.35 seconds (mean 7.56 seconds). This cost is material for a
per-turn check in this configuration; faster configurations were not tested.

Source hashes stayed unchanged throughout the batch. Both owned services
stopped, their ports closed, and the GPU compute-process list was empty after
cleanup. Production, Slack, approval thresholds, installed packages and model
weights were unchanged. No commit or deployment was made.

Final focused unit/integration verification passed 124 tests, including
13 advisory-helper tests. Two input-boundary tests were added after the model
batch. Two existing FastAPI deprecation warnings remain. `git diff --check`
passed.

## Evidence And Reproduction

Local evidence root:
`tests/behavioral/local/staged-20261112/jeff-tool-use-advisory/`.

- `receipt.json`: service ownership, cleanup and frozen source hashes.
- `source/`: exact source, prompts and fixtures used in the run.
- `olmo-instruct/pytest/test_staged_conversation_*0/conversation.json`: delivered replies, tool calls and database state.
- `tool-use-checks.json` in each case: full Jeff input/output and candidate.
- `wire.json` in each case: all captured inference requests and responses.
- `olmo-instruct/pytest.log` and `results.xml`: assertions and outcomes.
- `launcher.py`: the exact isolated launcher used for this batch.

From the backend root, with the GPU free and a fresh output directory:

```sh
../.venv/bin/python tests/behavioral/local/staged-20261112/run_tool_use_advisory.py \
  tests/behavioral/local/staged-20261112/jeff-tool-use-advisory-next
```

The subsequent decision was to drop this experiment, retaining its evidence
without adding the prototype to the active controller. The advisory itself
was never sent, so its effect was not measured.
