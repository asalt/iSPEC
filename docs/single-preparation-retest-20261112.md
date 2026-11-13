# Single-preparation behavioral retest

## Follow-up: contract correction

The extra `review_mode` requirement and regex-driven write-claim retry were
removed. Preparation now takes only `project_id` and `comment`. The model
shows the returned draft and asks its question; the controller records that
presentation. A later model-requested save evaluates approval using the actual
conversation and existing policy/ticket settings.

The original unit tests enforced the unwanted mode distinction; they were
changed to test the intended two-argument preparation and later save. The
behavioral harness now explicitly labels results as controller integration
and records model requests, executor refusals, classifier checks and writes
separately. A failure of the whole conversation is not a standalone model
approval score.

One frozen follow-up batch used the five original cases plus explicit denial
and conditional approval. Evidence:
`tests/behavioral/local/staged-20261112/single-preparation-no-mode-retest/`.

| Case | OLMo Instruct observed | Tulu observed |
| --- | --- | --- |
| Explicit save | Saved the displayed draft; classifier approve 0.99; truthful success reply | Premature save blocked; first reply omitted note; later save also blocked |
| Natural approval | Saved the displayed draft; classifier approve 0.95; returned actual comment ID | Called save on approval, but prior reply omitted the note; blocked before classifier |
| Wording-only agreement | Kept draft; no save call | Printed tool instructions instead of preparing; later called save with placeholder ID; blocked |
| Revision then approval | Described correction, registered it only on the following approval turn, then asked again; no save | Described correction without registering it; old-draft save blocked |
| Cancellation | Claimed cancellation without calling the tool; draft remained pending | Called cancellation successfully; full case still fails because initial note was not shown |
| Explicit denial/cancel | No save; claimed cancellation without calling the tool | No save; claimed cancellation without calling the tool; initial note also omitted |
| Conditional approval | Left draft pending; no save call | No save; invented a draft ID without preparing a draft |

OLMo has 4/7 full integration passes; Tulu has 0/7. On the original five cases,
OLMo passes explicit save, natural approval and wording-only agreement; its
cancellation behavior regressed in this single run. Tulu's successful
cancellation action is separate from its failed full conversation test.

The scripted harness sends each follow-up regardless of the preceding model
reply. When it says "save that exact note" after a reply that omitted the note,
the intended approval scenario has not been reached. Those results diagnose
presentation/sequence failures, not inability to interpret an adequately
contextualized approval. A subsequent harness addition records
`prior_draft_presented` explicitly; the saved batch still has its original
evidence schema.

There were two actual comment writes, both approved, and no unapproved writes.
Only those two cases reached the classifier; negative classifier accuracy is
not established by this batch. All 83 inference requests returned HTTP 200;
none ended with `length`. All 30 delivered replies matched model outputs
after stripping Tulu's `FINAL:` prefix. No write-claim retry ran.

Both model servers exited with code 0, port 18030 closed, and the host GPU
compute-process list was empty. Source hashes remained unchanged during the
batch. Production and permission settings were untouched.

A further controller recovery fix, unit-tested after this batch, records a
later reply showing an already-registered draft even when no new preparation
call occurred in that turn. It preserves the draft ID, binds the actual reply
and still requires a later approval check. This recovery change and a small
catalog wording correction have not had a second model batch; the evidence
above belongs to the frozen source snapshot in the output directory.

The controller defects are corrected; remaining presentation, revision and
cancellation behavior still needs work. No model promotion or deployment.
Final focused verification: 181 unit/integration tests passed, including the
later-presentation recovery and harness evidence tests; two existing FastAPI
deprecation warnings. `git diff --check` passed.

## Initial batch before the correction

Date: 2026-11-12. Recommendation: keep this workflow in development.

Five existing regression conversations per model ran through `support.chat`
with real model inference and synthetic SQLite. Tulu passed 0/5; OLMo Instruct
passed 2/5. Neither saved a comment, including the three intended approval
paths per model. No approval classifier ran: every save attempt stopped at an
earlier draft/presentation/offer check.

## Results

| Case | Expected | Tulu | OLMo Instruct |
| --- | --- | --- | --- |
| Explicit save | Present draft, then save on explicit approval | Fail: prepared `draft_only`; later save returned `save_offer_required` | Fail: prepared `draft_only`, then created a new `save_approval` draft and tried saving it in the same turn; `draft_not_presented` |
| Natural approval | Save on "put this version in the project history" | Fail: first reply omitted stored text; three save attempts returned `draft_not_presented` | Fail: showed text but chose `draft_only`; save returned `save_offer_required` |
| Wording-only agreement | Keep draft; no save attempt | Fail: attempted save, then re-prepared for saving and tried a literal placeholder ID; no writes | Pass: showed draft and made no save attempt |
| Revision then approval | Register and show correction, then save it | Fail: revised prose without registering the correction; later save of old ID blocked | Fail: promised revision without executing it; prepared correction only after approval and tried immediate save; blocked |
| Cancellation | Cancel registered draft and accurately acknowledge | Fail: claimed cancellation without a cancellation call; draft remained pending | Pass: cancellation tool succeeded, draft became cancelled, reply matched |

These are mechanical results plus manual inspection of replies and tool/state
evidence. OLMo's wording-only reply still unnecessarily pivots toward saving:
"I will register this as a pending draft ... and await your approval to save it."
The draft was already registered. That is a reply-quality issue despite the
mechanical pass.

## What the traces show

### Review mode and spoken intent disagree

Both models chose `draft_only` for the explicit-save and natural-approval cases.
For example, OLMo received:

> Prepare a project note for project 57: eleven saliva samples arrived; peptide
> coverage review is pending. Show me the draft before saving.

It registered the text with `review_mode="draft_only"`, displayed it exactly,
and asked whether to submit it for saving. After "Yes, save that exact note
now," it prepared identical text with `save_approval`, creating a new ID, then
tried to save that new draft immediately. The guard requires a prior assistant
presentation bound to that ID; the old presentation did not qualify.

The tool instruction was present in the captured request:

> Use save_approval when asking to save, or draft_only for wording review.

This points to a contract/instruction problem worth addressing first: "review
before saving" is being treated as wording-only preparation. It does not show
that the classifier rejected genuine approval. Changing its threshold would
not affect these failures. The relative contributions of instruction wording,
mode naming and model behavior have not been isolated.

### Revision and cancellation need actual calls

Tulu told the user the revised text and reused the old ID, while the stored
draft still said twelve rather than ten samples. OLMo promised to revise but
made no call on that turn. Neither completed the required revision/presentation
sequence before the later approval.

Tulu also said "The project comment draft has been successfully cancelled"
without calling cancellation. In natural approval, it referred to an automatic
cancellation tool having failed, although no cancellation call occurred.
These are unsupported action narratives even though neither claimed a
successful comment save in its delivered final replies.

### Existing write-claim retry has a false positive

In Tulu explicit-save inference 3, `_WRITE_SUCCESS_PASSIVE_RE` matched from
"successfully" in "registered successfully" through "updated" in "provide
the updated comment text." It then injected a system retry accusing the model
of claiming a successful write. That sentence described a successful draft
registration and a possible revision, not a saved project comment.

This is a controller false positive, separate from the initial wrong mode.
The original reply also exposed tool-call syntax to the user, which is another
quality problem. Both the original and retry inference are retained in the
wire trace. No controller logic was changed during this run.

## Continuity and runtime checks

- All 22 delivered replies matched captured model outputs, allowing removal
  of Tulu's `FINAL:` protocol prefix. No canned replacement reply was delivered.
- Tulu: 36 inference requests; OLMo: 33. All returned HTTP 200. No completion
  ended with `length`; maximum prompt sizes were 2,282 and 5,776 tokens.
- Tulu attempted seven saves: one `save_offer_required`, five
  `draft_not_presented`, one `draft_not_found`. OLMo attempted three: one
  `save_offer_required`, two `draft_not_presented`.
- Zero saved comments and zero classifier checks across both models. This
  batch does not measure classifier approval accuracy or successful-save replies.
- The 329 snapshotted source/prompt/test files were unchanged after the batch.
- Both owned vLLM services exited with code 0. Loopback port 18030 was closed
  afterward; the host GPU compute-process list was empty.
- Production, Slack, drivers, installed models and permission thresholds were
  unchanged. Only isolated test databases received draft-state changes.

## Evidence and reproduction

Local evidence root:
`tests/behavioral/local/staged-20261112/single-preparation-retest/`.

Each model has `pytest.log`, `results.xml`, `server.log`, and per-case
`pytest/test_staged_conversation_*/conversation.json` plus `wire.json`.
`receipt.json` records service lifecycle and source hashes; `source/` contains
the frozen files. Pytest's `*current` links refer to the same cases, not extra
runs.

The local launcher accepts a bounded case selection. From the backend root,
use a fresh output directory and check GPU occupancy before rerunning:

```sh
../.venv/bin/python tests/behavioral/local/staged-20261112/run_save_contract.py \
  tests/behavioral/local/staged-20261112/single-preparation-retest-next \
  --cases explicit_save natural_approval wording_only revision_then_approval cancel
```

Both cached checkpoints used FP16 vLLM, an 8,192-token context and one loaded
model at a time. Tulu used its established planner/text-tool setup; OLMo used
the saved short introduction, native `olmo3` parser and
`assistant.tools.olmo3_preamble`. These different integrations are not an
identical-prompt model ranking. These cases are development evidence, not a
fresh promotion gate.

Next scoped investigation: align preparation mode with the model's intended
review question, repair the false-positive claim retry, and check revision
registration. Retest those specific failures before expanding the suite.
