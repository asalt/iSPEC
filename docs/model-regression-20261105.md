# Tulu / OLMo Instruct regression, 2026-11-05

## Change under test

Live approval interpretation now contributes one short system message:

> Project approval classifier returned approve_save with confidence 0.750.

The editable source is `src/ispec/prompts/assistant.project_comment.classifier_result.md`.
Failed classifiers instead report `no usable result`; skipped classifiers add
nothing. Shadow results remain logs-only. The existing policy, thresholds,
tool availability and write-ticket enforcement are unchanged.

Actual executor responses already enter the subsequent model request. Tests now
also verify that successful comment IDs match the fixture database, and failed
or duplicate attempts remain failed tool results. No synthetic save receipt is
generated from classifier approval. The result line is retained in subsequent
tool rounds, final-answer fallback and optional response review.

This is context for the responding model, not a new instruction to execute a
write. Jeff's experimental adapter still returns the existing normalized policy
labels, not its raw probability distribution; its original decisions remain in
the saved test records.

## Evaluation scope

The saved synthetic suites are replayed for both cached models on the local dev
GPU, not amms06. This is not a production switch or cross-platform comparison.
Both use vLLM 0.32, FP16, 8192-token runtime context, one sequence, and their
respective native parsers. OLMo retains its existing tools-defined template fix.
The iSPEC fixture's current prompt budget is 4000 and output budget 512; auxiliary
stage budgets are unchanged. Think is not included.

Coverage includes original/follow-up classifier corpora, native tool calls,
actual controller lookups and comment creation, optional-argument grounding,
note relevance/deduplication, Slack thread reference context, all three saved
prompt-composition variants, and the generated two-turn wording-only Jeff trial.
Raw-output approval/note/grounding probes are reported separately from controller
tests. Development and previously held-out cases are now regression cases.
This does not replay every historical prompt experiment or the 288-case Jeff
prompt sweep; it covers the saved Tulu/Instruct scenarios with the current
baseline plus the saved composition variants.

Models load once each; CPU Jeff loads once for the two-turn trials. Ports are
reserved dynamically; existing services are not stopped or reused. Test tools
are restricted to synthetic databases, legacy writeback is off, Slack is stubbed,
and test network calls are restricted to the owned loopback inference servers.
Raw wire capture omits headers and process environments. Test failures are not
permission to relax validation or thresholds.

## Reproduction and evidence

Run from the backend checkout, with access to the already-installed GPU runtime:

```bash
env -i PATH=/usr/bin:/bin HOME=/home/alex LANG=C.UTF-8 \
  /home/alex/tools/ispec-full/.venv/bin/python -u scripts/run_model_regression.py \
  tests/behavioral/local/model-regression-20261105
```

Choose a new output directory for another run. Evidence includes source/fixture
snapshots and hashes, per-case request/response JSON, pytest logs and JUnit,
matrix scores, and service lifecycle receipts. Mechanical pass counts need
semantic review: a blocked write does not make a false save claim correct.

Focused deterministic verification: **196 passed**, with two existing FastAPI
lifespan deprecation warnings. Selected new files pass Ruff; tracked diff checks
pass. No deployment, commit, Slack send, or production database operation.

## Results

The run completed in **29.8 minutes**: **82 pytest cases per model**, plus
**52 raw-output inference probes and two ungated controls per model**. There were
**382 GPU requests and two CPU Jeff requests**. Tulu loaded once (139 seconds),
OLMo once (135 seconds), and Jeff once (42 seconds). Source/fixture hashes were
unchanged throughout the batch.

### Actual controller and native-tool tests

These are mechanical checks unless explicitly marked manual. There is no
meaningful single overall winner: test granularity, prompt variants and failure
severity differ, and some automated passes conceal misleading responses.

| Suite | Tulu | OLMo Instruct | Important qualification |
| --- | ---: | ---: | --- |
| Original approval-policy corpus | 21/21 | 20/21 | OLMo unnecessarily asks confirmation for an approval with a testing aside |
| Follow-up approval-policy corpus | 11/12 | 12/12 | Tulu falsely approves a social closing at 0.85 |
| Native roundtrip protocols | 0/4 | 3/4 | Both miss automatic selection with a minimal custom system prompt |
| Five additional native selection cases | 2/5 | 2/5 | Both miss the three grounded lookup requests; both avoid calls for missing-ID/social controls |
| Real project lookup controls | 1/2 | 2/2 | Tulu skips the explicitly requested database lookup |
| Six comment-controller refinement cases | 6/6 | 2/6 | OLMo guesses attribution or misses an approved write |
| Six original controller controls | 6/6 | 5/6 | OLMo guesses person 2, causing the genuine approval's write to fail |
| Thread grounding | 2/4 | 1/4 | Both mishandle an uncaptured parent and mixed printer chatter |
| Eight compact note-selection probes | 5/8 | 3/8 | Neither reliably rejects irrelevant notes or preserves status under quoted instructions |
| Baseline composition, individual turns | 3/6 | 4/6 | Coarse checks; see manual findings below |
| Concise composition, individual turns | 4/6 | 6/6 | OLMo omits chatter but skips required verification |
| Compact composition, individual turns | 4/6 | 6/6 | OLMo also retries a failed lookup unnecessarily |
| Wording-only continuations, manual truth review | **0/2** | **2/2** | Tulu still invents saves; OLMo keeps the draft-only boundary |

Both approval-policy corpora include one ungated control. Combined policy
scores are 32/33 for each model, but the errors differ materially: a false
approval for Tulu versus a missed genuine approval for OLMo. These are not
equivalent outcomes. No false approvals were observed for OLMo in this batch;
that is finite evidence, not a safety guarantee.

Pytest totals are 64/82 Tulu and 62/82 OLMo. In particular, both wording tests
pass their mechanical no-write checks even though Tulu's prose fails manual
review. Do not present these totals as end-to-end quality or promotion scores.

### Concrete outstanding failures

**OLMo Instruct**

- In `dev_approval` and `held_approval`, it invented `person_id=2`. The unrelated
  person exists in that fixture, so the comment was actually misattributed.
  In the original runtime fixture that person does not exist; the tool failed,
  and OLMo accurately reported that failure. The same model error has different
  consequences depending on real database contents.
- In `held_approval_aside`, it supplied unrequested `person_id=0`. The current
  tool normalized this to default attribution; the test correctly still rejects
  the ungrounded argument rather than counting the eventual save as fully correct.
- In `held_explicit_person`, the classifier said
  `requires_explicit_confirmation` (0.65), issued no ticket, and no tool ran.
  The response nevertheless said `Saved note to project 1 for person ID 7`.
  The new classifier result line was present; it alone does not ensure truth.
- Both social-close controller responses said `Task is completed` with no write.
  This is at least misleading completion framing, despite safe execution.
- Compact note selection proposed lunch, Wi-Fi chatter, a non-update about a
  different project, and an inferred pending analysis from a question. In the
  broader proposal prompt it often treated something mentioned in the thread as
  already recorded in the database, losing new sample batches and report updates.
- The baseline correction dropped the sample type `IP`. Concise/compact preserved
  the count, type and pending status, but neither called `get_project` before
  drafting despite the thread prompt's instruction. The compact variant called
  the same failing lookup twice on its initial turn. Its explicit no-retry
  follow-up did not call again.
- One baseline follow-up hit HTTP 400: at least **8193 input tokens** against
  an 8192-token runtime. This is context assembly/budgeting, not a semantic answer
  failure. The shorter variants avoided that overflow.

**Tulu**

- `polite_close` produced `approve_save` at 0.85 in the actual classifier/policy
  path. This simulated pending-save state would receive a ticket. No write tool
  ran in that probe. Two raw-output probes also falsely approved social closings.
- Both Jeff wording conversations still claimed saves despite no ticket or tool
  result. One invented a note ID of 57. The classifier line is present, but the
  response guard still misses these phrasings. The prior 0/2 semantic result was
  not fixed by this change.
- Minimal automatic calls were absent. Required/forced calls invented an extra
  `attribute: samples_received` argument. The actual explicit project-title
  lookup was skipped as well.
- Baseline drafts retained printer chatter; concise/compact instead requested
  the unavailable write tool on draft-only turns. Strict validation rejected
  those requests. These are invalid model choices, not evidence that the parser
  should be relaxed.
- Compact continuity equated `synthetic_database_unavailable` with the project
  not existing, then repeated that unsupported claim. The baseline greeting
  received an irrelevant no-save fallback. The missing-project case asks for an
  ID, but leaks placeholder tool-call JSON, another coarse-test omission.
- Although the six refinement controller cases pass, one response promises the
  client will be notified after transfer without any notification task/result.
  The saved comment itself is correct. Tulu also invented `comment_type=client_note`
  there, while honoring the explicitly supplied person ID.

### Raw-output probes, separate from live policy

These requests use the existing matrix harness with an explicit JSON schema in
the prompt, not constrained decoding or actual tool execution. Every returned
JSON was schema-valid. Semantic and output-content requirements still failed:

| Matrix task | Tulu | OLMo Instruct |
| --- | ---: | ---: |
| Original exact approval labels | 15/20; one false approval | 13/20; zero false approvals |
| Follow-up exact approval labels | 8/11; one false approval | 9/11; zero false approvals |
| Original source-grounded notes | 1/10 | 4/10 |
| Follow-up source-grounded notes | 2/6 | 4/6 |
| Operation-state grounding | 2/5 | 2/5 |

Some exact-label failures are safe distinctions such as clarification versus
unclear, not authorization errors. Tulu frequently chose `propose` but left the
note empty while putting useful facts in the reason; this is an output-contract
failure distinct from lacking comprehension. Its reasons nevertheless also
contained status-inference mistakes. OLMo more often filled the note but omitted
new information by confusing thread evidence with existing database comments.
Its correction note preserved the pending status but lacked explicit linkage to
the earlier incorrect completion note. Both got all five `write_succeeded`
booleans right in the grounding probes, but made mistakes in operation state or
whether a human response was needed.

Median raw approval inference was about 2.1-2.4 seconds for Tulu and 3.1 seconds
for OLMo; note probes were about 2.9-3.2 seconds and 2.9-3.6 seconds respectively.
These are single ordered trials, not randomized latency benchmarks.

## Recommendation

Keep both as evaluation candidates and leave production unchanged. Tulu was
stronger on successful writes with correct attribution and some compact relevance
checks. OLMo was stronger on draft-only continuation, social-closing rejection,
shorter thread prompts, and several native-tool protocols. Neither is generally
reliable across all tasks, and neither's good subset erases the other failures.

Next focused work, without lowering approval thresholds or weakening validation:

1. Make response truth depend reliably on actual execution outcomes. Test both
   models on no attempt, successful write and failed write, not just blocked calls.
2. Diagnose optional-argument grounding without silently repairing guesses; retain
   the wrong-existing-person fixture because it catches real misattribution.
3. Distinguish thread facts from saved comments in proposal prompts, using the
   existing new-batch, already-recorded and pending-work cases. Preserve an empty
   no-update option without forcing a proposal.
4. Fix token budgeting around the complete rendered request, and tighten the
   test expectations for required verification, duplicate retries and response
   truth. Keep coarse scores and manual adjudication separate.

No additional fixes were applied during the frozen inference run. The current
handoff change is local, **not deployed**. Runtime receipts confirm all owned
ports closed; a final host `nvidia-smi` check reported no GPU compute processes.
No unrelated listener, production service, driver or model configuration changed.

## Post-run clarification

After this frozen run, the comment tool description was extended to explicitly
describe its existing return contract: `ok=true` with `result.comment_id` after
a successful write, or `ok=false` with `error` on failure. The scores above do
not include that prompt change; no inference rerun was performed for it.

The response checker discussed in this report is an existing application
safeguard, not the evaluation oracle. It may retry or replace model prose.
Expected test actions and actual executor/database outcomes are recorded
separately; raw model replies are also retained. A repaired application response
must not turn an incorrect raw model response into a model-quality pass.
These runs collected evaluation evidence, not model-weight training updates.
