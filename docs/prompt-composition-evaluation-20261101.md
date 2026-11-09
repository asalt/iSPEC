# Prompt composition and short thread continuity, 2026-11-01

## Scope and evidence

This is a local development experiment through `support.chat`, its real router,
strict provider contract and fixture tool dispatcher. It is not a production
benchmark, promotion, replay of all historical chats, or long-term memory test.
The fixture uses temporary core/assistant/schedule databases, synthetic Slack
thread retrieval and a deliberately unavailable project lookup. No Slack sends,
legacy writes, production changes, downloads or installs are involved.

Four conversations contain six evaluated turns: mixed project/printer discussion
and a subsequent count correction; historical approval explicitly superseded by
the current no-action request; failed lookup and a no-retry follow-up; and a
greeting without a task. Mixed/quoted seeds reuse the earlier synthetic corpus.
Follow-ups, failure and greeting were frozen before this run, not fitted to its
answers. After inspection these are development examples, not a reusable claim
of unseen held-out performance. Every turn expects **no write**; this slice does
not establish positive approval quality.

Three arms keep permission gates, strict parser validation and tool schemas:

| Arm | Experimental change |
| --- | --- |
| baseline | Current development base/controller |
| concise | Replace only the base with `experimental.assistant.base.concise` |
| compact | Concise base plus merge adjacent plain system messages |

The merge preserves text order and priority, does not merge across user/tool
messages, and leaves named/extra-field messages alone. It changes message
boundaries, so equivalent model behavior is **not** assumed. Router, approval
controls and tool-result truth guards remain active. Self-review, response
contracts and automatic compaction are disabled by the existing isolated fixture.
The configured 4,000-token message estimate is not the server's token limit.

Each follow-up receives the actual preceding generated answer, not an idealized
answer. First turns have matched evidence; follow-up histories can diverge by arm.
Mechanical fragment checks are supplemented by manual answer review below.
Their original scores are retained, including scoring mistakes.

## Runtime

Local Quadro RTX 6000, 24 GiB, SM 7.5; driver 595.112.07; vLLM 0.32.0;
PyTorch 2.13.0. The initial host check found no compute jobs and about 22.8 GiB
free. All trials use FP16, 8,192 context, one sequence, 0.80 memory fraction,
eager execution and TRITON_ATTN on loopback port 18030. Each model is unloaded
before the next one starts. No production runtime is used or replaced.

| Model | Cached checkpoint revision | Main output budget | Dialect |
| --- | --- | ---: | --- |
| Tulu 3 8B | `666943798adbde0b1aff34626007e26986a3c107` | 512 | `llama3_json` |
| OLMo 3 7B Instruct | `6e5971d9eba42665f5bd5a0fcf047f299ce1dccc` | 512 | `olmo3` |
| OLMo 3 7B Think | `d97e442d7cc678210054dbcc9b440894d62c89a4` | 2048 | `olmo3` + reasoning parser |

Think weights are read from the already mounted amms06 cache, not downloaded.
The model's stored BF16 weights are cast to FP16 for this GPU. Think uses the
previously documented **evaluation-only** tools-aware template adapted from
Instruct, not its stock template. A one-line `tools | default(none)` guard fixes
startup introspection when no `tools` variable exists; unit tests verify unchanged
rendering for defined tools. Think's larger budget and different template mean
this is not an equal-budget latency comparison or proof of stock tool support.

## Context findings

The introduction occurred once per main inference request, not several times
within one request. Stateless calls still need their instructions and selected
context on each request. Prompt composition metadata is an index, not persistent
model memory, and prefix caching is not permission to omit instructions.

There are two distinct sources of avoidable overhead:

1. In both inspected Tulu follow-ups, the prior user request and assistant answer
   appeared twice: once as chat history, once in quoted Slack thread context.
   A future deduplication pass should use message identity and provenance, retain
   revisions/coverage and keep evidence distinct from current instructions.
2. The OLMo evaluation template emits the entire tool schema block for each system
   message. The controller's character-based estimate excludes that rendered
   expansion. Instruct baseline's failed-lookup follow-up reached HTTP 400 at
   more than 8,192 input tokens. This is an integration-budget failure, not proof
   the model cannot understand the task.

For the identical initial mixed-thread request, first main input tokens were:

| Model | baseline | concise | compact |
| --- | ---: | ---: | ---: |
| Tulu | 1,859 | 1,341 | 1,331 |
| OLMo Instruct | 5,174 | 4,656 | 2,384 |

The first wire call is usually the router; these are **main** request counts,
not router counts. Tool/template overhead explains much of the model difference.
This is a more useful optimization target than removing necessary policy text.

## Completed results and manual review

Thirty-six turns completed across the six Tulu/Instruct arms. These coarse
scores are **not** promotion scores; see the missed failures below. Medians include
unsuccessful responses and ordered cache effects, not just successful answers.

| Model / arm | Coarse checks passed | Median end-to-end seconds |
| --- | ---: | ---: |
| Tulu baseline | 3/6 | 3.37 |
| Tulu concise | 4/6 | 2.81 |
| Tulu compact | 4/6 | 3.64 |
| Instruct baseline | 4/6 | 7.34 |
| Instruct concise | 6/6 | 7.64 |
| Instruct compact | 6/6 | 3.65 |

- Tulu baseline included the unrelated printer detail in both drafts. Its greeting
  became an unsolicited note draft; the unsupported-write-claim guard then
  replaced the answer with an irrelevant no-save fallback. The original draft
  explicitly said it was unsaved, so this also exposes a guard precision issue.
  Keep the guard; isolate that precision case rather than weakening protection.
- Tulu concise and compact requested unavailable `create_project_comment` during
  draft-only turns. Strict validation rejected it; no tool write occurred. This
  remains a model-behavior failure even though the boundary worked.
- Tulu compact interpreted database unavailability as project nonexistence and
  repeated that error on the next turn. A saved assistant answer is not stronger
  evidence than the actual failed tool result.
- Instruct concise retained nine then six IP samples, pending processing, and
  omitted printer chatter. It also explained the failed lookup without retrying
  on the follow-up. However, the draft turns skipped the prompt's explicit
  `get_project` verification instruction. Content checks alone missed this.
- Instruct compact also produced grounded drafts, but retried the same failed
  lookup twice on its initial lookup turn and introduced unnecessary note-related
  discussion into a status question. The explicit no-retry follow-up did not call
  tools again.
- Think's first baseline router exhausted the existing 200-token auxiliary
  budget with `finish_reason=length`, so strict validation correctly rejected the
  incomplete result. Falling back to a broader tool catalog then overflowed the
  main request's context. This is a concrete stage-budget/template/controller
  interaction; increasing only the main output budget does not solve it.
- Automated checks mis-scored some answers: baseline Tulu's truthful "cannot be
  retrieved" failed the fragment vocabulary; its inappropriate greeting passed
  the no-tool check; compact Tulu's nonexistence claim passed the failure-word
  check. Preserve these records and add separate adjudication, not post-hoc score
  inflation. Missing `IP` in Instruct baseline's corrected draft is a real loss of
  requested detail.

Think baseline and concise completed twelve diagnostic turns, all returning the
failure fallback. Their routers exhausted 200 output tokens; main requests then
overflowed context or timed out. Compact was intentionally stopped before any turn
was recorded, following Alex's budget question. Think is excluded from the
comparative quality table: correct its stage budgets before drawing a verdict.

Across all 48 recorded turns there were zero created comments, reported issued
write tickets or executed comment-write calls. The unavailable write attempts
above still count as failures. No positive approval cases were run, so this is
not evidence that write authorization is ready for model promotion.

## Recommendation and next small test

Keep production and its fallback unchanged. Instruct with compact assembly is
promising, not ready: source verification, unnecessary retries and unsolicited
note framing remain. Do not adopt the terse base wholesale for Tulu.

Next, isolate **only the router** on fixed inputs and try 512 and 1,024 output
tokens for Think, recording completion/JSON validity, correct tool category,
truncation and latency. Both models already have an 8,192-token total context;
that is different from the router's 200-token generation cap. Do not change
planner prompts or permission thresholds in the same experiment. A CPU router
can later be compared against this baseline; it is not needed to diagnose this
budget mismatch. No extra router experiment was started at this stopping point.

The subsequently authorized router-only pilot is documented in
[router-budget-evaluation-20261101.md](router-budget-evaluation-20261101.md).
It varies only the output cap and separates truncation from JSON-contract failures.

Subsequent separate tests should address rendered tool-schema budget accounting,
bounded router-failure behavior instead of catalog expansion, and message-ID-aware
history deduplication. Keep task-specific note/status guidance as small components
beside the concise identity. Add fresh source-grounded drafts and natural approval,
denial, revision and social-close controls before promotion; a false approval is
still a blocker. No bandit, CPU model, training or historical graph backfill was
introduced.

## Reproduction and artifacts

The tracked entrypoint reuses the existing isolated runner. Start an authorized,
otherwise idle local model separately with the parser/template shown above, then:

```sh
.venv/bin/python scripts/run_instruct_refinement.py \
  --suite composition --trial baseline concise compact --split dev \
  --url http://127.0.0.1:18030 --model YOUR-SERVED-MODEL \
  --tool-parser olmo3 --max-output-tokens 512 \
  --output-dir tests/behavioral/local/NEW-composition-run
```

For Think add `--reasoning-parser olmo3 --max-output-tokens 2048`. Tulu uses
`--tool-parser llama3_json`. The runner requires `/health` and exact model identity
before creating trial output; it inherits neither credentials nor live DB paths.
The fixture network guard allows only the specified loopback inference endpoint.

Local ignored evidence lives in `tests/behavioral/local/composed-models-20261101/`:
source hashes, runtime snapshots, synthetic requests/responses, per-turn tool
results and metadata, usage/composition JSONL, pytest logs and lifecycle receipts.
`olmo-instruct-compact-prompt-paths.html` is a four-call graph export from real
inference on synthetic evidence, not fabricated historical composition.

Canonical comparison logs are `tulu-ready/{baseline,concise}/pytest.log`,
`tulu-compact-finished/compact/pytest.log` and `olmo-instruct/*/pytest.log`.
`completed-instruct-summary.json` summarizes those six completed arms. Think logs
under `olmo-think/` are diagnostic only; `think-stop-note.json` records the stop.

Excluded preliminary runs remain preserved: `tulu/` had a fixture transaction
held across independent inbox writes; `tulu-complete/` started before model
readiness; `tulu-ready/compact` had an interrupted final greeting. The fixes are
request-scoped fixture commits, a preflight readiness check, and an owned-service
driver that waits for the runner to exit before cleanup. Do not count those
harness/lifecycle failures as comparative model quality. Early summary files
from incomplete runs are not final evidence.

The runner's composition suite is development-only and rejects `--split held`;
that validation was added after inference without changing the frozen cases or
their scores. The owned-service receipts record cleanup. After stopping Think,
host verification found no GPU compute jobs and port 18030 was closed. No model
services or forwards from this experiment remain running. Narration remains
deferred; the saved examples include failures worth reviewing rather than a
capabilities-only demonstration.

Final focused verification: **197 passed, 4 skipped**, covering prompt registry/
composition, logging and observability, providers/runtime contracts, approval
policy, response guards, runner/template helpers and the stubbed composition
scenario. The four skipped tests are the opt-in real-model cases during this
ordinary test invocation; their isolated runs are reported above. Two existing
FastAPI lifecycle deprecation warnings remain. `git diff --check` passed. No
commits were made and unrelated backend/deployment work was preserved.
