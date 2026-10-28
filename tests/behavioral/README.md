# Behavioral Tests

`tests/behavioral` is the opt-in scenario lane for experiments that sit between
fast unit tests and heavier live integration tests.

Use this lane for:

- conversational flow checks that benefit from a seeded sandbox DB
- controller / classifier / routing experiments
- shadow-mode and replay-style evaluation
- draft end-to-end scenarios we want to iterate on before promoting into the
  regular unit or integration suites

These tests are skipped by default.

The [October 25 OLMo follow-up](../../docs/olmo-vllm-evaluation-2026-10-25.md)
adds a separate curated holdout directory and an opt-in real-controller fixture
test. Its experimental Think template is never loaded by ordinary application
startup. Model failures are distinct from deterministic test-suite failures.

Run them with:

```bash
pytest tests/behavioral --run-behavioral
```

or:

```bash
ISPEC_RUN_BEHAVIORAL=1 pytest tests/behavioral
```

The reusable datastore helpers for this lane live in
[`datastore.py`](./datastore.py). They create a small sandbox environment with
seeded core data plus initialized assistant, schedule, and agent SQLite DBs so
behavioral scenarios can share a common fixture style without polluting the
faster suites.

Suggested contents for this lane:

- shadow-mode support chat scenarios
- replay-fixture evaluation for controller/contract experiments
- bounded classifier/controller behavior checks before promotion to `unit`
- sandboxed end-to-end flows that are too stateful to read well as tiny unit tests

## Local extracted corpus

The ignored local corpus lives under `tests/behavioral/local/`. Use it for richer
real-world scenarios pulled from the dev assistant DB without checking raw
conversations into git.

Extract scenarios with either:

```bash
make extract-backend-behavioral BEHAVIORAL_SESSION_PKS=244 BEHAVIORAL_TAGS='project-note slack transaction'
```

or directly:

```bash
cd iSPEC
python scripts/extract_behavioral_scenarios.py --session-pk 244 --tag project-note --tag slack --tag transaction
```

The checked-in behavioral smoke set should stay small and sanitized. Promote a
local extracted scenario only after manually curating it into a stable fixture.

## Optional reply-phrase trial

[`data/response_phrase_cases.json`](./data/response_phrase_cases.json) contains
eight synthetic review scenarios with context and flexible review criteria.
The experimental corpus is `src/ispec/prompts/experimental.response.phrases.md`;
it is not included in live prompts. The initial raw-model comparison exposed
unsupported success/reminder claims, so it remains an evaluation artifact.
See [`slack-project-note-monitoring.md`](../../docs/slack-project-note-monitoring.md)
for findings and limitations. Prompt wiring tests are not model-accuracy tests;
repeat with production-shaped context before claiming controller behavior.

## Benchmark runner

For a real-model queue/context replay without live writes or Slack sends:

```bash
ISPEC_RUN_VLLM_TESTS=1 pytest tests/behavioral/test_slack_thread_queue_vllm_live.py --run-behavioral -q -s
```

This calls the supervisor's queued-turn entrypoint against temporary databases;
tools are intercepted, and only inference at localhost:8000 is permitted. It
prints replies, proposed tool arguments, context inclusion and timing for review.
Passing assertions establish context/target grounding, not complete note quality
or permission to save. The running supervisor/model are not restarted by this test.

Use the benchmark runner when we want timing, tool-loop counts, and scenario
comparison against the live local support stack without turning pytest into a
performance harness.

Run it from the repo root with:

```bash
make benchmark-support-vllm
```

Or pass through extra selectors/flags such as:

```bash
make benchmark-support-vllm BENCHMARK_SUPPORT_ARGS='--scenario-label tmux-unique-pane-summary --allow-supervisor-running'
```

This runner is intentionally separate from `tests/behavioral`: behavioral tests
prove correctness and regression safety, while the benchmark runner measures
full-turn latency, model-call counts, tool-loop rounds, and related support-chat
performance signals.

The default benchmark path forces inline chat and expects the background
orchestrator to be disabled unless explicitly enabled. Timing buckets default to
90 seconds for `slow` turns and 180 seconds for `problematic` turns; override
them with `--slow-threshold-seconds` and `--problematic-threshold-seconds` when
calibrating a particular local machine.

## Offline model comparison

`scripts/eval_model_matrix.py` prepares the existing 20 gated approval cases,
one no-pending-action control (no model call), ten synthetic note cases, and five
context-grounding probes. New inputs live in `data/model_note_cases.json` and
`data/model_grounding_cases.json`. No Slack history is fetched, no tools are
exposed, and no application data is written.

From the backend root, with an already-running local inference endpoint:

```bash
.venv/bin/python scripts/eval_model_matrix.py prepare tests/behavioral/local/my-trial
.venv/bin/python scripts/eval_model_matrix.py run tests/behavioral/local/my-trial/requests.json \
  --provider ollama --model gpt-oss:20b --think low --url http://127.0.0.1:11434 \
  --output tests/behavioral/local/my-trial/results.json
.venv/bin/python scripts/eval_model_matrix.py score tests/behavioral/local/my-trial/manifest.json \
  tests/behavioral/local/my-trial/results.json --output tests/behavioral/local/my-trial/scores.json
```

Only loopback inference is permitted; an explicit temporary SSH forward can
reach a trusted host. The runner does not start services, load workspace auth,
download weights, or dispatch tool calls. An existing inference server may load
the selected cached model into GPU memory. Expectations are kept in a separate
manifest and input fingerprints must agree when scoring. Rendering records prompt
versions in the trial directory's `prompts.db`, not the live registry. Raw results
are checkpointed after each case and remain ignored by Git.

The same `requests.json` can be passed to `eval_cached_transformers_chat.py` in
an existing CUDA-compatible environment, using `--model-path` for cached weights
and `--output` for a new result file. This tests content, not vLLM tool parsing.
The default per-case budgets are 160 approval, 512 note, and 256 grounding tokens;
`run --max-output-tokens 1024` creates a separate reasoning-budget diagnostic.
Temperature is zero, JSON is prompt-only for cross-runtime comparison, and no
automatic retries replace failed trials.
Reasoning is unset by default: pass `--think low` for GPT-OSS, and omit it for
non-reasoning models such as Tulu or OLMo Instruct. Reasoning settings are recorded
in the runtime metadata rather than imposed across unrelated model families.

`prepare --note-prompt compact --tasks note grounding` selects an experimental
compact note prompt. This is not activated in the bot. The initial October 23
grounding probe was ambiguous about what counted as a write and was revised;
preserved request fingerprints identify that difference. Never combine unlike
inputs into a single model ranking. See
[`model-comparison-2026-10-23.md`](../../docs/model-comparison-2026-10-23.md).
The OLMo Instruct Q4/FP16 and Tulu Ollama Q4 follow-up, including native tool and
actual classifier-path failures, is in
[`model-runtime-comparison-2026-10-23.md`](../../docs/model-runtime-comparison-2026-10-23.md).

Scores separate exact label agreement, JSON validity, false approvals at 0.75,
and the current six-second approval budget. Note fragment checks are deliberately
small regression signals, not semantic judges; manual review remains necessary.
The no-pending control checks evaluator gating, not model accuracy. Treat prompt
revision results as exploratory, then validate on newly held-out cases and the
real controller before any deployment decision.

## Isolated Instruct interface refinement

`scripts/run_instruct_refinement.py` runs native tool probes, actual `support.chat`
turns with real temporary fixture writes, and source-grounded note trials. It uses
a clean child environment, never loads credential/env files, allows only loopback
inference and never starts a service. Output directories must be new. The frozen
corpus has three development and sixteen held-out cases; two separately reported
real-schema lookup diagnostics were added afterward. No prompt retuning used the
held-out answers. See the [refinement report](../../docs/instruct-refinement-2026-10-25.md)
for exact commands, failures, runtime and cleanup. These live tests are opt-in,
and expected model failures remain failures rather than being marked xfail.

## Reviewed private-snapshot pipeline

Use `scripts/prepare_behavioral_case.py` to stage selected assistant/Slack snapshots,
optionally request a model rewrite, review privacy and preserved meaning, and
explicitly export a fixture. None of these steps add or commit files to Git.
See [the workflow](../../docs/behavioral-corpus-workflow.md). The existing session
extractor remains unchanged; its raw outputs are private, not automatically anonymized.

For narrated saved-evidence reviews, captions and an optional quiet musical outro,
see [behavioral review videos](../../docs/behavioral-review-videos.md). Rendering
does not run inference, turn failures into passes, or publish the result.
