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

## Benchmark runner

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
