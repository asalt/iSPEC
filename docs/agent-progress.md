# Agent Progress Summary (Local Dev Agent + Supervisor + Queue + Inference Broker)

Last updated: 2026-03-06

This document is a continuity anchor for ongoing work on iSPEC’s local agent runtime (supervisor + orchestrator + queued chat + telemetry commands). It is intentionally concise and code-referenced, so we can restart tomorrow without reloading a huge chat transcript.

## Where We Are (High Level)

- The supervisor is the primary “runtime manager” process: it checks system health, owns the agent command queue, and is the single writer to the agent SQLite DB.
- Support chat can run in **queue mode** (API enqueues a command; supervisor executes it) with an inline fallback.
- The orchestrator runs as a queued command (`orchestrator_tick_v1`) and schedules its own future ticks with backoff.
- We added a **single-lane inference broker thread** inside the supervisor process so the main thread stays responsive while vLLM inference is in-flight.
- We added tackle-oriented queue commands so external tooling (tackle) can enqueue “assess results” and “freeform prompt” jobs via the API-key protected `/api/agents/commands` endpoint.

## Execution Model: Processes And Threads

### Processes

- **API server process**: FastAPI app serving `/api/*` routes.
  - Relevant include: `src/ispec/api/main.py`
- **Supervisor process**: long-running loop that:
  - owns agent-command scheduling/claiming/finalization
  - owns agent DB writes (single-writer contract)
  - performs health checks and periodic work
  - Relevant entry: `src/ispec/supervisor/loop.py:run_supervisor`

### Threads (Inside Supervisor Process)

- **Supervisor main thread** (`supervisor-main`):
  - Declared as main via `set_main_thread(owner="supervisor")`
  - Enforced by `assert_main_thread(...)` for DB-mutating code paths
  - Implementation: `src/ispec/concurrency/thread_context.py`
- **Inference broker thread** (`supervisor-inference`):
  - Executes one blocking vLLM/Ollama call at a time (`generate_reply`, `stream=false`)
  - Must not touch SQLite or shared state
  - Implementation: `src/ispec/supervisor/inference_broker.py`

The design goal is “single writer to SQLite, single lane to the GPU,” while keeping the supervisor loop responsive to new queued commands and housekeeping.

## Agent Command Queue (AgentCommand)

The “queue” is persisted in the agent DB (`agent_command` table). The supervisor:

1. claims one queued command (`status=queued`, `available_at <= now`) and marks it `running`
2. executes it
3. finishes it (`succeeded`/`failed`) or defers it back to `queued` with a delay

Key code paths:

- Claim/finish/defer: `src/ispec/supervisor/loop.py` (`_claim_next_command`, `_finish_command`, `_defer_command`)
- Stale recovery: `src/ispec/supervisor/loop.py:_recover_stale_running_commands`
  - Env: `ISPEC_SUPERVISOR_RUNNING_STALE_SECONDS` (default 300s)

Important operational note:
- Long-running in-flight commands are protected from “stale recovery” by periodically touching `AgentCommand.updated_at` while inference is running (`_touch_command_updated_at`).

## Inference Broker (Responsiveness While Inference Runs)

### What It Does

When enabled, LLM-using commands do not block the supervisor main thread:

- main thread starts an LLM “task” and enqueues an `InferenceRequest`
- inference thread runs the blocking HTTP call to vLLM (`generate_reply`) and pushes an `InferenceResult`
- main thread drains the result, advances the task, and finalizes the command in SQLite

This enables the supervisor to keep doing:

- heartbeats (so queue-mode chat routing stays “safe”)
- non-LLM commands (Slack scheduled sends, dev restarts, sync tasks, etc.)
- prompt/result bookkeeping and DB writes

### Enablement

Env var (tri-state):

- `ISPEC_SUPERVISOR_INFERENCE_BROKER_ENABLED=1` forces on
- `ISPEC_SUPERVISOR_INFERENCE_BROKER_ENABLED=0` forces off
- unset/`auto` enables when `ISPEC_STATE_DIR` points at a `.pids` directory

Broker status is recorded in:

- `.pids/supervisor.json` (key: `inference_broker`)
- `agent_run.config_json["inference_broker"]`

## Support Chat: Queue Mode + Heartbeat

Queue-backed support chat flow:

1. `/api/support/chat` decides between inline vs queue mode.
2. If queue mode is enabled and supervisor heartbeat is fresh, API enqueues a command:
   - `command_type=assistant_support_chat_turn_v1`
3. API polls the command row for completion and returns the result.

Key implementation:

- Queue routing + wait loop: `src/ispec/api/routes/support.py`
- Supervisor command execution: `src/ispec/supervisor/loop.py:_run_support_chat_turn`

Primary env vars:

- `ISPEC_ASSISTANT_CHAT_QUEUE_ENABLED` (tri-state; default “auto-enable” in dev `.pids` layouts)
- `ISPEC_ASSISTANT_CHAT_QUEUE_WAIT_SECONDS` (default 120)
- `ISPEC_ASSISTANT_CHAT_QUEUE_POLL_SECONDS` (default 0.5)
- `ISPEC_ASSISTANT_CHAT_QUEUE_SUPERVISOR_MAX_AGE_SECONDS` (default 60)
- `ISPEC_SUPERVISOR_HEARTBEAT_SECONDS` (default 15)
- Storage layout now has separate canonical DB vars:
- `ISPEC_DB_PATH` for core metadata, `ISPEC_ANALYSIS_DB_PATH` for E2G/volcano/GSEA, and `ISPEC_PSM_DB_PATH` for large PSM tables
- `ISPEC_OMICS_DB_PATH` remains as a deprecated compatibility alias for `ISPEC_ANALYSIS_DB_PATH`

Failure mode to remember:
- If the supervisor is running but blocked (e.g., long inference without broker), the API wait loop can hit the timeout and return 504. The inference broker exists primarily to prevent this.

## Tackle Telemetry: Two Queue Command Types

We added two tackle-oriented agent commands:

1. `assistant_assess_tackle_results_v1`
   - Structured output (guided JSON) assessment
2. `assistant_run_tackle_prompt_v1`
   - Freeform plain-text commentary

Command constants:
- `src/ispec/agent/commands.py`

Remote enqueue/fetch endpoints (API-key protected):

- `POST /api/agents/commands`
- `GET /api/agents/commands/{id}`

Implementation:
- `src/ispec/api/routes/agents.py`
- `src/ispec/api/main.py` (agents router is under `Depends(require_api_key)`)

Example payloads:

```json
{
  "command_type": "assistant_assess_tackle_results_v1",
  "payload": {
    "project_id": 1498,
    "tackle": {"argv": ["tackle", "..."], "config_file": "/abs/path/config.conf"},
    "results": {
      "pca": {"ctrl-1": [0.1, -0.2]},
      "limma": {"top_table_preview": [{"gene": "TP53", "logFC": 2.0, "adj.P.Val": 0.001}]}
    }
  }
}
```

```json
{
  "command_type": "assistant_run_tackle_prompt_v1",
  "payload": {
    "project_id": 1498,
    "prompt": "Here are PCA coords on [-1,1]... Please comment on what you see.",
    "context": {"dataset": "foo", "n": 12},
    "max_tokens": 900
  }
}
```

Operational notes:
- `/api/agents/commands` prunes payloads before storing to prevent agent DB bloat. Keep raw artifacts in tackle’s telemetry store if needed.
- Remote command allowlist is intentionally narrow to avoid creating a generic remote LLM runner.

## Tool Continuity: Avoid “Invented FINAL”

Observed failure mode:
- The model sometimes “wants” a tool that wasn’t included in the tool subset for that turn.
- With forced tool choice, it can pick a “nearest” tool and then hallucinate an answer.

Mitigations in place / direction:
- Include a meta tool catalog: `assistant_list_tools` so the model can discover available tools instead of guessing.
  - Implementation: `src/ispec/assistant/tools.py` (tool catalog payload)
- Tool routing improvements in the assistant support pipeline:
  - `src/ispec/assistant/tool_routing.py`
  - `src/ispec/api/routes/support.py`

Rule of thumb:
- If the needed tool is unavailable, the assistant should say so and ask to enable/provide it, not substitute silently.

## Dev Ops: tmux + Make + PID/State Files

### Recommended “Source Of Truth” For Process Status

Prefer:

1. `.pids/*.pid` for a flat “is it running” PID view
2. `.pids/*.json` for richer status (run id, thread main info, broker status)
3. `make dev-status` / `make dev-pids` for human-friendly summaries

tmux inspection is secondary (useful for “what pane is stuck”), not the primary status source.

### Codex/Agent Process Assessment Approach

When the Codex agent needs to assess “what is currently running in tmux ispecfull”:

- Use `make dev-status` and `make dev-pids` (top-level Makefile one directory above `iSPEC/`):
  - from repo root: `make dev-status`, `make dev-pids`
  - from inside `iSPEC/`: `make -C .. dev-status`, `make -C .. dev-pids`
- Cross-check `.pids/supervisor.json`:
  - confirms supervisor run id, main thread identity, and inference broker thread status
- Only then, if needed, use tmux commands to inspect panes/logs.

### Codex Skill

There is a dedicated skill for this workflow:
- `skills/ispec-dev-make/SKILL.md`

It documents `dev-tmux`, `dev-status`, `dev-stop`, `dev-restart`, `dev-pids`, and service selection via `DEV_RESTART_SERVICES=...`.

## Quick Verification Checklist (Local)

1. Supervisor state exists:
   - `.pids/supervisor.pid`
   - `.pids/supervisor.json` includes `thread_main` and `inference_broker`
2. Queue-backed chat works:
   - enqueue a chat via UI/API and confirm it returns before `ISPEC_ASSISTANT_CHAT_QUEUE_WAIT_SECONDS`
3. Tackle enqueue roundtrip:
   - `POST /api/agents/commands` then `GET /api/agents/commands/{id}` until `status=succeeded`
4. Broker responsiveness:
   - while an LLM command is in-flight, confirm supervisor heartbeat stays fresh (no 504 wait timeout)

## What’s Next (Not Implemented Yet)

- Queue manipulation semantics (cancel / reprioritize) for pending work.
- DB pressure management (archiving/pruning very old heavy `agent_step` runs).
- Further tuning of backoff intervals and “how often we should check things” for supervisor + orchestrator.
- Continue consolidating env vars under tri-state patterns where appropriate.
