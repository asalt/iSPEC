# Native Ollama tool routing

The adapter uses `/api/chat`, preserving tool definitions, assistant tool calls,
and named `role=tool` results. Controller-facing calls retain the existing
OpenAI-shaped IDs and JSON argument strings. Structured JSON/choice requests,
per-call timeouts, temperature, token limits, and token-usage accounting are
translated rather than silently discarded. Thinking text is not user-facing
output; empty or token-exhausted responses fail closed.

The support controller now admits Ollama to the existing synchronous tool-group
router, explicitly enabled turn decisions, and comment-intent hints. This does
not enable background compaction, post-send preparation, a receiver, or live
approval policy. Existing authorization and write tickets remain the actuator.

Native Ollama has no matching `tool_choice` request field. A named choice limits
the supplied tool list; required/named choices are checked on return. Failure to
return the requested tool is an error, not a success, prose-to-write fallback,
or retry. This is validation, not a promise of forced upstream decoding.

Optional configuration, shown in `.env.example`:

```dotenv
ISPEC_ASSISTANT_PROVIDER=ollama
ISPEC_ASSISTANT_TOOL_PROTOCOL=openai
ISPEC_OLLAMA_MODEL=gpt-oss:20b
ISPEC_OLLAMA_URL=http://127.0.0.1:11434
ISPEC_OLLAMA_NUM_CTX=8192
ISPEC_OLLAMA_NUM_PREDICT=1024
ISPEC_OLLAMA_THINK=low
ISPEC_OLLAMA_TIMEOUT_SECONDS=120
```

Thinking controls depend on the model. Caller-specific output caps take priority
over the environment default and include reasoning tokens; a short classifier
budget must be evaluated, not assumed adequate for every reasoning model.

## Offline validation

`tests/unit/assistant/test_ollama_protocol.py` tests wire translation and failure
handling without networking. `tests/integration/test_llm_ollama_smoke.py` is
opt-in with `ISPEC_RUN_LLM_TESTS=1`; it checks text, a synthetic read-only tool
round trip, and bounded denial JSON without touching a project database.

`scripts/eval_slack_note_proposals.py CASES --output REPORT` evaluates selected
thread snapshots. It cannot read Slack, send messages, or save project notes.
Its prompt is `experimental.slack.note_proposal`; it excludes expected labels
from model inputs. Cases contain `id`, `date`, `project_id`, `thread`,
`existing_comments`, and an optional `expected_decision`. Real cases/reports
belong under ignored `tests/behavioral/local/`; they are not public fixtures.
Use `--no-json-constraint` for a prompt-only baseline. Input hashes allow
comparison against the standalone cached-Transformers replay script.

The existing approval corpus also supports `ISPEC_APPROVAL_EVAL_PROVIDER=ollama`
with `ISPEC_RUN_LLM_TESTS=1`; it permits only the configured loopback inference
endpoint and never executes a write. A correct JSON shape is not proof of a
correct semantic decision. Review false approvals and missed/duplicate proposals
before production activation.

## Production boundary

The production website and its JDBC sync do not depend on this adapter change.
Deploying code is separate from activating the assistant and Socket Mode. Keep
public support/internal-service routes blocked until activation is reviewed;
do not weaken browser authentication to make the Slack bridge work. Confirm
staff identity, same-thread pending state, durable receipt paths, and exactly one
receiver before posting a proposal that invites confirmation.
