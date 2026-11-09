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

Use `--prompt-family experimental.slack.note_proposal.minimal` to test an
explicit candidate without replacing the default prompt. Reports include the
actual prompt family, source path and body hash. The `minimal_status` sibling
adds only the status examples; `minimal_fields` adds one sentence explaining
the existing `note` and `reason` fields. These are experimental prompt families,
not production activation settings.

### Tulu prompt trial, October 28, 2026

Remote Tulu 3 8B was tested through the current iSPEC evaluator with temperature
zero, a 512-token output cap, strict protocol validation and the existing JSON
schema. Five saved private cases were replayed with two shorter prompts. Six
fresh synthetic cases were evaluated with the original and both shorter prompts:
28 calls total. Existing original-prompt results on private cases were reused,
not rerun. No Slack sends, project writes, service restarts or configuration
changes occurred. This is not an end-to-end Slack approval test.

The six synthetic cases are in
`tests/behavioral/data/note_proposal_minimal_20261028.json`. They test repeated
receipts, a new batch with the same count, a transfer plan, completed upload with
pending notification, a correction, and unrelated conversation. Expectations
and review criteria are excluded from model input. These are small, freshly
authored variations on known failure categories, not a representative benchmark.

| Prompt | Words | Correct labels on fresh cases | Valid field combinations on fresh cases |
| --- | ---: | ---: | ---: |
| Original | 131 | 5/6 | 4/6 |
| Minimal | 76 | 2/6 | 0/6 |
| Minimal plus status examples | 81 | 2/6 | 0/6 |

All 28 outputs were syntactically valid JSON. Label counts above include labels
inside otherwise rejected responses; they are not usable-response or grounding
scores. Short prompts often put explanations in `note` after choosing not to
propose. The unchanged validator rejects that field combination. They also
incorrectly treated a new batch, changed completion status and a correction as
already recorded. Status examples did not resolve these errors.

Eight additional diagnostic calls compared a one-sentence field explanation
against a conditional JSON schema on four private cases. Both variants produced
4/4 valid field combinations but only 2/4 correct decisions. The conditional
schema allowed a nonempty note for `propose` and forced an empty note for the
other labels. It was accepted by the current remote runtime and was used only
as a per-call test override; the default schema and validation remain unchanged.
This separates structural enforcement from factual judgment: fixing the former
did not fix the latter.

Private inputs, outbound schemas, raw outputs, timings and prompt hashes are
under ignored `tests/behavioral/local/slack-draft-20261028/minimal-ablation/`.
Initial trial calls took 1.35-5.28 seconds; warm-cache/order effects were not
controlled, so these are observations rather than comparative performance scores.
Keep the default prompt until further review. A useful next bounded test is
explicit separation of incoming evidence from stored comments, emphasizing new
events, status changes and corrections rather than adding prohibition lists.

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
