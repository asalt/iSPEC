# Prompt composition: inspect first, adapt later

This slice preserves component identity through rendering and records it at the
existing inference boundary. It does not introduce an agent framework, a graph
database, a new inference pass, or an adaptive policy. The live prompt wording,
tool permissions and approval decisions are unchanged.

## Objects before strings

`@prompt_binding` remains explicit function-to-family metadata, discovered by the
existing registry sync. It does not log a model call merely because a builder ran.
`RenderedPrompt` and `PromptComponent` are frozen, keyword-only dataclasses.

```python
@prompt_binding("assistant.answer.system", kind="system")
def build_answer(*, base: RenderedPrompt, response_format: str) -> RenderedPrompt:
    return load_bound_prompt(build_answer, values={
        "base_prompt": base,  # retain the object, not base.text
        "response_format_block": response_format,
    })

prompt = build_answer(base=base_prompt, response_format=format_text)
# Only the provider boundary needs prompt.text.
context = prompt_observability_context(prompt, extra={"stage": "answer"})
```

The example illustrates the boundary; the current service keeps its existing
builder names. Planner, answer, review and review-decider now retain their shared
base object. Other plain strings remain supported. Only components actually
inserted into a template are recorded, in placeholder order, including repeated
occurrences. Raw values are not recursively reinterpreted as templates.

The template `values` mapping is intentional: its keys match source placeholders.
It now accepts `Mapping[str, PromptValue]`, including `RenderedPrompt`, rather than
adding a chain of arbitrary `**kwargs`. Semantic helpers should use named typed
arguments. The existing observability `extra` dict remains a serialization boundary;
this change does not attempt a broad service/configuration refactor.

## What is recorded

Successful and failed inference usage events can carry `prompt_composition`:

```json
{
  "schema_version": 1,
  "coverage": "template_components_only",
  "nodes": [
    {
      "id": 0,
      "parent": null,
      "slot": null,
      "occurrence": 0,
      "family": "assistant.answer.system",
      "source_sha256": "<64-character source-body hash>",
      "version_num": 1,
      "binding": "ispec.assistant.service:_system_prompt_answer",
      "variant_id": null,
      "value_slots": ["response_format_block"]
    }
  ],
  "truncated": false
}
```

Children reference an earlier parent's local index and name the insertion slot.
IDs are per-call occurrences, not new global prompt families. The existing SQLite
registry remains the source of indexed versions; a null version means the exact
source body is not indexed. Each child keeps its own version and binding.

The graph is capped at 64 nodes and marks truncation. Log/export boundaries validate
and whitelist metadata. Invalid manifests are omitted from usage logs with
`prompt_composition_unavailable: true`, rather than preventing an inference call.
Support-chat trace entries also retain the composition.

No raw prompt body, template value, conversation text, or model response is added
to this graph. Existing call/session IDs link it to existing evidence where access
permits. Exports are still internal artifacts: identifiers are not anonymized.

Source hashes identify template versions, **not full assembled requests**. No new
rendered-prompt hash is introduced. Existing runtime override hashing is retained:
a full override clears displaced components; appended extras preserve children and
mark a `runtime_extra` value slot. Tool catalogs, history and other unconverted
strings are not falsely represented as tracked prompt nodes. This is not a complete
context-token accounting system or a lossless replay packet.

## Read-only graph viewer

Use a bounded inference-usage JSONL extract (at most 16 MiB). The exporter keeps
the latest 100 valid calls by default, accepts up to 500, reports skipped legacy
or malformed rows, and refuses to overwrite an existing output.

```sh
PYTHONPATH=src .venv/bin/python scripts/view_prompt_composition.py USAGE.jsonl \
  --output NEW-prompt-paths.html
```

The standalone HTML needs no server or network. Step through calls and click
components to inspect family, source hash, binding, version and offered variant.
It does not execute any prompt, tool, or logged content.

The synthetic behavioral fixture exercises actual rendering and usage logging with
stubbed HTTP responses. Its short base and acknowledgment prompts live under
`experimental.*`; none replaces the live base. Four manually selected variants
(`none`, `gotcha`, `understood`, `freeform`) demonstrate stable source identity with
different offered inputs. They are not model-quality or learning results.

```sh
.venv/bin/python -m pytest tests/unit/prompt \
  tests/unit/assistant/test_usage_logging.py \
  tests/unit/api/test_support_chat_prompt_observability.py \
  tests/behavioral/test_prompt_composition_sandbox.py --run-behavioral -q
```

## State, selection and possible learning

For now, call this **traceable prompt assembly**. "Stateful assembly" would mean
that a defined state snapshot influences the next composition. "Context-driven
node selection" would mean an explicit selector chooses among known components.
Neither persistent adaptive state nor such a selector is enabled by this slice.

An eventual bounded selection experiment could record a state/feature reference,
selector version, candidate IDs, selected ID, and human-rated outcome tied to a
call ID. Randomized experiments would also need selection probabilities for
meaningful policy evaluation. Keep feedback and policy versions separate from
template source versions; do not create a new family for every wording or turn.

"Emergent persona" is a hypothesis, not an automatic consequence of collecting
logs. Preference learning requires a feedback objective, an update rule and
held-out evaluation. Start with offline comparisons of brevity, relevance, status
fidelity and acknowledgment appropriateness, including no acknowledgment and
freeform alternatives. Do not reward agreeable-sounding text over factual accuracy.
An offered phrase is not proof the model used it, and a user's approval to save a
note is not automatically a positive style rating. User/task differences may make
one universal preferred persona inappropriate.

Any later style state can shape presentation, never permission checks, write
tickets or tool-result truth. No RL training or automatic preference updates are
implemented here. Narrated review should describe the tested slice, label synthetic
examples, and keep these future ideas visibly separate from completed behavior.

## Verification and review stopping point (2026-11-01)

162 focused tests passed across prompt parsing/sync/composition, service/provider
contracts, usage logging, support observability, response formatting, review and
project-comment approval policy, plus the stubbed composition scenario. Two existing
FastAPI lifecycle deprecation warnings remain. No real-model quality inference was
run for this metadata-preserving slice.

The synthetic export was browser-tested at 1440, 375 and 320 pixels, including
call navigation, selected-node details, and a 64-level nesting fixture. No
JavaScript errors or external requests occurred. Desktop and mobile screenshots
were visually checked. Deep branches reduce indentation rather than overflowing.

Local review artifacts are under
`tests/behavioral/local/prompt-composition-review-20261101/`; generated HTML,
screenshots and usage records remain ignored. `prompt-paths-reviewed.html` is the
four-call demonstration. No production configuration, services or databases were
changed. The concise introduction and acknowledgment choices remain experimental;
the next behavioral step is a reviewed, held-out comparison, not automatic adoption
or persona learning. Narration remains deferred while that next slice is unsettled.

The subsequent isolated real-model comparison is recorded separately in
[prompt-composition-evaluation-20261101.md](prompt-composition-evaluation-20261101.md).
It tests base wording, system-message layout and two-turn continuity; it does not
retroactively assign component graphs to old logs or activate the experimental
base. Its saved responses include failures, not just successful demonstrations.
