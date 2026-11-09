# OLMo Think router budget pilot, 2026-11-01

## Question and controls

Does the existing iSPEC router produce a usable, correct category decision when
OLMo Think has 512 or 1,024 output tokens instead of the fixed 200-token cap?
This isolates the budget issue from planner prompts, tool execution, approvals,
CPU-router architecture and end-to-end answer quality.

`scripts/eval_router_budget.py` calls the existing `route_tool_groups_vllm` and
strict `service.generate_reply` adapter. An evaluation-only callback changes
`max_tokens`; prompt, groups, JSON schema and temperature zero remain unchanged.
There is no planner or tool executor in this path. Production code/configuration,
write permissions, model/template and drivers are not changed.

The six cases are frozen before inference:

| Case | Expected primary | Origin |
| --- | --- | --- |
| Mixed project update and printer chatter | projects | Exact saved synthetic router input |
| Prior quoted approval, current no-action request | projects | Exact saved synthetic router input |
| Explicit project lookup | projects | Exact saved synthetic router input |
| Greeting, no task | assistant | Exact saved synthetic router input |
| Find Mira Example's person record | people | New synthetic control |
| List experiment runs for a project | experiments | New synthetic control |

These are capability categories, not authorizations or commands. In particular,
the current schema has no `none` category: `assistant` is the greeting control's
expected routing label, not proof that retrieval tools should run. The no-action
project example tests subject routing, not approval classification.

The four saved messages and schemas must match today's actual router byte-for-byte
before inference. Both budgets receive identical inputs. Budget order alternates
by case to reduce systematic warm-cache bias. One sample per case/budget is a
diagnostic pilot, not a robustness or calibrated-confidence benchmark. A usable
result requires provider success, an unmodified full-schema-valid JSON response,
a validated router decision and no tool calls. Truncated/invalid replies are not
repaired or credited. Category correctness is scored separately from validity.

## Runtime and isolation

Local Quadro RTX 6000, driver 595.112.07; initial occupancy check: no GPU compute
jobs, 22,869 MiB free, test port 18030 closed. Existing vLLM 0.32.0, FP16,
8,192-token context, eager/TRITON_ATTN, one sequence and 0.80 GPU memory fraction.
Cached `allenai/Olmo-3-7B-Think` revision
`d97e442d7cc678210054dbcc9b440894d62c89a4` is read from the existing amms06 mount.
The tools-aware evaluation template from the previous experiment is unchanged;
this is not stock-template validation. No downloads or installations.

The launcher starts a clean evaluator child environment without inherited
credentials or live DB paths. The network guard permits only the explicit
loopback health/model/completion endpoints, disables redirects/proxy environment
use, and rejects authorization headers and tool schemas. Raw synthetic request
and response bodies are retained, not headers or process environments. No Slack,
legacy writes, real project reads, or production service changes are involved.

## Results

All twelve requests completed at HTTP 200. Every paired wire request was checked:
only `max_tokens` differs. Input sizes were 277-580 tokens, far below the
8,192-token context window. There were no tool schemas or tool execution.

| Output cap | Cases | Truncated | Finished | Full-schema usable | Median router time |
| --- | ---: | ---: | ---: | ---: | ---: |
| 512 | 6 | 4 | 2 | 0 | 18.18 s |
| 1024 | 6 | 2 | 4 | 0 | 25.41 s |

| Case | 512 | 1024 |
| --- | --- | --- |
| Mixed project/printer thread | Truncated | Truncated |
| Quoted old approval, current no action | Truncated | Finished, wrong fields |
| Explicit project lookup | Truncated | Truncated |
| Greeting | Truncated | Finished, wrong fields and `other` category |
| Person lookup | Finished, `group` alias only | Finished, `group` alias only |
| Experiment runs | Finished, `selected_groups` only | Finished, `selected_groups` plus explanation |

Increasing the budget lets two more cases finish but does not make the current
integration reliable. The completed person and experiment answers identify the
intended subject, and the completed quoted-approval answer identifies projects.
That is evidence of some classification ability, **not** successful adherence to
the requested contract. The schema requires `primary`, `secondary`, `confidence`
and `clarify`, with no additional fields. The full schema was present in every
captured request's `structured_outputs` field, yet the completed outputs violate it.

The existing router's compatibility validator can normalize `{"group":"people"}`
and supply defaults, including confidence 0.7. The evaluation records that
normalized decision separately but does **not** count it as schema-compliant or
as model-reported confidence. Neither normalization nor an HTTP success should
hide the output-contract problem. No validators or permission gates were relaxed.

Startup/model-loading time is excluded. These are twelve sequential pilot calls,
not a stable throughput benchmark. The runtime's reasoning-token counter reported
zero even when a reasoning field existed, so completion-token totals, finish
reasons and raw response fields are retained rather than trusting that breakdown.

## Separate integration finding

A read-only check of the installed vLLM 0.32.0 OLMo parser and cached tokenizer
reproduced a boundary mismatch:

| Encoded text | Token IDs | Token-level end detected | Text-level end detected |
| --- | --- | --- | --- |
| `</think>` | 524, 27963, 29 | yes | yes |
| `</think>` followed by two newlines | 524, 27963, 1363 | no | yes |
| Space + `</think>` + two newlines | 694, 27963, 1363 | no | yes |

The installed `olmo3_reasoning_parser.py:is_reasoning_end` matches the exact
closing-token sequence ending in token 29; the tokenizer can merge that character
with newlines into token 1363. Meanwhile `extract_reasoning` recognizes the
delimiter in string space. vLLM's structured-output manager uses the token-level
predicate to decide when grammar constraints start. This makes the mismatch a
concrete candidate explanation for final JSON escaping the requested schema.

**Limit:** these are canonical tokenizer encodings, not token IDs captured from
the twelve generated responses. We have reproduced an interface mismatch, not
proven it caused each observed schema violation. No installed parser/runtime was
patched. The local `check_reasoning_boundary.py` and `boundary-check.json` preserve
the reproduction without model inference. Its sandbox NVML warning is unrelated
to the tokenizer/CPU check and is not evidence of a failed GPU runtime.

## Recommendation and stopping point

Keep Think experimental and production/fallback unchanged. Do not raise all
budgets globally or introduce a CPU-router architecture on these results.

The next discriminating test is one short case that already finishes within 512
tokens: capture its generated token IDs and verify the reasoning-to-schema handoff.
If the mismatch is confirmed, test a narrowly scoped evaluation-only fix while
preserving exact JSON validation, then repeat this frozen budget comparison.
The two cases still exhausting 1,024 tokens remain a separate reasoning-cost
problem; repairing output formatting will not automatically shorten them.

No positive approval behavior, planner output or full assistant answer was tested
here. A router category never grants authority. There were no external sends,
live database writes, model downloads, new installations or production changes.
The owned server has exited; independent host verification found no GPU compute
jobs and loopback port 18030 closed. No commits were made.

## Reproduction

With an explicitly started isolated Think runtime using the same template/parser:

```sh
.venv/bin/python scripts/eval_router_budget.py \
  --source tests/behavioral/local/composed-models-20261101/tulu-ready/baseline/pytest.log \
  --url http://127.0.0.1:18030 --model olmo-think-router-eval \
  --output-dir tests/behavioral/local/NEW-router-budget-run
```

The selected source is synthetic and explicitly named, not a bulk history import.
The script freezes `cases.json` and source hashes, and writes `results.jsonl`,
`summary.json`, `worker.log` and `run.json`. The output directory must be new.
The already-installed JSON Schema validator is declared in test extras for
reproducible local tests; production dependencies are unchanged.

Evidence for this run is under `tests/behavioral/local/router-budget-20261101/`.
The one-off `run_owned.py` reuses the previous owned-service lifecycle helpers;
`runtime/receipt.json` records startup/cleanup and `runtime/snapshot.json` records
installed versions and the checkpoint. It is not a new production service manager.

Focused tests: 114 passed across the evaluator, routing, runtime/template
contracts and approval controls, with two existing FastAPI lifecycle warnings.
`git diff --check` passed. Unrelated dirty backend/deployment work was preserved.
