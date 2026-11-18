# Focused Staff API Routing Fix

## Outcome

The missing `misc` category is now included for M02. On the focused ten-case
staff set, required-category inclusion improves **7/9 -> 9/9**; top-choice
accuracy improves **7/10 -> 9/10**. All four experiment-record controls still
rank experiments first. Pending-save and wording-only controls are unchanged.

M02 remains a ranking ambiguity: experiments 0.545, misc 0.422. At the unchanged
0.9 cumulative target, both categories are selected. This repairs the missing
API tool exposure, not the top-choice judgment. An actual OLMo tool-use test must
still verify that it calls `search_api`, not an experiment-record lookup.

This is an offline catalog repair only. No live routing, commits, production
changes, or tool execution. General-user unavailable capabilities remain filtered
and are deferred to the [OLMo directory-recovery proposal](olmo-directory-recovery-proposal-20261113.md),
not further Jeff wording tuning.

## One Description Changed

`misc` owns `search_api` in the application catalog. `experiments` owns actual
experiment/run retrieval. The requested operation determines ownership, not the
record type mentioned in an API question.

Old `misc`:

> Other utilities (API schema search, DB file stats, recent activity).

New `misc`:

> Inspect API schemas and documentation; find endpoints, operations, accepted parameters and request/response fields for any record type; DB file stats and recent activity.

Implemented in the offline `routing_question` helper in
`scripts/eval_jeff_tool_routing.py`. N3, the draft-aware projects description,
all other choices, the complete input context, model/checkpoint and selection
budgets are unchanged. No keyword-based routing or question-specific exception.
The application `_GROUP_DEFS` and production router remain untouched by this fix.

## Focused Results

Eight earlier N3 staff predictions reused exactly; two fresh cases frozen and
labeled before scoring. Twenty comparisons, twelve new predictions, one CPU
model load / one compute thread / low OS priority. Exit zero; owned tmux/model
closed. Frozen source hashes match current source after the run.

| Case | Expected | Old top / selected | New top / selected |
| --- | --- | --- | --- |
| E01 project experiments | experiments | experiments / experiments | experiments / experiments |
| E02 run number/search | experiments | experiments / experiments | experiments / experiments |
| E03 latest runs | experiments | experiments / experiments | experiments / experiments |
| M01 password-change API | misc | misc / misc | misc / misc |
| M02 experiment-run API schema | misc | experiments / experiments | experiments / experiments + misc |
| M03 project-comments API | misc | projects / projects + misc | misc / misc + projects |
| P02 pending draft save | projects | projects / projects | projects / projects |
| Z03 wording-only agreement | none | none / empty | none / empty |
| Fresh person API parameters | misc | people / people | misc / misc |
| Fresh run label/reference DB | experiments | experiments / experiments | experiments / experiments |

Concrete probabilities:

| Input | Old misc | New misc | Old experiments | New experiments |
| --- | ---: | ---: | ---: | ---: |
| M02 | 0.023 | 0.422 | 0.910 | 0.545 |
| E01 | 0.0005 | 0.0002 | 0.980 | 0.981 |
| E02 | 0.0045 | 0.0015 | 0.973 | 0.973 |
| E03 | 0.0027 | 0.0006 | 0.995 | 0.997 |
| Fresh run metadata | 0.021 | 0.012 | 0.899 | 0.902 |

Fresh schema case: "Find the API endpoint and request parameters used to retrieve
a person record." Misc improves 0.048 -> 0.922 and becomes top choice.

Fresh record case: "For experiment run 764, fetch the recorded label type and
reference database." Experiments remains top and selected alone.

M02 exact request: "Which API operations accept an experiment run identifier?
Inspect the schema." It is a **staff** case; the earlier general-user selection
already included misc. No general-user inference was repeated here.

All cases are synthetic/hand-authored evaluation data, with the two fresh cases
labeled separately from retained cases. No real collected messages or training
data were added. This targeted result is not a rerun of the full 36-case set.

## Evidence And Next Test

Evidence: `tests/behavioral/local/router-repair-20261113/jeff-schema-followup/`
contains the frozen manifest, exact questions/cases, source hashes, scores,
per-case selections, `run.log`, and zero `run.exit`.

```sh
python -m scripts.eval_jeff_router_wording prepare-schema --output <new-directory>
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m scripts.eval_jeff_router_wording run --output <new-directory>
```

39 deterministic tests pass; Ruff and diff whitespace checks pass. The added test
verifies only misc wording changes, old evidence remains intact, cases are
bounded, and eight exact baseline inputs are reused.

Next proposed step: eight real OLMo conversations through isolated `support.chat`
with a turn-local authorized directory adapter. Test correct selection, deliberately
omitted authorized categories, unavailable general-user capabilities, project ACL
denial, and a greeting. No OLMo process or integration has been started here.
