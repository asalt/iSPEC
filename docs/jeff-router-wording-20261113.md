# Jeff Router Wording Sweep

## Status

**Focused follow-up:** the [staff schema-description repair](jeff-schema-routing-20261113.md)
now includes misc for M02, with four experiment controls unchanged. M02 still
ranks experiments first. Only that description changed; the six-arm sweep below
is retained as historical evidence. Unavailable general-user requests move to
the [proposed OLMo directory test](olmo-directory-recovery-proposal-20261113.md),
not another Jeff wording sweep. No live wiring or commits.

**Completed: recommend N3 plus draft-aware projects for the next offline
integration test.** It has the best supported-request/no-tool balance, not a
universal win on unavailable requests. The historical save now ranks projects
first in both roles. This is not a production recommendation.

444 combinations completed: 443 new predictions and one exact historical reuse.
Exit zero; the owned tmux session closed and the model exited. One resident Jeff
v1.3 CPU model, one compute thread, low OS priority; no tool execution, controller
wiring, approval-threshold changes, commits, deployment or production changes.
New inference took 2,236 seconds under competing CPU load, not a speed benchmark.
No wording changed during the batch. All frozen source/fixture hashes verified.

## Results

The new selection rule already recovers the old save case with baseline wording.
**Wording fixes its ranking too:** N3 lowers staff `P(none)` from 0.815 to 0.147,
general user from 0.713 to 0.189. Just adding draft verbs (N0) reduces it further,
to 0.111/0.099, but causes more unnecessary categories on genuine no-tool requests.
Thus the projects description itself accounts for a substantial improvement.

N3: **No lookup, draft, save, or other action is needed; a direct reply is enough.**

All means exclude the separate historical case. Tool-request denominators are
26 staff / 14 general; genuine-none denominator is ten in each role. The other
12 general-user requests concern unavailable capabilities. Schema-token counts
use the real OLMo tokenizer on compact JSON, not a full prompt. Costs include
the directory and four authorized comment tools whenever a draft is pending.

### Staff

| Arm | Top correct | Category included | Mean none: tool | Mean none: genuine none | Empty on none | Mean categories | Mean tools | Mean tokens | None: historical save |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 26/36 | 25/26 | 0.267 | 0.915 | 7/10 | 0.92 | 6.83 | 629 | 0.815 |
| N0 | 29/36 | 25/26 | 0.201 | 0.836 | 5/10 | 0.94 | 7.53 | 679 | 0.111 |
| N1 | 28/36 | 25/26 | 0.232 | 0.861 | 6/10 | 0.92 | 7.44 | 673 | 0.155 |
| N2 | 30/36 | 25/26 | 0.105 | 0.725 | 2/10 | 1.06 | 7.86 | 702 | 0.016 |
| **N3** | **33/36** | **25/26** | **0.118** | **0.883** | **8/10** | **0.89** | **7.03** | **645** | **0.147** |
| N4 | 30/36 | 25/26 | 0.005 | 0.526 | 2/10 | 1.08 | 7.94 | 707 | 0.0004 |

### General User

| Arm | Top correct | Category included | Mean none: tool | Mean none: genuine none | Empty on none | Mean categories | Mean tools | Mean tokens | None: historical save |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 30/36 | 14/14 | 0.364 | 0.902 | 7/10 | 0.92 | 3.47 | 329 | 0.713 |
| N0 | 30/36 | 14/14 | 0.246 | 0.817 | 5/10 | 0.97 | 3.75 | 354 | 0.099 |
| N1 | 31/36 | 14/14 | 0.275 | 0.834 | 6/10 | 0.94 | 3.72 | 353 | 0.143 |
| N2 | 28/36 | 14/14 | 0.155 | 0.747 | 4/10 | 1.03 | 3.92 | 369 | 0.035 |
| **N3** | **27/36** | **14/14** | **0.184** | **0.864** | **8/10** | **0.94** | **3.72** | **353** | **0.189** |
| N4 | 21/36 | 14/14 | 0.021 | 0.569 | 2/10 | 1.33 | 4.78 | 443 | 0.0009 |

N1 wins general-user top-1 when unavailable requests are included (31/36 versus
N3 27/36). Among the 24 available-or-genuine-none cases, N1 and N3 both score
21/24; baseline scores 19/24. N3 leaves more genuine none requests without a
selected category (8/10 versus N1 6/10) and lowers none on supported actions.
This motivates N3 as the next-test candidate, not a win on every metric. N4
eliminates none on actions partly by losing genuine no-tool cases. N2 has a
similar, smaller over-selection tradeoff.

### Pending Actions And Limits

Fresh prepare/save/revise/cancel requests rank projects first with N3 for both
roles. These are routing outcomes, not executed writes or approval tests.

| Case | Staff none baseline -> N3 | General none baseline -> N3 |
| --- | ---: | ---: |
| P01 prepare a draft | 0.436 -> 0.116 | 0.476 -> 0.192 |
| P02 save this exact note now | 0.776 -> 0.335 | 0.723 -> 0.287 |
| P03 revise four to six aliquots | 0.839 -> 0.328 | 0.705 -> 0.235 |
| P04 cancel the pending draft | 0.766 -> 0.477 | 0.637 -> 0.399 |
| Historical save project 57 | 0.815 -> 0.147 | 0.713 -> 0.189 |

- **Every staff arm misses M02:** "Which API operations accept an experiment run
  identifier? Inspect the schema." It selects experiments, not misc/schema
  search. This is subject-versus-requested-work confusion. The correct category
  remains available in the directory; actual OLMo recovery is untested.
- **Two extra selections remain with N3:** Z08 asks for wording help only and Z09
  leaves a draft pending without changes. Both expose projects in each role.
  Staff none remains top (0.704/0.527), but the 0.9 cumulative target includes
  projects. General-user Z09 ranks projects first, none 0.468. None of this
  authorizes a write; the approval pathway was not changed or evaluated.
- **Unavailable requests remain problematic:** all 12 unsupported general-user
  requests select some available category with N3. Baseline/N0/N1 leave one of
  12 empty; N2/N4 leave zero. Hidden schemas are never exposed, but an unrelated
  available category is not an adequate availability response. This warrants a
  controller-level limitation/recovery test before any live integration.

The historical general-user save baseline is now 0.713 rather than the earlier
0.677 because scoring choices are filtered first. Staff baseline is identical
to the earlier request and is the single cached prediction. Full per-case
selections, probabilities, timing and costs are in `summary.json` / `results.json`
beside the frozen manifest. This balanced corpus is still small; selection from
a wording sweep needs independent validation, not a claim of general accuracy.

## Frozen Evaluation

Labels were written before inference: 36 hand-authored, synthetic evaluation-only
cases, including ten genuine no-tool requests. There are three cases each for
schedule, files, omics, people, experiments and misc; three staff capabilities;
five projects cases including prepare/save/revise/cancel; and ten greetings,
thanks, wording-only, deferral or general-knowledge requests. One historical
explicit-save context is added separately, with its recorded model reply intact.
None of these are real collected conversations or training examples.

Six wording arms, both roles, 444 combinations. Exact-input caching preserves
choice order and context; historical scores are reused only when the complete
model request and checkpoint match. The model is loaded once for the batch.
All case labels, complete questions and source hashes were frozen in:

`tests/behavioral/local/router-repair-20261113/jeff-wording-sweep/manifest.json`

The fresh fixture is
`tests/behavioral/data/jeff_router_wording_20261113.json`. This folder is ignored
by the current repository rules, so this explicitly synthetic fixture will need
deliberate inclusion in a future reviewed commit. Nothing has been staged.

The question remains: **Which tool category best fits the latest user request
in this conversation, or none?**

Baseline projects: `Create/list/count/search projects and project comments/metadata.`

New projects: `Create/list/count/search projects; draft, revise, cancel, or save project notes/comments; project metadata.`

| Arm | Projects description | None description |
| --- | --- | --- |
| baseline | Old | No tool category is needed to answer the latest request. |
| N0 | New | No tool category is needed to answer the latest request. |
| N1 | New | No tool category is needed to help fulfill the latest request. |
| N2 | New | The latest request can be answered from the conversation alone, without looking up or changing any data. |
| N3 | New | No lookup, draft, save, or other action is needed; a direct reply is enough. |
| N4 | New | Only greetings, thanks, or small talk. |

## Selection And Authorization

The new scoring choices are filtered **before inference**. The staff fixture is
feature-complete (13 categories), using static application schemas, with no live
account or feature configuration changes. The general-user fixture applies the
existing public/user tool scopes and has four categories: projects, files,
schedule and misc. People, experiments and omics are not offered. Requests for
unavailable capabilities are reported separately from genuine no-tool requests.

Selection sorts all choices and accumulates their full mass, including `none`.
Stop at 0.9, three real categories, or 24 tools including the directory. The top
real category is retained even if it exceeds the soft tool cap. `none` being
highest is not a veto: selection is empty only when its mass reaches 0.9.
An authorized directory schema is always present. Pending drafts additionally
retain the four authorized comment-workflow tools, deduplicated with selections.
These are exposure estimates; no authorization or execution is performed here.

The historical limiter-only replay keeps the older eligible-catalog normalization
so that only selection changes. It gives **64/65 staff and 63/63 general-user
category inclusion**, versus 31/65 and 39/63. Staff top non-none subject matches
are 63/66. Mean categories are 1.30/1.08 with the older catalog.

The remaining staff miss is N03 wording-only agreement after an earlier failed
assistant reply. `none` contributes 0.804, then assistant 0.087; adding projects
would exceed the 24-tool cap. It therefore selects assistant, not projects. That
old projects label names the subject rather than a demonstrated need for a tool.
With the full synthetic staff catalog the result is still 64/65, mean 1.36
categories. Thus the reported 65/65 staff figure was not reproduced under the
specified cap; this is not a new model judgment or a reason to change the cap.

## Analysis Plan

For each arm and role, report tool-request category inclusion, mean `P(none)` on
available tool requests and on the ten genuine no-tool cases, top category
accuracy, empty selections on genuine none, average categories/tools/schema
tokens, and the exact historical save case. Report unsupported requests
separately. Keep the balanced 36-case averages separate from the historical case.
Choose wording by coverage plus separation, and state tradeoffs if no arm wins
both. No probability reweighting or threshold tuning during this experiment.

The directory and always-present pending tools make a missed category less
costly, but this replay cannot establish that OLMo will recover through the
directory or actually perform the correct action. That needs a later bounded
controller test, separately authorized.

## Verification And Reproduction

35 deterministic offline tests passed before launch; the post-run check including
existing tool-routing tests passed 38. Ruff passed on the five
changed evaluation/selector/test modules. Tests cover pre-scoring role filtering,
full none mass, the historical 0.815-none save distribution, top-category tool-cap
exception, pending-tool retention/deduplication, and exact-input caching.

```sh
python -m scripts.eval_jeff_router_wording prepare --output <new-evidence-directory>
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m scripts.eval_jeff_router_wording run --output <new-evidence-directory>
python -m scripts.eval_jeff_router_wording summarize --output <new-evidence-directory>
```

Use the frozen source copies for exact reproduction after later edits. Running
against the same evidence directory resumes exact completed inputs. The completed
batch used tmux `ispec-jeff-wording-20261113`, now closed; `run.log` and `run.exit`
are in its evidence directory. Scores were checkpointed after every completed input.
