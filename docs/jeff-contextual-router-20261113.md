# Contextual Jeff Router: Offline Evaluation

## Current Experiment

The [frozen wording sweep](jeff-router-wording-20261113.md) completed all 444
combinations: 36 balanced fresh cases plus the historical save example, six
arms, both roles. N3 with draft-aware projects is the next offline-test candidate;
the historical save now ranks projects first. Choices are filtered before
scoring, and full-mass selection replaces the top-none veto. Remaining issues
include API-schema routing and unavailable general-user requests. No live
wiring or commits. The results below retain the earlier design.

## Earlier Contextual Replay

**Completed; do not wire Jeff in as an exclusive router yet.** The inputs now
contain the preceding exchange and authoritative prior draft state, but Jeff
still prefers `none` on several clear pending saves, revisions and cancellations.
The deterministic limiter/directory prototype works; actual OLMo directory
recovery remains untested. No production changes or commits were made.

One categorical question per input, not 13 independent relevance questions.
The staff pass has 13 categories plus `none`; the general-user pass removes six
staff-only categories and retains `none`. One resident Jeff v1.3 CPU model,
one compute thread and one interop thread, low OS priority. No thread-timing
probes, production changes, approval-threshold changes, or tool execution.

Both passes completed and the detached job exited zero. Its tmux session closed.
Results were saved after every completed prediction.
Four matching staff predictions from the interrupted first single-choice attempt
are reused with their source recorded; that attempt did produce partial scores.
The first two-pass process later ended with SIGTERM (exit 143, sender unknown)
after 60 saved staff scores. A detached local job resumed those scores, computing
the remaining six staff inputs and the general-user pass. The recovery change adds
validated resume support and atomic score-file replacement; questions, context
and selection rules are unchanged. The model reloaded once for recovery.

## Results

| Measure | Staff | General user |
| --- | ---: | ---: |
| Top-1 match to predeclared subject labels | 32/66 (48.5%) | 41/66 (62.1%) |
| Initial requests | 26/32 | 29/32 |
| Follow-up requests | 6/34 | 12/34 |
| Correct category in selection, among available category cases | 31/65 (47.7%) | 39/63 (61.9%) |
| Subject-label directory opportunities | 34 | 26 |
| Of those, category available in directory | 34 | 24 |
| Expected category unavailable under current scopes | 0 | 2 |
| Greeting correctly selects none | 1/1 | 1/1 |
| Observed median prediction wall time | 7.17 s | 3.44 s |

The two unavailable general-user cases are the person and experiment lookups.
On the common 65-category-case denominator, general-user inclusion is 39/65
(60%). Its raw top-1 subject score includes recognizing those unavailable topics;
that does not grant their tools. CPU contention, differing choice counts and the
resumed/reused observations prevent treating these timings as a controlled speed
comparison. There were no thread-timing probes in the completed two-pass run.

**Label limitation:** these labels name the conversation's subject, not a fully
adjudicated need for a tool on each turn. For example, wording-only agreement,
deferral, and thanks were labeled projects; `none` can be appropriate for them.
Do not read 34/26 directory opportunities as 34/26 proven behavioral errors.
The greeting is the only predeclared no-tool label. An always-projects classifier
would match 62/66 subject labels, illustrating the severe class imbalance. The
raw labels and scores are retained unchanged rather than relabeled to improve
the result. The concrete pending-action failures below are unambiguous enough
to block exclusive routing independently of that aggregate score.

### Cumulative Selection Curve

Tool cap 24, including the directory. Correct-category inclusion stays **31/65
staff and 39/63 general user in every row**; increasing mass or k never restores
the category when `none` is the highest eligible choice.

| Target mass | Max k | Staff mean tools | Staff mean schema tokens | General mean tools | General mean schema tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0.70 | 1 | 7.7 | 623 | 4.0 | 383 |
| 0.70 | 3 | 7.8 | 626 | 4.0 | 383 |
| 0.70 | 5 | 7.8 | 626 | 4.0 | 383 |
| 0.80 | 1 | 7.7 | 623 | 4.0 | 383 |
| 0.80 | 3 | 7.8 | 630 | 4.0 | 383 |
| 0.80 | 5 | 7.8 | 630 | 4.0 | 383 |
| 0.90 | 1 | 7.7 | 623 | 4.0 | 383 |
| 0.90 | 3 | 8.2 | 649 | 4.0 | 384 |
| 0.90 | 5 | 8.2 | 649 | 4.0 | 384 |
| 0.95 | 1 | 7.7 | 623 | 4.0 | 383 |
| 0.95 | 3 | 8.8 | 689 | 4.2 | 406 |
| 0.95 | 5 | 8.8 | 689 | 4.2 | 406 |

The cap-12 sweep also has unchanged coverage. Staff means stay 7.7 tools / 623
tokens for every target/k; the top category alone exceeds that soft cap in 29
inputs, and is retained as requested. General-user numbers match the cap-24
rows, with no cap exception. All 24 settings and per-input selections are in
each pass's `summary.json`.

The standalone directory is **166 schema tokens for staff, 112 for general
users** (descriptions alone: 95/56 tokens). The original 51-schema catalog is
4,097 compact-JSON tokens. These use the same real OLMo tokenizer; none of these
figures includes the application prompt or conversation. A 0.9 cumulative score
is a selection budget, not a demonstrated 90% statistical coverage guarantee.

### Concrete Behavior

Raw choice probabilities, before filtering/renormalizing unavailable categories:

| Request and actual context | Staff | General user | Interpretation |
| --- | --- | --- | --- |
| Look up project 57 | projects 0.938 | projects 0.959 | Correct subject |
| Look up person 73 | people 0.996 | people 0.997 | Correct subject; unavailable to general user |
| List experiment runs | experiments 0.863 | experiments 0.926 | Correct subject; unavailable to general user |
| Hi iSPEC, good morning | none 0.840 | none 0.863 | Appropriate no-tool choice |
| Save that exact draft now, pending draft present | none 0.815; projects 0.133 | none 0.677; projects 0.242 | Missed action category despite context |
| Drop this note entirely, pending draft present | none 0.785; projects 0.159 | none 0.720; projects 0.239 | Missed cancellation category |
| Change received to expected next week, pending draft present | none 0.759; projects 0.131 | none 0.640; projects 0.217 | Missed stored-draft revision category |
| Yes, that's the phrasing I wanted, wording-only context | none 0.793 | none 0.841 | Defensible none; subject label is too broad |

The failed explicit-save input was not an isolated "save it":

```text
User:
Prepare a note for project 57: seven serum specimens arrived on Monday;
sample preparation is still pending. Show me the draft before recording it.

Assistant:
Here is the draft note for project 57 before recording it:
"Seven serum specimens arrived on Monday; sample preparation is still pending."
You can now choose to Save or Cancel this draft.

Server draft state before latest reply:
Draft b3430abbd5b84b91813c735743c4b65b; status=pending; project=57
Seven serum specimens arrived on Monday; sample preparation is still pending.

Latest user request:
Save that exact draft to project 57 now.
```

Line wrapping above is for reading; exact input bytes are in the manifest under
key `6c94fc0ff9749e43bc59b8ea1d6aa5f8a58d8ed95f9fc0d17b81161d313c0ef3`.

## Integration Proposal, Not Implemented

1. Keep Jeff advisory until the clear pending-action misses are resolved. Its
   category selection must remain separate from authorization; it grants no
   permission to save a comment.
2. Register a single permission-filtered `open_tool_category` in place of
   `assistant_list_tools`. Resolve from the already-authorized schema registry,
   append newly revealed schemas for the next model step, and recheck current
   permissions when executing any tool. A routing failure starts with this
   directory, not the full catalog.
3. Preserve the controller's existing availability of authorized comment-workflow
   tools. A pending save/cancel/revision should not become inaccessible because
   an advisory router says none. This would be retaining existing behavior, not
   letting Jeff authorize or execute the operation.
4. Keep actual rendered-token budgeting as a hard guard. The user's top-category
   exception applies to the soft tool-count cap, not to the model context limit.
5. Before integration, run a small native OLMo directory-recovery test and review
   the project category wording. Its current description says create/list/count/
   search and comments/metadata, but never explicitly names prepare/revise/cancel
   of stored drafts. That is a specific hypothesis to test, not a demonstrated
   explanation or a prompt change made during this run. Add adjudicated no-tool
   labels and broader categories before reporting general routing accuracy.

## Context And Data

The question is: "Which tool category best fits the latest user request in this
conversation, or none?" The projects choice remains "Create/list/count/search
projects and project comments/metadata." The none choice is "No tool category
is needed to answer the latest request." Other choice descriptions and the full
Jeff input are preserved in the manifest; there was no prompt tuning mid-run.

Each input contains the latest request, up to two preceding user/assistant
exchanges, and the server draft snapshot from before the latest reply. Longer
conversations also retain the original user request. Quoted source context is
preserved. The current turn's response, write result and later draft state are
excluded. Missing preceding exchanges fail preparation rather than inventing
history. An aborted current request may be recovered from the saved case
definition, validated against the actual router request.

The existing saved runs produce 66 distinct contextual inputs: 62 project cases,
one people lookup, one experiment lookup, one scheduling question and one greeting.
Identical latest messages with different preceding histories stay distinct.
These are synthetic conversations with recorded model outputs, not collected
real Slack conversations or training data. The strong project skew limits any
claim of general category accuracy.

Historical OLMo router calls usually received only the latest request. Their
scores remain reference evidence, not a matched-input model comparison against
this context-enriched Jeff run. Several captured assistant replies were failure
fallbacks; those remain in the context exactly as delivered.

## Selection Rules

1. Renormalize probabilities over categories actually available in the catalog
   plus `none`. Unavailable categories receive no tool access.
2. If `none` is the highest eligible choice, select no category; the directory
   remains available. Otherwise include the highest eligible category, even if
   its tool count alone exceeds the soft tool-count cap.
3. Visit remaining choices in probability order. `none` contributes to cumulative
   coverage without adding a category or tool. Stop at the target coverage, max
   category count, or before an additional category would exceed the tool cap.
4. Count overlapping tools once and include the directory in every tool count.
   Record top-category cap exceptions explicitly. Token-window limits are a
   separate hard constraint that a future integration must retain.

For example, `projects=0.6, none=0.3, people=0.07, files=0.03` reaches a 0.9
target with only the projects category. A high-confidence single category is
retained even if its schema count exceeds the configured soft cap.

Curves use cumulative targets 0.7/0.8/0.9/0.95, category limits 1/3/5 and tool
caps 12/24. These are offline routing settings, unrelated to the comment
approval threshold. All curves reuse the same saved model scores.

## Directory And Permissions

The prototype `open_tool_category(name)` replaces `assistant_list_tools` and
returns schemas only from the supplied authorized catalog. It is not registered
as an application tool. The recorded staff catalog has nine available categories;
four other scored categories have no enabled tools in that snapshot.

The general-user simulation filters that snapshot with the existing `ToolScope`
public/user rules and hides assistant, devops, tmux, bridge, repo and automation.
Its available categories are projects, files, schedule and misc. Existing scopes
make the people and experiment lookup controls unavailable to this user; report
those separately from recoverable routing misses. This prototype neither creates
accounts nor changes permissions. Real account/project ACLs and deployment gates
must still run during actual tool execution.

"Directory needed" means the expected category was not preselected. "Recoverable"
means that category exists in the authorized directory. It does not establish
that OLMo will actually request it. Token costs use the real OLMo tokenizer on
compact schema JSON, including the directory; they are not full rendered prompts.

## Checks And Reproduction

27 deterministic tests pass, including prior-state context, no future-outcome
leakage, one model input, `none=0.3`, hidden mass, top-category retention, tool
deduplication, category/tool caps, non-staff filtering and immediate persistence.
Resume checks require matching question, context, checkpoint and role catalog.
The final focused run also included three existing tool-routing tests: 30 passed.

Original evidence root:
`tests/behavioral/local/router-repair-20261113/jeff-contextual-dual-catalog/`.
Resumed evidence:
`tests/behavioral/local/router-repair-20261113/jeff-contextual-dual-catalog-resumed/`.
Its manifest contains source receipts, exact questions/context, role catalogs and
source hashes. `runner.py` and `catalog.py` retain the frozen evaluation source.

```sh
nice -n 19 env CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m scripts.eval_jeff_tool_routing \
  --inputs tests/behavioral/local/staged-20261112/olmo-application-template-default/olmo/pytest \
           tests/behavioral/local/staged-20261112/olmo-integration-repair-v2/olmo/pytest \
  --checkpoint /mnt/e/tools/jeff/jeff-base-v1.3 \
  --catalog-wire tests/behavioral/local/staged-20261112/olmo-integration-repair-v2/olmo/pytest/test_application_conversation_3/wire.json \
  --output tests/behavioral/local/router-repair-20261113/jeff-contextual-dual-catalog-new
```

Add `--prepare-only` to inspect the inputs without loading weights. A separate
`--reuse-staff-results` option validates matching context/question/checkpoint
before reusing partial predictions; the report identifies reused observations.
Use `--resume-from <two-pass-evidence-directory>` to recover completed scores
for either role without repeating matching inputs.
