# Staged comment fixes: controlled trials, November 6

Development candidate only. No deployment, production configuration change,
Slack send, legacy write, model download or permission/parser relaxation.
The behavioral failures are not all fixed. A false approval remains a blocker.

## Retained Implementation

- Successful preparation renders the exact registered text and yields the turn
  to the user. The neutral presentation says it is not saved; it does not ask
  for permission on behalf of a wording-only request.
- Save/cancel results and concrete errors produce receipt-backed final status.
  Raw model responses remain in traces and presentation metadata for review.
- Ambiguous pending state does not select the newest draft. Unknown IDs are
  rejected without silently substituting another draft. Later-turn, scope,
  permission, expiry, classifier and atomic write checks remain intact.
- Classifier results reach the model as a short system message in own mode,
  not shadow mode. Only model-requested saves invoke the check in retained code.
- Tool descriptions clarify registration/revision/cancellation. The prompt and
  comment argument ask for project facts rather than unrelated conversation.
  These wording changes have not reliably solved model tool selection/grounding.
- Assertions now include preparation, exact presentation, revision IDs and
  supersession, actual cancelled state, facts and unexpected save attempts.
  Responses repaired/rendered by the server are not credited as raw model success.

## Fresh Results

| Trial | Cases/model | Tulu passed | OLMo passed | Interpretation |
| --- | ---: | ---: | ---: | --- |
| state-routed-trial | 6 | 3 | 1 | Rejected automatic save dispatcher and expanded workflow prompt |
| exact-presentation-trial | 6 | 2 | 0 | Model-requested saves restored; repeated model omissions remained |
| yield-on-prepare-trial | 8 | 3 | 2 | Preparation yields; clearer tool descriptions; two fresh cases added |
| neutral-presentation-control | 4 | 3 | 2 | Only the presentation invitation removed; approval controls retested |

These are stricter tests than the earlier 6/11 baseline, so pass totals are not
directly comparable. Runs are sequential, temperature zero, but not repeated
statistical estimates or isolated single-variable comparisons except the last
presentation change. Most cases have now been seen repeatedly.

Latest Tulu: explicit save and natural approval prepared and saved correctly.
The old wording-only case still saved incorrectly after the main model called
save and its classifier approved. The fresh wording-only variation did not
save, but its response was awkward. Removing automatic routing and neutralizing
the presentation were both insufficient to fix the unsafe case.

Latest OLMo: both wording-only cases avoided saves. Explicit and natural approval
failed because the first reply claimed/prepared only in prose; the subsequent
save had an unknown ID. The executor correctly rejected it before classification.

The immediately preceding eight-case trial also found unregistered revisions
in both models and missed cancellation. Tulu saved WiFi/IT chatter in the
mixed-thread note: this is a grounding failure, not a missed save.

One OLMo fresh-approval failure was an assertion defect: the correct saved phrase
`No runs have started` was missing from acceptable status alternatives. That
fixture was corrected after the trial; saved JUnit evidence was not rewritten
and that conversation was not rerun. Semantic review supports its save outcome.

## Rejected Route and Next Decision

The first trial routed replies to any unambiguous presented draft through the
classifier and dispatched approved saves. Tulu returned 0.90 for wording-only
agreement and wrote a fixture comment. That dispatcher was removed, not enabled
behind a default-on setting. The broader wording-only boundary remains unsafe
even when both main model and classifier agree.

Do not promote this candidate. The next bounded design question is how to retain
the distinction between wording review and a pending save approval explicitly,
without inferring it again from assistant prose or introducing semantic regex.
Any proposed state distinction needs tests for transitioning from wording-only
review to an explicitly requested save, denial, revision and ambiguous agreement.
Missing preparation/revision calls and source grounding remain separate tasks;
adding more save-approval language is not a demonstrated solution for them.

## Evidence and Verification

All evidence is under `tests/behavioral/local/staged-20261106/` in the four trial
directories above: exact source snapshots, hashes, launcher copy, runtime logs,
JUnit, conversation state and wire requests/responses. Each receipt reports no
source changes during its run. Models were loaded once per batch; no repeated
weight load per test. The final narrow trial used four conversations per model.

`fix-trials.html` compares all attempts; `neutral-presentation-control/postmortems.html`
contains only the latest four cases per model. Review notes distinguish false
saves, missing calls, semantic failures and the one assertion defect.

175 focused unit/integration tests passed, including concurrent saves, read-only
deployment, shadow/off modes, actor/session scope, stale IDs and same-turn saves.
Ruff and whitespace checks passed. The original five omitted regression cases
were not rerun against the final presentation; no full-suite success is claimed.

All owned model processes exited and test ports 18030/18031 closed. Final host
GPU query showed no compute processes. The pre-existing Ollama listener on
11434 was left intact. Trials used isolated SQLite fixtures and loopback-only
inference guards; no production or legacy records were touched.

Reproduce the last control batch in a new output directory:

```sh
../.venv/bin/python tests/behavioral/local/staged-20261106/run.py \
  tests/behavioral/local/staged-20261106/new-neutral-control \
  'explicit_save or natural_approval or wording_only'
```

For future batches, a separate monitor can watch the existing PID/log/sentinel
and report completion or stalls while the owner works. It should not start a
duplicate run, retry a failed batch or change services. The launcher remains
responsible for process cleanup; a single long wait is enough when there is no
other work to parallelize.
