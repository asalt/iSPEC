# Evaluation Checkpoints

## Current Status

Updated November 12, 2026. This section is the current handoff; linked reports
retain historical results. Update this section after each run.

- **Design:** one `prepare_project_comment(project_id, comment)` call registers
  the draft. The model shows it; the server records exact presentation. A later
  model-requested save invokes the approval classifier and existing guards.
  Refactor checkpoint: `19e2c5d`; controller docstrings: `3cdf122`. No second
  offer tool or server-written replacement assistant reply.
  See [workflow](staged-project-comments.md).
- **Blockers:** showing draft text does not establish intent to save. Wording-only
  agreement can reach the classifier if the model requests a save. N02-N04's
  historical false approvals remain untested on this refactor. Also open:
  false cancellation claims with pending drafts, prose-only revisions, and
  R01/N08/N16/N18 grounding failures.
- **Last refactor batch:** OLMo 4/7, two approved saves, zero observed unapproved
  writes; only the two true approvals reached the classifier. This is not
  negative-approval safety evidence. Tulu 0/7 and excluded from further runs
  unless Alex asks. [Results](single-preparation-retest-20261112.md).
- **Latest behavioral run:** the Jeff tool-use trial passed three no-action
  controls and missed three required actions; zero advisories fired, about
  7.5 seconds per check. Retired by Alex: no further tuning or integration.
  This is separate from the save-approval classifier.
  [Evidence](jeff-tool-use-advisory-20261112.md).
- **Harness repaired:** cancellation requires the original draft to be cancelled
  in storage and a successful matching tool call; revision requires a new
  registered/presented draft, a superseded prior draft and matching preparation.
  Negative diagnostics now run on pre-turn state, remain separate from all
  actual checks, and report completed/failed/skipped inference. A diagnostic
  ticket cannot authorize execution. These are test changes, not model fixes.
- **Latest verification:** 190 focused unit/integration tests passed;
  `git diff --check` passed. No model inference ran. Passing stubs do not resolve the
  behavioral blockers above; two existing FastAPI deprecation warnings remain.
- **Next:** run the committed refactor/harness frozen with OLMo only: N01-N04,
  N13, N15, N18, N19, deny/cancel, explicit/natural approval, revision and
  printer-thread cases; include N08/N16 for count/date/QC grounding. The new
  behavioral batch has not run yet.
  Report structural remedies for approval intent and truthful state display
  before implementation; no regex claim detection or silent prose replacement.
- **Boundary:** Alex approved three commit groups and focused harness repairs.
  No prompt/threshold tuning, service restarts or deployment. Retired Jeff
  prototype code is excluded from these commits; its result report is retained.
  Production remains Tulu with comment writes enabled per Alex's handoff;
  production was not inspected or changed during this assessment.

## Historical Checkpoints

The November 8 organization checkpoint preserved experiments without selecting
a model or authorizing deployment. The [November 7 save-offer gate](save-offer-gate-20261107.md)
describes the superseded separate-offer design, not today's controller.
Private transcripts, generated reports, SQLite databases and runtime receipts
stay under ignored `tests/behavioral/local/`.

## Retained Experiments

- [Jeff approval](jeff-approval-prompts.md), [choice results](jeff-choice-evaluation-20261105.md)
  and [replay](jeff-approval-replay.md): offline probability/threshold studies with
  editable prompts. The adapter does not issue tickets or execute writes; it is
  not wired into production approval.
- [Prompt composition](prompt-composition-evaluation-20261101.md): comparison of
  baseline, concise and compact layouts, separate from provenance logging itself.
- [Router budgets](router-budget-evaluation-20261101.md): inference-only probes of
  output contracts and token budgets, not permission or parser changes.
- [Slack proposal prompts](ollama-tool-routing.md#tulu-prompt-trial-october-28-2026):
  explicit candidate prompts, unchanged validation and no posting/saving.
- [Thread grounding](local-thread-model-evaluation-20261030.md),
  [two-turn drafts](draft-conversation-evaluation-20261105.md) and
  [model regression](model-regression-20261105.md): historical controller evidence.

The last group predates the registered-draft flow. In particular,
`test_draft_approval_conversation_live.py` patches the now-removed
`support.run_project_comment_approval_classifier`, and older controller fixtures
seed assistant prose rather than registered drafts. `run_model_regression.py`
includes those probes. Use their saved source snapshots for historical replay;
do not interpret failures against today's controller as fresh model-quality
evidence. Porting them to the new contract is separate behavior work, not part of
this commit cleanup. The shared isolation fixture and raw template tests remain
useful independently.

## Validation Boundary

Unit tests use stubbed inference and synthetic storage. Passing them validates
the experiment machinery, not model behavior. No fresh model inference, downloads,
production activation, Slack sends or driver changes are part of this checkpoint.
Runtime launchers require already installed software/checkpoints and explicitly
requested execution. Inspect occupancy first; retain their ownership/cleanup
receipts and use new output directories rather than replacing prior evidence.
