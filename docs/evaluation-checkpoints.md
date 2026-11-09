# Evaluation Checkpoints

Source organization checkpoint, November 8, 2026. These commits preserve experiments;
they do not select a model, change live prompt bindings, or authorize deployment.
Private transcripts, generated reports, SQLite databases and runtime receipts stay
under ignored `tests/behavioral/local/`.

## Current Workflow

The current controller uses registered drafts and recorded save offers. Start
with [the workflow](staged-project-comments.md) and
[the November 7 gate](save-offer-gate-20261107.md). That gate remains **do not ship**:
zero unapproved writes, but no completed genuine approvals and unresolved grounding.
Current conversation probes are `test_staged_comment_conversation_live.py` and
`test_staged_promotion_live.py`. Model inference is opt-in and uses synthetic storage.

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

The last group predates the registered-draft/save-offer flow. In particular,
`test_draft_approval_conversation_live.py` patches the now-removed
`support.run_project_comment_approval_classifier`, and older controller fixtures
seed assistant prose rather than registered offers. `run_model_regression.py`
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
