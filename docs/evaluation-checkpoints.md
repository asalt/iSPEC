# Evaluation Checkpoints

## Current Status

Updated November 12, 2026. Keep this single section current after each run;
linked reports retain historical evidence.

- **Current design:** one balanced shared application base prompt; native schemas
  replace only their duplicate prose catalog. The checked-in OLMo profile selects
  `assistant.tools.olmo3_preamble`, its native checkpoint template and `olmo3`
  parser. There is no concise application mode or short-prompt plugin.
- **Comment workflow:** one preparation call registers a draft. Exact server cards
  stay separate from model prose. Both button saves and model-requested text
  saves are enabled in the local implementation. Text uses the classifier and
  existing ticket path at **0.75**; Alex accepts its false-approval risk for local
  comments. Both paths share the atomic claim and saved-card Undo. Authoritative
  draft state returns to the model. No model-authored messages are replaced.
  [Implementation and remaining delivery limits](project-comment-cards.md).
- **Repairs implemented:** one catalog per OLMo request; one native tool-result
  message; server-tokenized input/output budgeting; no lexical-policy lookup
  forced after OLMo's strict native no-tool response. Other parsers retain their
  previous lookup behavior. The omitted-provider planner fallback is repaired.
  Full base instructions,
  stored assistant prose, schemas and approval thresholds are preserved. The
  approved checkpoint `tools | default(none)` startup correction remains; no
  vLLM edit or monkeypatch. [Causes, repairs and case tables](olmo-integration-repair-20261112.md).
- **Last completed batch:** nine real OLMo cases, frozen repaired working tree:
  12 native calls, one approved save out of three save scenarios, no unapproved
  writes (one classifier check, true approval). Cancellation and greeting worked;
  project lookup worked through a valid alias. No server context errors. P01's
  input fell from 7,519 to 3,559 tokens, post-tool from 9,643 to 3,784. All 41
  preflight token counts matched actual inference usage. Eight preparations
  delivered cards, including an unwanted wording-only card in N02.
  Evidence: `tests/behavioral/local/staged-20261112/olmo-integration-repair-v2/`.
- **Current blockers:** P01/R01 prepare again instead of saving; R01 retains
  printer chatter. N07's router returned incomplete JSON, then the existing
  fallback exposed 51 tools (7,236 tokens); the new budget check stopped that
  request. This fallback still needs repair. C05's follow-up completed: it asks
  for the missing ID, invents none, but first makes an unnecessary tool-discovery
  call (the strict no-tool assertion fails). N09 truly
  cancels but inaccurately implies the draft had been in project history.
  **Do not promote.** These results do not establish false-approval safety.
- **Previous full-application gate:** 27 cases plus five controls, 28 native
  calls, 43 context-overflow HTTP errors, 0/8 saves, no classifier checks. Raw
  pytest 23/32 was not a behavioral score: failed inference left negative write
  counts at zero. Retained intact at `olmo-application-template-default/` under
  the same dated evidence root; the initial failed-start attempt is separate.
- **Next:** consolidate rendered-prompt tests and package the exact guarded
  template for explicit launcher use; cover both token budgets, long history
  and failed routing. Fix the router-error catalog expansion, then rerun the
  missed-save, revision, grounding and cancellation cases with frozen source.
  A guarded refusal is not a successful conversation. No additional prompt
  tuning or full gate has been performed. See the
  [deployment checklist](deployment-readiness-20261112.md).
- **Previous behavioral baseline:** `3be57d2` with the short introduction and
  custom template completed 27 cases: 14 mechanical passes / 13 failures,
  5/8 intended-save conversations wrote, and N02 made one unapproved write.
  N09/N10 falsely claimed cancellation; revisions and grounding also failed.
  [Baseline case evidence](olmo-frozen-gate-20261112.md). Those remain historical,
  not scores for the current card/prompt integration.
- **Deterministic checks:** 257 focused backend checks passed from a clean archive
  of committed source `29af63a`, including opt-in HTTP tests and rendered OLMo
  prompts with the installed tokenizer. Another 16 template/runtime-snapshot and
  prompt-observability checks passed. The separate card commit's staged snapshot
  passed 165 checks. Wrapper: all 126 tests passed. Frontend: build, six mocked
  Cypress checks and two Node test files passed; desktop/mobile captures inspected.
  These overlap with earlier runs; do not sum historical totals. This is not the
  full backend suite or fresh model evidence. Alternate compare/choose card
  delivery remains unfinished.
- **Commit checkpoint:** backend `9fb7c1f` (provider default), `026a188` (dual-path
  cards/Undo), `9071b99` (OLMo integration/budget checks), `29af63a` (isolated gate
  tooling); wrapper `9013687`; frontend `4d11116`. Documentation is checkpointed
  separately. Five abandoned Jeff tool-use experiment files remain uncommitted
  and unused; preserve them outside any release source rather than deleting or
  staging them to make this worktree clean.
- **Scope and data:** OLMo only; Tulu and Jeff tool-use arms remain excluded.
  Real collected examples stay distinct from synthetic/derived evaluation cases.
  Training-related synthetic generation would use gpt-oss-20b; no DPO/training here.
  Local commits were authorized and made. No push, deployment, production setting
  change, existing-service restart, external send or real-database write occurred.
  Production state is retained from Alex's handoff, not newly inspected. The
  prior model batch cleaned up its owned GPU services; this commit pass started
  no model. Its isolated frontend preview was stopped after the browser tests.

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
the experiment machinery, not model behavior. November 12 included a failed
startup, a completed full-application run and focused integration repair retests.
Each has a separate frozen working-tree snapshot and service receipt. None made
downloads, production changes, Slack sends or driver changes.
Runtime launchers require already installed software/checkpoints and explicitly
requested execution. Inspect occupancy first; retain their ownership/cleanup
receipts and use new output directories rather than replacing prior evidence.
