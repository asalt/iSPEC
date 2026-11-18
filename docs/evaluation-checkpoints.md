# Evaluation Checkpoints

## Current Status

**Full offline baseline, November 18: do not promote.** Tested backend `37fe2be`
with wrapper profile `ca57e20`, from Git archives/blobs rather than uncommitted
files. All 32 synthetic cases have transcripts. Five of eight requested saves
completed (previous full run: zero). N02 made an unapproved write after “The text
reads well.” The approval classifier saw the earlier wording-only request and
still returned approve_save at 0.95. P02/P07 called save correctly but the
classifier refused. R01 prepared printer chatter rather than saving. N18 showed
revised prose without registering it; N08 registered a revision but lost the
four-sample count. P06/N07 revisions and N09/N10 cancellations worked. N16
thanks did not trigger save. N17 and C04 stopped at the fixture tool boundary;
zero HTTP errors, 25/32 mechanical passes, 58/60 delivered replies.

[All cases, previous outcomes, transcript quotes and the rerun command](olmo-application-baseline-37fe2be.md).
The model still requests the save; the classifier, ticket and ACL checks are
unchanged. No identical-reprepare, no-tools-round or tool-trimming fix is
claimed in this baseline. Local only; production unchanged. Use this reference
to measure future changes before considering deployment.

## Earlier Evaluations

**November 18: cross-turn native history repaired on the OLMo application path.**
P01 and N07 went from 0/2 to 2/2 in the focused local GPU rerun, with the same prompts,
17 native tools, approval threshold and synthetic cases. Turn two now receives the
original assistant tool call and matching tool result before the original visible
reply. P01 saved the exact presented draft once (classifier approve_save 1.0,
SQLite comment #1; delivered reply reported #1). N07 prepared the requested
correction, superseded the old card and wrote no comment. Stored receipts are
bounded and scoped to the actor and access grants; old turns without receipts
are not reconstructed. [P01 transcript](../tests/behavioral/local/comments-native-history-20261118/olmo/pytest/test_application_conversation_0/transcript.md)
and [N07 transcript](../tests/behavioral/local/comments-native-history-20261118/olmo/pytest/test_application_conversation_1/transcript.md).
No promotion or live changes. Next: test negative approval and stale-card paths;
R01 grounding and tool-budget exhaustion are separate unresolved problems.

**Earlier native-prompt audit, November 18:** Stop directory tuning. Three deliberately
omitted categories, zero recovered after cleanup. D2 reopened experiments then
requested get_experiment(0); D3 searched endpoint docs then requested db_file_stats
(both fixtures stopped before a delivered reply); D4 refused despite an explicit
directory instruction and authorized files category. P01 prepared again, R01 registered
printer chatter on the save turn, N07 revised only in prose, N19 respected refusal.
Neither intended save completed. 82 focused deterministic tests passed. No promotion,
commits or production changes. Both owned endpoints closed.

**Read the conversations:** [D4 before](../tests/behavioral/local/directory-labeled-20261118/olmo/pytest/test_directory_D4_0/transcript.md)
and [D4 after](../tests/behavioral/local/directory-native-clean/olmo/pytest/test_directory_D4_0/transcript.md).
Other cases have the same sibling transcript.md. Sources in order: profile preamble
(checkpoint XML/native parser convention); assistant.base.system (persona, grounding,
status and scope); assistant.planner.system (formerly legacy conditional tool-use,
native-protocol reminder and FINAL wrapper; removed on ordinary native OLMo tool turns);
CONTEXT (server snapshot); assistant.project_comment.workflow (authoritative staged flow).
Directory hint is conditional on directory exposure. Compare-mode answer labels retained.
The directory fixture accidentally rebuilt raw create schemas after the application
had staged them; fixed in the fixture. D4 system-only tokenizer count 1677 -> 1168,
excluding schemas. Because both prompt and fixture changed, this run does not isolate
their individual effects.

November 18 labeled-directory retest: **0/3 recoveries, unchanged** in D2/D3/D4.
The captured native requests distinguish already-loaded groups from categories
available to open. D2 reopened experiments, passed an empty run ID, and delivered
literal tool syntax. D3 searched API docs for run 764, then attempted DB stats
(blocked by fixture; no final reply). D4 refused without opening files despite
its explicit attachment label. The open-category argument enum still includes
already-open groups. No evidence to promote this description-only change.
Evidence: `tests/behavioral/local/directory-labeled-20261118/`. Real readiness
completion, three cases, owned server exited and port closed. No live changes.

November 18: eight offline OLMo directory conversations completed on the local GPU;
see `tests/behavioral/local/directory-20261118/olmo/pytest/` for wire requests,
tool results and delivered replies. Frozen synthetic role/selection fixtures, not
live Jeff predictions. Native schema lookup worked with both categories visible.
None of the three deliberately omitted categories was recovered. D2 opened the
already-exposed experiments group, D3 attempted DB stats (fixture stopped it),
D4 incorrectly refused an authorized file lookup. Both unavailable lookups
produced refusals; D5 also advertised unavailable experiment capabilities.
ACL blocked D7 but the reply incorrectly inferred global nonexistence. Greeting
passed. Mechanical 5/8; do not promote directory recovery. No live wiring.
Next test: distinguish already-open vs available-to-open groups in the directory
itself, then retest the three misses. Preserve ambiguous not-found semantics.

Updated November 13, 2026. Keep this single section current after each run;
linked reports retain historical evidence.

- **Latest focused Jeff schema follow-up complete:** `misc` owns `search_api`;
  its offline description now explicitly names schema/API inspection, endpoints,
  operations and accepted parameters for any record type. N3 otherwise frozen.
  Ten staff cases, eight cached + twelve new predictions across two arms:
  required category inclusion 7/9 -> 9/9, top choice 7/10 -> 9/10. M02 now includes
  misc (0.422), though experiments still ranks first (0.545). All four experiment
  controls remain correct; fresh person-API schema case now selects misc first.
  Pending-save and wording-only controls held. 39 deterministic tests pass;
  source hashes verified, run exited zero, owned model/session closed.
  [Focused evidence](jeff-schema-routing-20261113.md).

- **Latest Jeff wording sweep complete:** 444 combinations, 443 new CPU
  predictions plus one exact historical reuse; exit zero, owned model/session
  closed. Frozen before scoring: 36 hand-authored evaluation cases, ten genuine
  none cases, plus a separately labeled historical save. Six arms, both roles.
  **Recommend N3 with draft-aware projects for the next offline integration test.**
  Historical save `P(none)` falls 0.815 -> 0.147 staff and 0.713 -> 0.189 general;
  projects now ranks first. Fresh prepare/save/revise/cancel also rank projects
  first. Available-category inclusion: 25/26 staff, 14/14 general; genuine-none
  empty selection: 8/10 in each role. N3 staff top-1 33/36; general supported-or-
  genuine-none 21/24. N1 wins general top-1 including unavailable requests
  (31/36 vs N3 27/36), so this is not a universal winner.
  [All arms, limitations, evidence and reproduction](jeff-router-wording-20261113.md).
- **Offline selection now:** filter choices before scoring using authorized
  nonempty categories plus none. Count all none mass without reweighting; rank
  to mass 0.9 / k3 / 24 tools, preserving the top real category when needed.
  Always include the authorized directory and, while pending, the four comment
  tools. General fixtures have projects/files/schedule/misc, not people or
  experiments. No controller wiring or approval changes. 38 focused deterministic
  tests passed, Ruff clean, frozen hashes unchanged. Saved-score limiter-only
  replay verified 64/65 staff and 63/63 general, not the proposed 65/65 staff:
  one old wording-only case hits the tool cap after selecting assistant.
- **Jeff remaining limits:** M02's top rank remains experiments, but corrected
  misc is now exposed. Two genuine none cases in the earlier sweep exposed
  projects; those were not rerun in this schema follow-up. General-user requests
  for unavailable capabilities can select unrelated available categories without
  exposing hidden tools. Stop Jeff wording tuning for that: evaluate OLMo's
  truthful availability response instead. The proposed eight-conversation
  [directory recovery test](olmo-directory-recovery-proposal-20261113.md) has not
  been implemented or run. No live wiring, commits, reweighting or approval changes.

- **Latest repair/replay:** centralized `disable_any_whitespace=True` for vLLM
  structured calls; the OLMo launcher also supplies the runtime-wide XGrammar
  setting its V1 compiler uses. Two exact saved whitespace failures now complete
  in 27/24 output tokens, both selecting projects. The supplied full-gate path
  was N19; actual N07 was replayed too. Prompts, 200-token router budget and
  approval threshold are unchanged. Owned OLMo stopped after replay cleanup.
  [Evidence and compatibility](router-repair-20261113.md).
- **Router-error fallback:** now keeps authorized projects, active comment tools
  and discovery. Exact saved N07 input falls from 51 tools / 7,236 tokens to
  17 tools / 3,545 tokens. The strict expected-failure regression flipped to
  XPASS, then became a regular passing test. All 118 focused backend and 127
  wrapper checks passed. Additional overlapping regression runs passed 141
  controller/workflow tests and 44 classifier/usage-log tests. Full conversation
  behavior has not been rerun.
- **Historical Jeff contextual replay:** 66 staff + 66 general-user inputs, one
  categorical question with preceding exchanges and prior draft state. Top-1
  subject-label matches: 32/66 staff, 41/66 general user. Category inclusion stays
  31/65 available staff cases and 39/63 available general-user cases across the
  tested cumulative targets/k/caps. Raw labels overcount wording-only/deferral
  cases as projects, but clear pending saves/cancels/revisions also choose none.
  **Do not use Jeff as an exclusive router yet.** No live wiring or commits.
  [Curves, complete example input and integration proposal](jeff-contextual-router-20261113.md).
- **Earlier offline routing design (superseded above):** the runner asked one categorical question
  with recent exchanges, prior draft state and the latest request. `none` is
  always included. Selection renormalizes eligible categories plus `none`, counts
  its mass, preserves the top eligible category even above the tool cap, and caps
  additions. A simulated permission-filtered directory replaces old discovery;
  nothing is wired into the controller. All 27 deterministic prototype tests and
  three existing tool-routing tests pass (30 total in the final focused run).
  Both role passes used one resident CPU model/one thread, no timing probes.
  Existing general-user scopes exclude people/experiment lookups; unavailable
  capabilities are reported separately. Four matching staff scores from an interrupted early attempt
  were reused; the old input and scores are retained separately. The first two-pass
  job received SIGTERM (source unknown) after 60 saved staff scores. A detached
  one-thread resume reused those 60 and computed only the remainder, then exited
  zero and closed its tmux session. No prompt or selection policy changed during
  recovery. Observed medians were 7.17 s staff / 3.44 s general user under load;
  these are not an isolated speed comparison.
- **Earlier Jeff timing comparison deferred:** stopped the owned CPU benchmark after Alex noted
  competing workloads. Six timing probes completed at one and four threads:
  medians 33.7 and 73.4 seconds for all 13 independent category questions.
  The eight-thread trial and 60-input accuracy/threshold comparison are incomplete.
  Host load after stopping was 35.13/30.65/27.91; these timings are contention-
  affected observations, not an isolated Jeff speed benchmark. No controller wiring
  or approval-classifier change. Frozen inputs and partial timing evidence remain
  in `tests/behavioral/local/router-repair-20261113/jeff-categories/`.
- **Current design:** one balanced shared application base prompt; native schemas
  replace only their duplicate prose catalog. The checked-in OLMo profile selects
  `assistant.tools.olmo3_preamble`, the packaged native checkpoint template with
  its tools-default guard, and `olmo3` parser. Launchers now resolve that exact
  hash-checked resource with `--chat-template`. There is no concise application
  mode or short-prompt plugin.
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
  printer chatter. N07's mechanical routing failure is repaired, but its complete
  revision conversation still needs a real rerun. C05's follow-up completed: it asks
  for the missing ID, invents none, but first makes an unnecessary tool-discovery
  call (the strict no-tool assertion fails). N09 truly
  cancels but inaccurately implies the draft had been in project history.
  **Do not promote.** These results do not establish false-approval safety.
- **Previous full-application gate:** 27 cases plus five controls, 28 native
  calls, 43 context-overflow HTTP errors, 0/8 saves, no classifier checks. Raw
  pytest 23/32 was not a behavioral score: failed inference left negative write
  counts at zero. Retained intact at `olmo-application-template-default/` under
  the same dated evidence root; the initial failed-start attempt is separate.
- **Packaging check:** the shared template, explicit launcher selection and wheel
  resources are implemented but uncommitted. Wording and template bytes are
  unchanged. An offline wheel check verified 265 source/resource files and loaded
  all 65 packaged prompts; the abandoned Jeff files were excluded. All 127 wrapper
  checks and 70 focused backend checks from the clean packaging snapshot passed.
  These are offline checks, not a new behavioral score.
  [Packaging and reproduction](olmo-template-packaging.md).
- **Next:** review the eight-conversation offline OLMo directory-recovery /
  unavailable-capability proposal. No OLMo launch, directory integration or live
  routing has been performed. Defer CPU optimization. Finish long-history coverage,
  then rerun the
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
  The subsequent packaging pass started no services and made no new commits.
  The November 13 isolated router replay started and stopped only its owned local
  OLMo server; no other services were restarted. These repairs are uncommitted.
  Alex reports amms06's NVIDIA driver updated, with reboot still pending; leave
  its GPU/runtime alone until that reboot and a separately authorized check.

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
