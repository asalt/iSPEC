# Model-directed project comments

Development implementation; not deployed or model-promoted. The support-chat
path now uses explicit tools rather than inferring a pending save from prose.

## Flow

1. The responding model calls `prepare_project_comment(project_id, comment)`.
2. iSPEC validates project access and stores the exact draft, returning its ID.
   No project comment is inserted. Server presentation shows the returned exact
   text for review, rather than relying on the model to repeat it correctly.
   Preparation alone creates no save offer. The model can call
   `offer_project_comment_save(draft_id)` to request an explicit offer; the server
   renders the stored text and save question and binds a receipt to that assistant
   message. A save attempt without an existing offer presents one instead of writing.
3. On a later reply the model may revise, cancel, answer normally, or call
   `create_project_comment(draft_id, confirm=true)`.
4. A model-requested save answering the preceding recorded offer runs the bounded
   approval check. Policy applies the existing confidence/timeout settings. Its
   result cannot automatically initiate a save; repeated calls share one cached check.
5. A permitted save inserts the stored text and records the resulting comment
   ID in one core-DB transaction. Tool results tell the model what happened.

The approval classifier interprets only gated pending state. It receives the initiating
request, actual preceding assistant presentation, exact pending draft, and
current user reply. No lexical evidence vector or regex-derived pending state
is used by this path. A non-approval, malformed response, or timeout cannot fall
back to lexical approval. The result is cached for repeated calls in that turn.

## Boundaries

- Tools remain visible regardless of approval. Deployment read-only settings,
  tool allowlists and role/access restrictions still apply.
- Save accepts a draft ID, not replacement text, project ID, or person ID. The
  author remains the existing iSPEC Assistant person, with the authenticated
  account in `com_AddedBy`. Per-person attribution is not part of this slice.
- Drafts are bound to the authenticated principal, support session/thread, and
  Slack actor when supplied by the receiver. Another participant cannot confirm
  someone else's draft in this first slice; cross-staff approval is separate.
- An offer must match the draft ID/hash, project, thread/user and preceding
  assistant message, including its presentation hash. Model prose is not an offer.
  A same-turn call cannot self-approve. This proves server-side conversation
  presentation, not delivery/read receipt.
- Preparing a revision supersedes the old pending draft and invalidates its offers.
  Cancellation invalidates offers without deleting existing comments. Pending
  drafts and their offers expire after 24 hours.
- The core `project_comment_draft` and `project_comment_save_offer` tables are
  created by normal `initialize_db`.
  Keeping it alongside comments allows one atomic claim/insert/completion.
  The workflow closes its draft-store transaction before inference. Duplicate requests return
  the saved ID or a changed-state failure, never a second insert.
- A failed write is not automatically retried within that turn. Legacy push
  enqueue happens after commit; enqueue failure does not misreport a saved note
  as an unsuccessful write.

`ISPEC_ASSISTANT_PROJECT_COMMENT_APPROVAL_EVAL_MODE=own` is required for staged
chat saves. The existing `off`, `auto` (off/shadow), and `shadow` settings leave
drafts unsaved; shadow evidence cannot authorize execution. Threshold remains
0.75, timeout 6000 ms, ticket lifetime 120 seconds by default. No environment
values or running services were changed for this implementation.

Existing draft-like prose is not migrated into pending actions. Ask the model
to prepare the note again. The native tool schemas and tool catalog now explain
this flow. Existing non-chat low-level callers retain their legacy interface;
this is not a wholesale removal of regex or other routing throughout iSPEC.
The generic read-tool router and final-response safeguards remain separate.

Prepared text and save/cancel outcomes are rendered from tool state after model
response review. The raw model reply is preserved under `presentation.model_content`
and in inference traces, not silently credited as correct. Unknown/stale IDs
still fail without substitution; an ambiguous pending set is not selected.
This improves transaction continuity, not semantic grounding: the model still
has to prepare a relevant note and register revisions or cancellation.
The classifier result is supplied as a short system message after a save check
in `own` mode only. Shadow evidence is not presented as actionable authority.

## Current context and postmortems

The pending-draft system message is rebuilt before each model round and the
tool-limit answer fallback. A replacement, cancellation, or successful save
therefore changes the current snapshot immediately. Earlier tool results remain
historical evidence. Native `tool` results also retain the established system
`TOOL_RESULT` compatibility copy: the cached Tulu template does not render
`role=tool` at all. Removing that apparent duplication hid successful results
from Tulu in an isolated diagnostic. A parser name alone is not proof of template
capabilities. Separate result evidence for the response-review path is retained.

The default workflow is `src/ispec/prompts/assistant.project_comment.workflow.md`.
The shorter `experimental.project_comment.workflow.concise.md` is retained for
replay, not selected by default: the comparison did not establish improvement.
The earlier prompt-only trial restored the original descriptions. The latest
retained prompt adds a short project-facts-only instruction. Server presentation
is separate from model prose; access, thresholds and parser/ticket checks remain.

The November 6 `state-routed-trial` tested automatic dispatch after a classifier
ticket. It was rejected: Tulu misclassified wording-only agreement at 0.90 and
the dispatcher saved it in the synthetic DB. That dispatch code was removed,
not hidden behind a production flag. The associated expanded workflow prompt
also regressed OLMo's preparation calls and was reverted. Saved trial evidence
must not be described as the current implementation.

For a saved behavioral run, generate a standalone, local postmortem viewer:

```sh
../.venv/bin/python scripts/review_staged_conversations.py \
  tests/behavioral/local/staged-20261106/catalog-fixed \
  tests/behavioral/local/staged-20261106/context-refresh \
  tests/behavioral/local/staged-20261106/context-and-wording \
  tests/behavioral/local/staged-20261106/compatibility-preserved \
  --notes tests/behavioral/data/staged_comment_review_notes.json \
  --template-map tests/behavioral/local/staged-20261106/prompt-templates.local.json \
  --output tests/behavioral/local/staged-20261106/postmortems.html --overwrite
```

This reads `conversation.json`, `wire.json`, matching `results.xml`, and optional
review annotations; it performs no inference, tool execution, or network calls.
Existing HTML requires `--overwrite`; source evidence is never modified.
The index reports recorded pytest outcomes, separately from comment-count matches,
automatic review signals and manual behavioral findings. Missing/ambiguous JUnit
results are unknown, not inferred passes. Search and batch/model/result filters
work offline; each case expands to its evidence. Reading notes are at the bottom.
The generator's HTML/CSS/navigation assets are in `scripts/staged_review_assets/`.
Each inference now shows IN (reconstructed template text with model role markers),
ordered API messages, and OUT (returned text versus parsed tool calls). Per-turn
links find requests containing that latest user message; the full timeline also
includes routing/classifier requests. Stored drafts are database state, not chat
history. A printed function-call string is not a parsed or executed call.

Template reconstruction is optional and offline. The local template map explicitly
identifies each served model's tokenizer config and template override, their
SHA256 fingerprints, and provenance. Unknown models, changed fingerprints, or
unsupported request overrides are not silently guessed. Omit `--template-map`
for an API-only view. Reconstructed text is not captured token IDs; pre-parser
output is unavailable where the runtime returned parsed tool calls.

The trial now uses named `RecordedTurn`, `RecordedDraft`, and `RecordedComment`
snapshots, serializing only when saving evidence. Count failures report the
specific turn, conversation, registered draft and presentation state, executed
tools and approval-check count instead of only differing integer lists.
Per-turn expectations now check registered state, exact presentation, revision
IDs/supersession and note facts. A no-save turn also fails on an unexpected save
attempt, even if the executor blocks it. Cancellation tests require an actual
cancelled draft; an empty table and an unsupported cancellation claim cannot pass.
Guard interventions and server-rendered responses are retained for separate
semantic review, rather than treated as evidence that raw model prose was correct.
The full request/response timeline includes raw output before final-response
guards, alongside delivered replies, actual calls/results, draft transitions,
classifier decisions and write outcomes. Pytest's `*current` symlinks are
deduplicated. These runs contain synthetic data; treat arbitrary real-session
exports as private, not automatically safe to share.

Review five categories separately: save decision, draft handling, execution,
response truth, and note grounding. A blocked premature save is still a bad
save decision. A successful database insert does not prove the drafted content
was good. A response guard correcting a false claim does not make the original
model response correct. Mechanical count checks are not semantic judgments.

## Evaluation

Run focused local tests with the backend environment:

```sh
../.venv/bin/pytest -q tests/unit/assistant/test_project_comment_workflow.py \
  tests/unit/api/test_project_comment_live_approval.py \
  tests/unit/api/test_project_comment_approval_eval.py \
  tests/integration/test_staged_comment_concurrency.py
```

These use synthetic data and stubbed inference, including competing SQLite
writers. Old API tests that required forced saves/tool hiding were replaced with
explicit staging, tool-visibility and no-prose-trigger regressions. Non-comment
routing tests remain. Existing model-evaluation transcripts are historical
evidence; they do not establish performance with this new tool sequence.

The first real Tulu/OLMo conversation comparison is recorded in
[`staged-model-evaluation-20261106.md`](staged-model-evaluation-20261106.md).
It exposed a contradictory legacy planner/catalog prompt, now fixed, plus
remaining model sequencing, presentation and grounding failures. The staged
workflow is not deployment-ready based on that small batch. Score model tool
choice, checker decision, actual write, truthful final response and latency
separately; compare checker versus model decisions offline without weakening
the live approval policy.

The follow-up context/template diagnosis and the rejected concise-prompt trial
are recorded in [`staged-context-cleanup-20261106.md`](staged-context-cleanup-20261106.md).

The latest save-offer gate is recorded in
[`save-offer-gate-20261107.md`](save-offer-gate-20261107.md). It blocked all unsafe
writes but also completed none of the eleven genuine approvals. The frozen prompt
still teaches prose offers rather than the new offer tool; that contract mismatch
and remaining grounding failures require review before deployment. No promotion
is implied by committing this development checkpoint.
