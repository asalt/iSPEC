# Project Comment Approvals And Undo

Local implementation for review. No production activation or model promotion.

## Flow

1. The model calls `prepare_project_comment(project_id, comment)`.
2. The server stores that exact text and generates a draft ID/hash. It returns
   a separately labelled confirmation card alongside the unchanged model reply.
3. A person clicks **Save to project history** or **Cancel**, or the model requests
   `create_project_comment(draft_id, confirm)` after a later text reply. The text
   path runs the existing classifier and ticket checks at unchanged **0.75**.
   Web and Slack buttons use
   `POST /api/support/project-comment-cards/{card_id}/actions`.
4. The executor validates identity, conversation/message binding, current project
   access, hash, expiry, committed presentation and pending state. A save consumes
   a request-local single-use ticket and inserts the exact stored text atomically
   with the draft outcome. Both paths share this executor and writer lock:
   text then button, button then text, and simultaneous saves insert once.
   A retry returns the existing saved comment ID.
5. A new preparation supersedes the prior draft. Its old buttons are rejected.
   The browser and Slack adapter render **Saved as comment #N**, cancelled,
   superseded, undone or expired state from the server result. Saved cards offer
   **Undo** until the original draft expires, subject to the local-only limits below.

`show_project_comment_card` remains a read-only redisplay tool; it neither runs
the approval classifier nor saves. `create_project_comment` is the explicit
model-requested text-save path. The non-chat low-level comment API is unchanged.
Alex accepts the possibility of classifier false approvals for local SQLite
comments. False saves via text are a primary behavioral metric, not something
this dual-path design makes impossible.

Wording-only help should stay in model prose. If the model mistakenly prepares
a history draft, a card appears: this is an unwanted-card behavioral failure,
not a button authorization. A subsequent model save request and false classifier
approval can still save it through text. Model cancellation and revision
follow-through still need behavioral testing. The structural change does not fix
factual grounding.

## Local Undo

Undo uses the same authenticated action endpoint, requester/allowlist and project
ACL as Save. The original draft expiry, normally 24 hours after preparation, sets
the deadline; saving does not extend it. Undo checks the exact saved comment ID,
project, content hash, attribution and creation/modification timestamps. It
refuses a changed or missing row rather than deleting later work.

The transaction deletes only that local comment, clears the draft's comment
foreign key, records `undone`, and retains the original comment ID plus both save
and Undo receipts. Repeated Undo is an idempotent read of the existing outcome;
the closed draft cannot be saved again. Undo is never a model tool.

`CommentUndoTarget` and `CommentUndoResult` in `project_comment_undo.py` provide
the typed boundary for a future backend-specific reversal. Only local SQLite is
implemented. If legacy push was enabled when saving, or the row has legacy
provenance, local-only Undo refuses the action. Manual/out-of-band FileMaker
copies are not detectable from these receipts and need separate reconciliation.
No FileMaker adapter, credentials or live writeback settings were changed.

## Identity And Storage

- Web actions require the authenticated requester, including existing browser
  origin checks. Current account state and project ACL are reloaded at the click.
- Slack actions come through the existing authenticated Socket Mode receiver and
  private backend API key. The backend checks the actual clicker is the requester
  or explicitly listed in `ISPEC_SLACK_ALLOWED_USER_IDS`, with matching workspace,
  channel and posted message. An empty sender list admits no additional clickers.
- Slack project ACL is evaluated using the existing service principal. There is
  no individual Slack-to-AuthUser project-grant mapping in this slice.
- New `project_comment_card` rows bind the draft, transport and action receipt.
  Existing database initialization creates the table. No live DB was migrated.
- Conversation and core databases are separate. A committed matching assistant
  receipt is required before acting, so a rolled-back reply leaves an inert card.
- Comment attribution remains the iSPEC Assistant person; the actual approving
  web user or Slack ID is recorded separately in `com_AddedBy` and the receipt.
- Legacy writeback uses the existing enqueue helper after the save commits. An
  enqueue failure is reported separately; it never changes a successful save
  into a retry that inserts another comment.

## Slack Activation Step For Alex

In [Slack app settings](https://api.slack.com/apps), select the existing iSPEC app,
then **Features > Interactivity & Shortcuts > Interactivity: On** and save changes.
Keep Socket Mode enabled. Socket Mode does not require an Interactivity Request
URL. If Interactivity is already on, no toggle is needed.
[Slack's interactive-components guide](https://docs.slack.dev/tools/java-slack-sdk/guides/interactive-components/).

No app settings, scopes, credentials, receivers or production services were
changed. Deployment still needs separate review. Actual Slack delivery has only
been stub-tested, with no test messages sent.

## Limits And Next Validation

- Ordinary single-response chat is wired. Compare/choose responses do not yet
  publish newly prepared cards; those drafts remain non-actionable until a normal
  reply presents them. Cover that alternate delivery path before enabling it.
- A failed Slack delivery-binding request leaves the posted buttons inert.
  Delivery is logged, without an automatic resend loop or recovery worker.
- The browser restores the latest 20 delivered cards for its conversation.
  Slack refreshes returned cards in place. Failed UI updates can leave stale
  buttons visible, but the executor checks current state on every click.
- Card state is separate from assistant prose. No model message is replaced to
  hide an inaccurate claim. Evaluate truthful cancellation/revision language
  independently of the authoritative card state.
- Every model turn receives the delivered draft's current state, including its
  saved comment ID, even if write tools are unavailable. Repeating a model save
  after a button save returns the existing outcome without another classifier
  call or insertion.
- Next: independent tool-result-reminder and classifier-question experiments,
  fresh held-out wording, then the existing 27 conversations. Compare with saved
  baseline `3be57d2`, distinguishing structural changes from prompt effects.

## Local Checks

The focused Python suite passed **165 tests** (two existing FastAPI deprecation
warnings). It covers HTTP identity/origin checks, Slack stubs, exact
text, stale/expired/superseded cards, both save orderings and concurrent saves,
cancel-then-save, ACL/account revocation, Undo scope/expiry/later edits, per-turn
authoritative state and consumed-ticket retry handling. A confidence-1 classifier
stub deliberately demonstrates the accepted false-text-approval risk; it is not
a measurement of a real model. See the current checkpoint for the latest totals.

Frontend build and six mocked Cypress checks pass: desktop/mobile exact text,
Save, Cancel, Undo, a text-saved card with Undo, and an action failure that stays
pending. Saved-card screenshots were inspected at desktop and mobile sizes.
The tests filter the specific Electron
`ResizeObserver loop limit exceeded` warning seen when opening the existing chat
panel; other application exceptions remain fatal. No browser requests reached a
live API. No live model or Slack behavior was tested in this structural batch.
Fixtures are hand-authored synthetic unit/browser data, not collected examples
or training data. Provenance rules are in `tests/behavioral/README.md`.

Reproduce the Python gate from the backend checkout:

```sh
../.venv/bin/python -m pytest --run-testclient -q \
  tests/unit/assistant/test_project_comment_cards.py \
  tests/unit/assistant/test_project_comment_dual_approval.py \
  tests/unit/assistant/test_project_comment_workflow.py \
  tests/unit/api/test_project_comment_live_approval.py \
  tests/unit/api/test_project_comment_card_routes.py \
  tests/unit/cli/test_slack_comment_cards.py \
  tests/unit/cli/test_slack_reply_failures.py \
  tests/unit/api/test_support_chat_openai_tools.py \
  tests/unit/api/test_support_chat_slack_meta.py \
  tests/unit/api/test_project_comment_approval_eval.py \
  tests/unit/api/test_support_chat_openai_tool_choice.py \
  tests/unit/assistant/test_tools_tool_catalog.py
```

Browser spec: `mspc-data-entry-ui/app/cypress/e2e/project-comment-cards.cy.js`.
Run against an isolated frontend build/preview with all API requests stubbed by
the spec. Test screenshots remain outside Git under `/tmp/ispec-dual-card-browser-images/`.
