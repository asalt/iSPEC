# Slack thread retrieval: October 29 evaluation

## Observed failure

An authorized request to make a project note about its Slack thread reached
production with `message_count=0` and `parent_present=false`. The parent had not
been captured in the local inbox. A read-only `conversations.replies` call using
existing access recovered the parent successfully. This was a retrieval gap,
not a requirement that every contextual message mention the bot.

The request was handled by the active comment-only production release. A later
attempted save was denied; no successful project-comment write was observed.

## Candidate implementation

- Verify the current inbox receipt and authorized requester before retrieval.
- Read one bounded page from the exact channel/thread, including unmentioned
  participants and bot replies as quoted evidence, not commands or approvals.
- Close the local DB session before network I/O; use a five-second timeout and
  no SDK retries. On failure retain local evidence with a sanitized error.
- Bound context to 24 messages / 8,000 characters, with source, coverage, missing
  parent, truncation and fetch outcome recorded in the normal support trace.
- Preserve the latest user message, sender authorization and all write gates.
- Keep detailed receipt fields in metadata and present a smaller quoted excerpt
  to the model, with separate author and mention-recipient identities.

See `slack-inbox.md` for the retrieval contract. This is source work, not proof
that the candidate is active in production.

## Validation and limits

144 focused backend unit/integration/behavioral tests passed, including ignored
senders, duplicate events, API failures, uncaptured parents, exact scope, and
old quoted approvals not authorizing writes. 40 wrapper deployment tests passed.
These tests use fake Slack and temporary databases.

Real Tulu inference through the staged support API used only synthetic projects,
temporary databases and a stubbed thread reader. No Slack messages were sent and
no production records were written. The initial six approval tests passed, but
the added missing-parent end-to-end case failed. Saved experimental outcomes:

| Presentation | Outcome |
| --- | --- |
| Full receipt JSON, separate user turns | Asked for project ID/content despite receiving both |
| Combined user turns or system-role guidance | Still asked for the project ID |
| Shorter quoted context | Recognized project/facts, then used an unsupported creation claim; existing guard blocked that answer |
| Explicit draft wording | Produced a draft, but invented a title or assigned work |
| Compact context plus explicit project lookup instruction | Recognized the ID, but asked the user to confirm it instead of calling `get_project` |

The early test assertion banning any mention of the recipient was too strict:
"Mira informed Sam" is valid attribution. It was replaced with a coarse identity
check and a project-lookup check. Neither substitutes for reviewing the actual
draft. The latest failure is not that old assertion: the model asked for the ID
again and made no lookup. Approval and false-save guards were not weakened.

The candidate has **not passed deployment preflight**. The active production
release remains `20261029-project-comments-ready`; API, Slack and Tulu were not
restarted. The grounded candidate release has no successful preflight receipt.
Do not activate the failed candidate or treat transport tests as model-quality
proof. The next useful test isolates controller/prompt effects on read-tool
selection and uses fresh source-grounded drafting variations before activation.

## Reproduction

From the backend checkout:

```sh
.venv/bin/python -m pytest -q --run-behavioral \
  tests/unit/assistant/test_slack_thread_fetch.py \
  tests/unit/assistant/test_slack_thread_context.py \
  tests/unit/assistant/test_slack_inbox.py \
  tests/unit/assistant/test_slack_participants.py \
  tests/unit/cli/test_slack_reply_failures.py \
  tests/unit/cli/test_slack_history.py \
  tests/integration/test_slack_inbox_bolt.py \
  tests/behavioral/test_slack_thread_context_sandbox.py \
  tests/unit/api/test_project_comment_live_approval.py \
  tests/unit/api/test_project_comment_approval_eval.py
```

The deploy checkout's `smoke_project_comments.py --thread-context-only` runs the
isolated end-to-end case but never writes an activation receipt. It checks model
occupancy first. On amms06, the current diagnostic reproduction is:

```sh
/home/amms06/services/mariana/venv/bin/python \
  /home/amms06/services/mariana/releases/20261029-slack-thread-context-grounded/deploy/smoke_project_comments.py \
  --release /home/amms06/services/mariana/releases/20261029-slack-thread-context-grounded \
  --thread-context-only
```

Private attempt traces and synthetic diagnostic logs are under ignored
`tests/behavioral/local/slack-thread-context-20261029/`. The checked-in report
does not contain the Slack transcript, credentials or participant identities.
