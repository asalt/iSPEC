# Named accounts for draft requests

Dot authenticates as its own viewer with `is_comment_agent: true`. An admin
creates that account with the existing `POST /api/auth/users`; leave project
grants empty. It can read internal comments across projects and chat in its
own threads. It can prepare and revise a draft, not save, undo, or authorize
a project comment. A message saying “I am Alex” does not change the principal.
Before rolling back to software without the agent restriction, deactivate
these accounts and revoke their sessions.

```sh
export ISPEC_API_URL=https://marianaproteomics.org
ispec auth login --server "$ISPEC_API_URL" --account dot
ispec support chat --account dot --staff-review "I'm Dot. Draft a note for project 57: eight samples arrived."
ispec support chat --account dot --session-id '<returned session_id>' 'Who requested the draft?'
```

The same API accepts `POST /api/support/chat` with `message`, an optional
`sessionId` to continue, and optional `reviewDestination: "staff_slack"` to
share **a draft prepared in this turn**. Without that destination the draft
stays in the requesting chat. No arbitrary channel or Slack metadata is
accepted from the agent. Account name and action receipts are server records;
the card displays the note, not a requester byline.

Other operations need no CLI equivalent. For example, use the **same HTTPS
origin** and the private Netscape-format jar path printed by login:
`curl -fsS --cookie "$JAR" "$ISPEC_API_URL/api/project_comment/by_project/57?limit=20"`.
Do not share or print cookie contents. The CLI checks the account before chat
and does not send a service key alongside the cookie.

## Staff Slack review (off by default)

Enable `ISPEC_SLACK_AGENT_REVIEW_ENABLED=1` on **both** the API and existing
Slack receiver, set `ISPEC_ASSISTANT_STAFF_SLACK_CHANNEL` to the staff channel
ID on the API, and set its `ISPEC_SLACK_ALLOWED_USER_IDS` to human reviewer IDs.
Normal deployment write controls and the receiver's existing card permission
are still required. No second bot or supervisor is involved.

The receiver checks every 30 seconds for a committed presentation and claims
one pending card per check. A claim is not delivery: only the Slack message
receipt enables the buttons in that channel. If Slack may have accepted the
post but no receipt was recorded, it is not automatically reposted. Inspect
the receiver warning and the card before requesting another delivery.
A permitted human click uses the normal idempotent save path. Mention iSPEC
in the posted card's thread to ask who requested it. The server supplies the
requesting account there, not the contents of the agent's private chat.
