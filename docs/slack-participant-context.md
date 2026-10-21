# Slack participant context

The receiver keeps real `<@U...>` mentions instead of deleting them. Only the
addressed bot mention is removed from an app-mention request. For each request,
it includes up to 16 participants: sender, mentioned users, retained-thread
authors and cached profiles referenced by name. Profile lookups reuse the
existing `users.info` client and receiver cache. No workspace-wide directory
scan or new Slack scope is needed; lookup failures retain the raw ID.

The bounded prompt carries username, display name and real name together with
the stable ID. These names are quoted data, not staff status or permissions.
The participant map is retained in the support message's client metadata, so
later review can see the identity context used for that turn. Receiver caches
reset at restart; plain names absent from the thread/cache cannot be reliably
resolved. An actual Slack mention resolves the person without guessing.

For generated replies, an explicit `@name` becomes `<@U...>` only when it matches
a unique name in that request's map. Plain names are not turned into pings.
Unknown/ambiguous names and code/link content are left unchanged. Slack displays
the actual mention using its own current friendly name. No name mapping changes
the authenticated actor, allowlists, project scope, or confirmation policy.

Tests use synthetic users and fake clients. No test message is sent to Slack.
