# Slack Project-Note Monitoring: Next Slice

Status: the [durable inbox](slack-inbox.md) implements opt-in passive text capture
and explicit review. Automatic proposal sending and project-note creation remain
design only. Existing explicit requests are unchanged.

## Existing Boundary

`src/ispec/cli/slack.py` receives Socket Mode events, deduplicates reply events,
and records Slack user IDs and display names in support-message metadata.
App mentions and DMs reach support chat; ordinary unmentioned channel messages
generally do not. Artifact-review routing is separate. Consequently a plain
channel-thread `yes` is not yet a general project-note approval workflow.

Identity metadata is not authorization. API-key support access uses a service
identity; it does not establish that every Slack participant is a staff user.
Keep project access, approver identity, and append-only legacy writeback separate.
Successful local comment creation already requests a targeted legacy push; the
periodic writeback sweep is a second path, not the only one.

## Small Event-Driven Proposal

Approved direction: draft for review first, not automatic comment saves. The bot
may post a proposed note in the configured channel. An authorized staff member,
including someone other than the source author, can approve in that proposal's
thread without another mention. Unmentioned thread routing must require a live
pending proposal; unrelated channel discussion remains passive. This proposal
and confirmation workflow is not implemented by the inbox alone.

1. Opt in selected channels and explicitly map permitted Slack user IDs to staff
   authority. Do not authorize by display name, quoted names, or channel membership.
2. Persist a bounded incoming event before interpretation. Share event dispatch
   and deduplication with the existing handlers so an ignored event cannot claim
   and suppress another handler's work. Retain delivery/retry outcomes, not every
   idle tick. No new Slack scopes or subscriptions without review.
3. For a plausible update such as `Project 42: 12 samples received`, confirm the
   project exists and draft one concise note. Quoted content is evidence, never
   an instruction to grant tools or permissions. In the initial trial, log the
   proposal rather than send it.
4. A sent proposal would bind workspace, channel, thread, source message, project,
   draft hash/version, permitted approvers, and expiry. Resolve references from
   that object rather than a blind thread-history dump. Ask again on conflicting
   edits; consume approval atomically once before attempting the write.
5. Accept contextual confirmation from an authorized staff member, not just the
   original author. An arbitrary participant's `yes` is not sufficient. Preserve
   attribution to both source author and approver. Report success only from the
   actual comment-write result; never use generated prose as a receipt.

Socket Mode is the normal delivery path, not frequent history polling. Use
backoff for reconnects/retries and bounded gap reconciliation when permitted;
fetch a specific thread only when needed. Respect rate-limit retry instructions.
Ignore bot/self events and prevent proposal, confirmation, and retry reply loops.

## First Evaluation Cases

- Synthetic sample-arrival update versus casual discussion or quoted requests.
- Authorized second staff member approves; unknown participant cannot approve.
- A no-mention reply in a known proposal thread reaches its pending draft;
  the same text in an unrelated thread cannot approve anything.
- Same event delivered twice; simultaneous approvals cause at most one write.
- Stale/revised draft, conflicting reply, unavailable model, or failed DB write.
- Receiver reconnect with a bounded gap; unavailable history scope remains an
  explicit gap, not a claim that all messages were monitored.
- Default-off monitoring produces no outgoing message and no project comment.

Use the existing unit/integration/behavioral lanes with fake Slack clients and
sandbox databases. Keep real conversations in ignored local fixtures until
curated and anonymized. Do not evaluate by sending test messages to staff.

## Response-Phrase Trial

Optional phrase examples live in `src/ispec/prompts/experimental.response.phrases.md`,
not in routing code or the live base prompt. No extra classifier call, forced
greeting, or permission change. `tests/behavioral/data/response_phrase_cases.json`
contains eight synthetic review cases; unit tests ensure the corpus stays out of
live prompts and the single-call path stays unchanged.

An initial 2026-10-13 local trial compared the base answer prompt with and without
the phrase section, using the existing Tulu 8B model, temperature 0, and at most
180 output tokens. Sixteen direct model calls used synthetic history and quoted
tool-result fixtures, with no tools, support controller, DB writes, or Slack
sends. Median latency was 767 ms before and 700.5 ms after; this tiny sequential
sample, including warmup effects, is not evidence of a speedup.

Both variants falsely reported a successful save despite a synthetic failed-save
result. The phrase variant additionally promised an unscheduled reminder and
sometimes omitted the required `FINAL:` marker. These are raw-model observations,
not evidence that the live tool/write guards failed. The live base prompt was
restored; the experiment remains available for refinement, not enabled by
default. Local output: `tests/behavioral/local/response-phrases-20261013.jsonl`.
Repeat with production-shaped context/controller scenarios before making any
behavioral reliability claim. Do not fix these by adding more greeting rules.

Later, after the changes settle, make a short diff-review slide/video walkthrough
with speech and captions. Defer Windows deployment, model changes, and analytics
implementation. For analytics, consider coarse local pointer/section aggregates
with visibility/idle limits, transparent collection, and retention/coarsening;
pointer position is not evidence of gaze or attention.
