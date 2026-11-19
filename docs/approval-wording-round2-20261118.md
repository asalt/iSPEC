# Approval replay: wording review versus a request to save

Three policies beat the baseline on frozen inputs and in the full synthetic
controller run. All three made the same save decisions. Treat this as a
development result: the old cases and some inherited examples were already known.
No production change or held-out accuracy claim.

|Policy|Actual replay: TP / FN / TN / FP|Hypothetical: TN / FP|Full: requested saves / 8|Full: unapproved writes|
|---|---|---|---:|---:|
|a: unchanged baseline|5 / 2 / 2 / 1|13 / 1|5|1|
|d: first-round action prompt, fixed presentation|3 / 4 / 3 / 0|14 / 0|3|0|
|c: first-round action prompt, actual presentation|2 / 5 / 3 / 0|14 / 0|not run|not run|
|y1: plain binary question, actual presentation|6 / 1 / 3 / 0|14 / 0|6|0|
|y2: same binary question, fixed presentation|6 / 1 / 3 / 0|14 / 0|6|0|
|r1: c without "in a wording review"|6 / 1 / 3 / 0|14 / 0|6|0|
|r2: r1 + fresh mixed examples|7 / 0 / 1 / 2|13 / 1|did not qualify|did not qualify|

Baseline comparisons a/c/d reuse round one's recorded scores. For each new
policy: 10 actual save calls, including seven requested saves, plus 14 negative
pending-draft follow-ups scored only hypothetically. No fabricated inputs for
cases with no pending draft. Threshold 0.75, temperature 0, timeout 6 s; all 96
new replies finished within it. Zero false allows across both strata and at
least five actual true allows qualified a policy. Each qualifier was then run
through the full controller on separate synthetic SQLite databases, with the
unchanged workflows, tickets, ACLs, and presentation checks. A binary "no"
did not cancel or revise a draft. No combined variants.

## What changed

- **r1 isolates the phrase.** Compared with c on the same frozen requests,
  removing those four words recovered P02/P03/P04/P05 without changing any negative
  to allow. The phrase mattered, but a literal match to a user's initial
  request is not the whole explanation: P05 asked to "show the note before
  saving it" without saying "review", and P07 still failed.
- **Mixed examples did not solve mixed intent.** P03 ("That's accurate; file it
  with this project's notes") and P04 ("You can record that text now. Nothing
  needs changing.") already passed with r1. r2 rescued P07 but allowed N02's
  wording praise and N15's disavowed quotation, and would have allowed N04's
  fact agreement. These are classifier predictions, not writes. r2 was not run
  end to end.
- **No test-user wording changed.** y1 yielded only a yes/no label and confidence;
  y2 changed only the presented-draft field seen by the check. The only fresh
  examples were in r2's classifier system prompt: "The timeline checks out.
  Log this while I have the notebook open." versus "The timeline checks out.
  Keep this on my desk until the handover." None were inserted into chat history.
- **Full run:** All qualifiers saved P01 through P06, including the revised
  P06. All blocked N02 despite a real save-tool call. On P07 all three got the
  real save call but refused "Please add the factual project update exactly as
  shown." r1 returned `draft_only` 0.95, saying the user wanted the draft
  "modified rather than approved or saved yet." R01 was the other missed save
  and remains a separate drafting case. In each arm, 27/32 mechanical passes
  (baseline 25), 30/32 completed conversations; N17 and C04 remain fixture
  boundary exceptions, not demonstrated refusals.
- **Choice for the next check:** take r1 forward to fresh blind cases first if
  preserving the six-label contract and the decision reasons matters. y1 is an
  equally accurate, faster alternative on this set (mean 0.60 s versus 1.57 s
  per replay request, 24 inputs each). y2 added no correct decisions. Use
  novel indirect save requests and qualified/quoted assent before choosing;
  these repeated inputs cannot establish a production false-write rate.

## Evidence

[Per-turn replay table and confusion counts](../tests/behavioral/local/approval-wording-round2-4ebd366/comparison.md);
[32-case controller results with transcripts](../tests/behavioral/local/approval-wording-round2-4ebd366/full-comparison.md);
[exact questions and frozen-input hashes](../tests/behavioral/local/approval-wording-round2-4ebd366/design.json).
The full report verifies the user scripts against the baseline, and final
comment counts against each case's SQLite DB. Runtime receipt records the
isolated endpoint shutdown. To repeat from the matching archived inputs and
`4ebd366`, use `python -m scripts.replay_approval_wording <fresh-local-dir>`,
then `python -m scripts.report_approval_action_trial --qualifiers <fresh-local-dir>`.

**Jeff:** No shipped approval or category-routing entry point calls Jeff, and
neither has a production enable flag. The staged check uses the configured
classifier reply generator; `ISPEC_ASSISTANT_PROJECT_COMMENT_APPROVAL_EVAL_MODE`
controls gate mode, not provider. The runtime category router uses
`route_tool_groups_vllm`. Setting `ISPEC_EVAL_JEFF_URL` opts in only in the
approval evaluation harness; the offline category sweep explicitly loads Jeff
with its `--checkpoint` option. These experiments remain available and were
not invoked. This was a source/deployment-config audit, not live traffic
observation on amms06.
