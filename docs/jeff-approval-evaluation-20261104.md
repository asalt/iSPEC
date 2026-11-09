# Jeff three-choice approval experiment

## Decision

The isolated experiment is implemented. **Do not connect this candidate to live
authorization yet.** It makes a false approval on an explicit denial. CPU latency
is promising; classification and calibration need more work. Tool-category
relevance remains a separate experiment, not part of this change.

The subsequent short-question sweep below improves the input arrangement but does
not establish safe authorization. The short `authorize` question with a correctly
ordered object is the next diagnostic candidate, not a promoted default. See the
[prompt file index and preview commands](jeff-approval-prompts.md).

No production routes, environment defaults, authorization gates or services were
changed. No Slack messages, real notes, model downloads or installations occurred.
The optional lexical fallback was exercised only in simulation. The main model
was not run: this trial measures approval interpretation, not final reply quality.

## Bounded interface

`assistant/jeff_approval.py` uses frozen, keyword-only records for the pending
note/exchange, trial settings and shadow result. It receives exactly:

1. The initial user request to make a note.
2. The assistant's draft and save question.
3. The latest user reply.

The known pending note and project are checked before inference. The note must
appear in the shown draft. There is no older chat history, automatic summary or
judgment of scientific correctness. Instructions live in the prompt family
`experimental.project_comment.jeff_approval`; transcript content occupies `state`,
not the instructions slot. A unit test verifies that the request equals the
installed Jeff client's `build_request` / `choice_question` representation.

Jeff returns `yes`, `no` or `unsure`. The experimental policy requires a top
probability of 0.75 and a top-versus-runner-up margin of 0.20; otherwise it abstains.
These were fixed before inference and are not calibrated approval guarantees.
Jeff's separate `confidence` field is not substituted for the option probability.
All results explicitly say `shadow_only=true`, `write_outcome=not_attempted`.
They cannot be used as existing iSPEC write tickets. A future executor must save
the stored exact note, not ask a model to reconstruct its arguments after approval.

Fallback is considered only after connection failure, timeout, loading (503), or
busy (529). No, unsure, low probability, invalid responses, excessive context,
authentication failure, and other HTTP errors never trigger another approver.
There are no automatic retries. Fallback reuses the legacy short-reply interpreter
but accepts only whole `yes`, `confirm`, or `save it` replies as positive protocol
commands. This intentionally narrows the old behavior: the old token scan calls
`yes but shorten it` an approval. Existing live lexical behavior is untouched.

The input guard rejects exchanges above 16,000 serialized context characters
without truncating any message. This is a conservative character limit, not a
token estimate. Jeff also enforces its own token limit; a 422 fails closed.
The transport only allows explicit loopback HTTP health/inference endpoints,
disables proxy environment use and redirects, and sends no auth or tool schemas.

## CPU results

Installed Python 3.12 / Jeff 0.2.0, torch 2.13.0, Transformers 5.19.0;
`/mnt/e/tools/jeff/jeff-base-v1.3`, no adapters, CPU explicitly selected,
both offline flags enabled. The launcher requested four OMP/MKL threads, but later
source inspection found that Jeff's model constructor explicitly sets PyTorch's
CPU threads to its default of eight. The receipt's `threads: 4` records launcher
intent, not a verified effective thread count. One owned server on loopback 18765.
The frozen corpus comprises 12 reused synthetic cases, three adapted examples,
14 new cases and six matched length controls. Two no-pending cases skip inference.

| Measure | Result |
| --- | --- |
| Total cases / actual inference calls | 35 / 33 |
| Correct raw labels on inference calls | 17 / 33 |
| Effective three-way decisions correct, including gates | 8 / 35 |
| False approvals | 1 |
| Missed approvals | 9 of 11 positive cases |
| Median inference latency | 653 ms |
| HTTP/contract errors or live fallback use | 0 |
| Actual writes | 0 |

The explicit denial `Yes that's the right project. No, don't save a comment.`
received P(yes)=0.75696 and would pass the provisional approval policy. The note
was **not** saved. A separate quality-only reply received P(yes)=0.74084 and was
blocked only by the threshold. Lowering the threshold to recover missed approvals
would therefore be the wrong conclusion.

Matched length controls used about 273-275, 533-535 and 1313-1315 input tokens.
Times were about 0.65, 1.21 and 3.00 seconds. The longer inputs repeat neutral
details to isolate length; they are synthetic stress cases, not realistic histories
or an estimate of the maximum supported context. The positive decision fell below
threshold on both longer inputs. More room does not guarantee better interpretation.

## Formatting diagnostic

After reviewing the first run, four already-seen cases were tested in a 2x2
diagnostic: object versus plain transcript context, and action-worded versus
descriptive choices. Question, thresholds, checkpoint and the three messages were
unchanged. This is not a new holdout or an unbiased model ranking.

Plain transcript context made `confirm` score 0.868 for yes and cancellation score
0.919 for no. However, it also made the explicit denial above score **0.946 for
yes** (0.938 with descriptive choices). Object context with descriptive choices
blocked that denial but still chose yes as its top label. The repeated baseline
returned identical probabilities. All responses were structurally valid.

Thus formatting affects behavior, but fixing serialization alone does not solve
approval understanding. No arm is promoted and no threshold was tuned afterward.
Next useful diagnostics are option-order sensitivity and latest-speaker isolation
on matched approval/denial pairs, followed by fresh held-out wording. Keep the
three-choice architecture; do not grow a semantic regex exception for this case.

### Subsequent source inspection: object ordering is significant

The installed checkpoint specifies `prompt_layout=live-last`. Jeff's
`src/jeff/model.py:decision_messages` serializes object values as JSON, but places
the object's last field under `Latest`, after the options. Our object ended in
`messages`, so the **whole exchange** occupied that position. A string instead
uses state-first ordering: transcript, question, options. The earlier comparison
therefore changed prompt ordering as well as representation; it was not merely
a JSON-versus-text syntax test. Nested `role` keys are data, not native chat roles.

A correctly arranged object with the initial request and draft first, and only
`latest_user_reply` last, had not yet been tested at that point; it is included in
the subsequent sweep below. Alex's separate latest-reply-only,
two-option test correctly assigned 0.886 to no for the denial above. That is
user-supplied evidence, not our controlled rerun: context, option count, question
and choice descriptions all differ. Preserve the failures, but do not attribute
the difference to model capability alone. Compare those factors independently
before deciding whether a different model or adapter is necessary.

## Verification and reproduction

Focused deterministic checks cover input/instruction separation, missing pending
state, unchanged full messages, size limits, malformed distributions, confidence
abstention, all fallback boundaries, local-only transport, and truthful reporting.
The existing ticket/authorization and comment-tool tests also run unchanged.
Final verification: **124 focused tests passed**, Ruff and `git diff --check`
passed. Two existing FastAPI lifespan deprecation warnings remain. No full
repository test suite was run.

With an explicitly started local CPU Jeff endpoint, from the backend checkout:

```sh
PYTHONPATH=src .venv/bin/python -m scripts.eval_jeff_approval \
  --url http://127.0.0.1:18765 --output-dir tests/behavioral/local/NEW-jeff-approval
PYTHONPATH=src .venv/bin/python -m scripts.eval_jeff_approval --layout-probe \
  --url http://127.0.0.1:18765 --output-dir tests/behavioral/local/NEW-jeff-layout
```

The runner never starts a service and refuses an existing output directory.
Exit status 1 preserves observed false/missed approvals; infrastructure test
success must not conceal model failures. It saves raw synthetic requests/responses,
token use, probabilities, timings, source hashes and source snapshots for new runs.

Original receipts: `tests/behavioral/local/jeff-approval-20261104/evaluation/`.
Diagnostic receipts: the adjacent `layout-probe/evaluation/`.
Owned server logs and cleanup receipts remain alongside those ignored outputs.
The first run predates source-snapshot capture and code formatting; its request,
response and source hashes are preserved, not relabeled as a run of later source.
Both owned servers were stopped and port closure verified. The user's 8765
endpoint was not started. No commits were made; unrelated dirty files were preserved.

## Short-question and context sweep

Question, choices and context formatting now live in separate editable prompt
files. The longer question remains the baseline control. The three alternatives
contain no extra paragraph of instructions:

- `affirmation`: From this conversation, is the user's last reply an affirmation to save the project comment?
- `authorize`: Does the user's last reply authorize saving the project comment?
- `save_now`: Does the user want this draft saved now?

The fixed matrix comprises four questions x four context layouts x eighteen cases
= **288 actual CPU inference calls**. Six cases were already seen; twelve new
wordings across three synthetic scenarios were frozen before this sweep. There is
no long repetitive padding. Choices, their order, checkpoint, thresholds and
validation remain unchanged. This is a small paired comparison, not a calibrated
authorization benchmark. The twelve fresh examples are no longer held out for
subsequent prompt development.

`messages` is a plain message array; `transcript` contains the same three messages
as labeled text. Both use Jeff's state-first layout. `object` is the historical
nested-object control. `live_last` is the corrected object with only the latest
reply after the options. The array omits the redundant outer project ID, which is
already in the conversation. The object comparison changes field organization and
native prompt placement, not simply serialization syntax. A test inspects the
installed Jeff renderer without loading weights to verify these differences.

False approvals below mean a non-approval passing the fixed shadow policy, not an
actual write. Missed approvals use the six positive cases as denominator. Raw
accuracy and effective policy decisions are separate; low-confidence results
become `unsure`, even when the raw label is `no`.

| Question | Context | Raw labels correct /18 | Fresh correct /12 | False approvals (fresh subset) | Missed approvals /6 | Median ms |
| --- | --- | --- | --- | --- | --- | --- |
| baseline | object | 10 | 7 | 3 (2) | 4 | 698 |
| baseline | messages | 12 | 9 | 2 (1) | 3 | 655 |
| baseline | live_last | 15 | 10 | 0 (0) | 3 | 633.5 |
| baseline | transcript | 13 | 9 | 2 (1) | 1 | 606 |
| affirmation | object | 10 | 7 | 3 (2) | 3 | 625.5 |
| affirmation | messages | 13 | 9 | 2 (1) | 0 | 587.5 |
| affirmation | live_last | 13 | 8 | 2 (1) | 3 | 583.5 |
| affirmation | transcript | 14 | 9 | 2 (1) | 0 | 567 |
| authorize | object | 5 | 3 | 1 (1) | 5 | 603 |
| authorize | messages | 12 | 9 | 2 (1) | 2 | 558.5 |
| authorize | live_last | 15 | 10 | 0 (0) | 4 | 568 |
| authorize | transcript | 11 | 7 | 2 (1) | 1 | 593.5 |
| save_now | object | 8 | 5 | 6 (4) | 3 | 612.5 |
| save_now | messages | 10 | 8 | 5 (2) | 0 | 569 |
| save_now | live_last | 11 | 7 | 2 (1) | 2 | 564.5 |
| save_now | transcript | 6 | 4 | 4 (3) | 0 | 541.5 |

### What this changes

The extra instruction sentence did not improve raw label accuracy in the corrected
object layout: baseline and short `authorize` both reached 15/18 overall and 10/12
fresh. The short version used median 222 input tokens versus 250 and took median
568 ms versus 633.5 ms in this run. These are paired diagnostic timings on a shared
host, not a general speed guarantee. Baseline accepted three of six genuine
approvals; short `authorize` accepted two. Neither is sufficiently useful yet.

The corrected layout matters. With exactly the same short `authorize` question,
the old object scored 5/18 raw labels correct, versus 15/18 with the latest reply
alone last. That is evidence for an input-layout problem, not just a weak model.
However, the corrected layout still ranks yes above no on two explicit denials;
threshold abstention prevents those writes in this set. For the familiar
`Yes that's the right project. No, don't save a comment.` it gives P(yes)=0.6355,
while plain `confirm` receives only 0.6169. A lower global threshold would admit
that denial before recovering the confirmation. Do not lower it to improve recall.

The proposed `affirmation` wording with labeled text accepts all six positive
cases but also falsely approves the denial above at 0.9469 and the fresh question
`Would approving this create a new comment or change an old one?` at 0.7865. The
shortest `save_now` question is not the strongest. Neither raw accuracy alone nor
accepting every yes is a sufficient selection criterion.

**Recommendation:** retain the short `authorize` question and latest-reply-last
object as a shadow diagnostic candidate, without adding the longer guidance back.
Keep the current live path unchanged. Next, vary choice descriptions/order on
matched approval/denial/revision pairs while holding that question/layout fixed,
then evaluate any selected variant on genuinely new wording. Alex's two-choice,
latest-reply-only success remains a separate useful control; this sweep deliberately
did not change label count or discard the initial request/draft context.

### Evidence and cleanup

All 288 responses passed the strict response schema. No transport errors, real
fallback use, tool calls or writes occurred. Overall median latency was 596 ms;
maximum was 884 ms. Native inputs ranged from 190 to 281 tokens. The script exits
1 because the model results include false/missed approvals, not because the test
transport failed. Source, all ten prompt files, fixture snapshots, hashes and raw
wire evidence are saved under:

`tests/behavioral/local/jeff-approval-20261104/short-prompts/evaluation/`

The adjacent `receipt.json` records the checkpoint/runtime hashes, CPU selection,
versions, evaluation exit code and successful owned-server cleanup. The launcher
records requested OMP/MKL threads separately from Jeff's constructor default of
eight; it does not claim an independently measured effective thread count. Ports
18765 and 8765 were verified closed afterward. Production was untouched.

**150 focused deterministic tests passed**, including prompt variants, exact text
preservation/JSON escaping, native ordering, offline preview, frozen matrix, and
the existing authorization/write-tool regression tests. Ruff and tracked
`git diff --check` passed. Two existing FastAPI deprecation warnings remain; the
full repository suite was not run. No commits were made.
