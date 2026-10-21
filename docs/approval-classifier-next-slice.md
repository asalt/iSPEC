# Pending-note approval: next evaluation slice

October 19: the local Slack receipt for "looks good please save" had a valid
pending-save gate. Both model interpretations approved: the existing shadow
classifier returned `approve_save`, confidence 0.85, in 1860 ms; turn_decision
returned `approve`, confidence 1. The live lexical path returned `clarify` and
blocked the write. This was not a model failure to recognize permission.

Do not fix this by adding more phrase branches or by allowing shadow results to
write. The next live change needs one explicit interpretation-to-policy path,
bound to the pending draft, project, authorized actor and current conversation.
A small CPU classifier may replace the GPU approval interpretation, not the
state/authorization gate, and should eventually avoid duplicate model passes.

Reuse `assistant/small_classifier.py` and its existing fixed-label task. Evaluate
a local CPU backend against `tests/behavioral/data/project_comment_approval_cases.json`
plus a held-out set. Input is the latest reply, pending draft/ask and bounded
state; it does not need a full Slack thread or the complete tool catalog.
The initial 12 cases include natural approval, denial, defer, revision, quoted
approval, conditions and missing pending state. They are annotated examples,
not proof that a classifier works. The companion corpus tests check schema
compatibility only. No new model/backend is required.

Keep abstention/revision distinctions internally even if the policy ultimately
returns write/no-write. Genuine ambiguity, missing/stale drafts, timeout and
invalid output must not become permission. Measure false approvals separately
from missed approvals, plus CPU latency, memory and calibration; self-reported
confidence is not itself a calibrated probability. No Jev service/key is needed.

Before cutover, replay the current controller and candidate policy in temporary
databases, test actor/project/thread/draft binding and one-shot writes, and
retain the regression that shadow evaluation cannot authorize a real save.
Do not auto-correct already saved comments as part of this evaluation.

## Explicit live mode

`ISPEC_ASSISTANT_PROJECT_COMMENT_APPROVAL_EVAL_MODE=own` now applies the bounded
classifier policy. `auto` and `shadow` remain observational; they never grant a
write. The default approve threshold is 0.75, decision threshold 0.60, classifier
timeout 6000 ms, and ticket lifetime 120 seconds. Invalid/non-finite confidence,
invalid JSON, timeout, denial, draft-only, revision and uncertainty do not grant
a ticket. The old comment-intent pass is skipped when this policy owns a gated
turn. Response-contract shadow evaluation is unchanged.

The tool consumes a server-created, request-local ticket for at most one write
attempt, matching project, tool, session, thread, turn, source-message hash and
pending-ask hash. Auth and project access checks still apply. Evaluation tickets
cannot be consumed. Consumed attempts are not automatically retried, including
failures. Receipts record attempted content hash separately from the pending-ask
hash; this is NOT a guarantee that the model reproduced the draft verbatim.
The existing pending-draft detector remains a legacy boundary, not a new durable
draft store. Further work on exact approved-content fidelity stays separate.

Run local-only inference evaluation with:

```bash
ISPEC_RUN_VLLM_TESTS=1 .venv/bin/python -m pytest -q -s --run-behavioral tests/behavioral/test_project_comment_approval_vllm_live.py
```

This test permits only the local model HTTP endpoint and does not execute tools
or save comments. It records exact labels and latency but asserts the policy's
write/no-write outcome, since denial vs defer can differ without authorizing a
write. Initial replay exposed three false approvals (conditional, uncertain,
quoted permission); the prompt now explicitly distinguishes current unconditional
permission from those cases. Review replay results before enabling a deployment.

October 19 local validation: 13 original cases plus 8 newly written held-out
variants passed the write/no-write assertions after the prompt clarification.
Measured classifier calls were about 1.1-1.7 seconds. This is a small smoke set,
not a calibrated confidence guarantee; a few non-write sublabels still differ
from annotations. The deployment opts into `own`/0.75 in ignored `.env.local`.
No existing saved note was rewritten and no Slack test message was sent.
