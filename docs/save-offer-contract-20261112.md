# Save-offer contract checkpoint, November 12

**Not ready to promote. Approved saving remains the goal, not a read-only
workaround.** This local pass changed instructions, not authorization. No
production settings, services, database, Slack messages or model selection changed.

## Change and validation

The prior workflow text told the model to prepare and ask in prose, although
the structural guard requires a recorded `offer_project_comment_save` call.
The workflow now explicitly describes prepare -> offer -> later approval ->
save, with wording-only review separate. Tool descriptions agree with that
sequence. Preparation, offers, revision invalidation, tickets, ACLs, the 0.75
threshold and the actual write implementation are unchanged.

Deterministic validation: **189 passed** across workflow, real synthetic writes,
policy, prompt and evidence-viewer tests. Two existing FastAPI deprecation
warnings remain. This does not establish that a model chooses the right calls.

Eight conversations per model used the real `support.chat` controller and
synthetic SQLite. Six retained cases plus two new phrasings were selected before
inference. Prompt/source hashes were captured and remained unchanged during
the valid batches. No classifier decision or save call was injected.

| Case | Expected | Tulu normal planner | OLMo short-introduction candidate |
| --- | --- | --- | --- |
| New approval | Offer, then save unchanged | Fail: late offer; earlier invalid offer arguments | Fail: save blocked, offer only |
| New conditional approval | Leave pending until verification | Mechanical pass; draft includes approval instructions | Pass: pending, no save |
| Explicit save | Save reviewed note | Fail: missing earlier offer | Fail: late offer, no save |
| Natural approval | Save reviewed note | Fail: initial same-turn save blocked; later re-offers | Fail: late offer, no save |
| Revision then approval | Register ten instead of twelve, then save | Fail: prose-only revision; original draft remains | Fail: prose-only revision; original draft remains |
| Cancel | Cancel registered draft, no write | Fail: prose-only cancellation | Fail: prose-only cancellation |
| Wording-only agreement | No save/offer | Mechanical pass; no preparation, tool instructions shown to user | Pass: prepared draft remains pending |
| Printer mixed thread | Project facts only, then save | Fail: unknown draft ID, later printer-contaminated draft | Fail to save; clean, grounded draft |

Both scored **2/8 mechanically, 0/5 approved saves**. Across the valid batches:
16 conversations, 34 turns, 89 inference requests (all HTTP 200), zero writes,
and **zero approval-classifier evaluations**. These negatives therefore do not
validate classifier behavior with an active offer. There were no delivered false
save/cancellation/revision-success claims in these reviewed cases; canonical
server presentation suppressed false raw prose. Tulu still exposed tool-call
instructions in a wording-only reply, and its mechanically passing conditional
case was not a clean semantic pass.

## Diagnosis and next decision

Both integrations received the new workflow instructions and had the relevant
tools available. OLMo's trace shows a valid preparation, then ordinary prose
asking to proceed instead of a native offer. Tulu sometimes calls the offer,
but can invent its arguments, attempt saving before presentation, or re-offer
after approval. Neither missing permissions nor read-only mode explains this:
the tests explicitly enable comment writes, and the guards return precise
offer/state errors. Lowering the classifier threshold would not help because
the classifier was not reached.

The instruction correction is necessary but insufficient. The smallest proposed
next structural change is an explicit preparation intent such as
`wording_review` versus `offer_to_save`, letting one model call register and
request server presentation. Plain preparation would still create no offer;
server delivery would record the exact bound offer, and only a later approval
plus native save call could write. This reduces one model-controlled handoff
without interpreting prose as permission. **Proposed, not implemented.**
Revision/cancellation tool adoption and grounding still need independent checks.
Retest failures first, then the broader gate and actual-offer negative classifier
cases before any promotion. Keep production unchanged.

## Evidence and reproduction

[Review viewer with role-tagged inputs, raw outputs, actual calls and stored state](../tests/behavioral/local/staged-20261112/postmortems.html).
Evidence remains local/ignored, not a public conversation corpus:

- `tests/behavioral/local/staged-20261112/save-contract/olmo-instruct/`
- `tests/behavioral/local/staged-20261112/tulu-native-planner/tulu/`
- Each contains `results.xml`, `pytest.log`, `server.log`, fixture databases and
  `conversation.json` / `wire.json` per case. Parent `receipt.json` records
  source hashes, timing, owned PID/port and verified shutdown.

The local launcher reuses the existing endpoint owner and fixture harness:

```bash
../.venv/bin/python tests/behavioral/local/staged-20261112/run_save_contract.py \
  tests/behavioral/local/staged-20261112/new-reproduction
```

It checks GPU occupancy, loads one cached model at a time, uses loopback with
offline mode, and stops only owned services. The output directory must be new.
No new downloads or installations are needed. Both completed services exited
and their ports closed; the host GPU had no compute processes afterward.

Runtime: local Quadro RTX 6000, 24 GiB, driver 595.112.07, installed Python 3.12,
vLLM 0.32.0, FP16, 8192-token window, single sequence, 0.8 GPU allocation,
eager execution, TRITON_ATTN. Checkpoints: cached Tulu revision `666943798adb`,
OLMo-3-7B-Instruct revision `6e5971d9`. OLMo retains the prior short introduction,
`assistant.tools.olmo3_preamble` and the recorded Instruct chat template.
Tulu uses its normal planner and `llama3_json`. These are different, necessary
integration profiles, not a controlled identical-prompt model ranking.

Excluded attempts: the first Tulu run failed before inference because a copied
test plugin resolved its data path incorrectly. A subsequent short-introduction
Tulu run omitted the normal text catalog/protocol; the cached Tulu template does
not render the API schemas. That run is diagnostic evidence of a test-profile
mistake, not a Tulu capability score. The normal-planner rerun above replaces it.
Both excluded attempts are retained locally rather than overwritten.
