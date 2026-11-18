# Router Repair And Offline Category Evaluation

November 13, 2026. Local evaluation only; uncommitted, no production changes.

## Implemented Repairs

1. `assistant/service.py` adds `disable_any_whitespace=True` in the existing
   vLLM structured-output normalizer, covering native JSON/choice constraints
   and legacy guided aliases. Schemas, prompts, output budgets and approval
   thresholds are unchanged. Caller dictionaries are copied, not mutated.
2. The OLMo Instruct profile and launcher pass
   `--structured-outputs-config '{"backend":"xgrammar","disable_any_whitespace":true}'`.
   Local vLLM 0.32 accepts the request flag but its V1 grammar backend reads the
   runtime-wide configuration, so the request flag alone is insufficient.
   This uses the supported configuration, with no vLLM source edits or monkeypatch.
3. `support.chat` uses a bounded fallback when routing is unusable and no group
   hints exist: authorized project tools, comment workflow tools and discovery.
   It filters the existing authorized schemas, grants no new tools, and logs the
   fallback explicitly. Other routing outcomes retain their existing behavior.

The request-level flag intentionally applies to Tulu's structured calls too;
its ordinary prose/native calls are unchanged and its runtime profile receives
no new server flag. Ollama bypasses this adapter. The KEEP/REWRITE choice grammar
already restricts output to literal choices; the flag does not change its choices.
Whitespace in JSON string values remains meaningful and permitted.

## Exact Local Replay

Cached OLMo Instruct FP16; vLLM 0.32.0; local Quadro RTX 6000, driver 595.112.07.
No compute workload was present before launch or after cleanup. A native public
vLLM module entrypoint ran on loopback port 18040; the owned server exited zero.
No application tools, databases or Slack operations were executed by this replay.

| Saved request | Before | After | Decision | Latency |
| --- | --- | --- | --- | --- |
| Supplied full-gate `conversation_26` (N19, not N07) | 200 tokens; incomplete JSON after `clarify` | 27 tokens; stop; schema-valid | projects, 0.95, clarify=true | 6.42 s, first request/grammar compilation |
| Repaired-batch `conversation_3` (N07 revision) | 200 tokens; incomplete JSON after `clarify` | 24 tokens; stop; schema-valid | projects, 0.85, clarify=true | 1.18 s |

Both preserve the recorded request messages/schema/budget; only the whitespace
option and supported runtime setting changed. These are completion checks, not
proof that every routing judgment or complete comment conversation is correct.
In particular, `clarify=true` is the returned model judgment, not a permission.

Saved evidence: `tests/behavioral/local/router-repair-20261113/olmo-replay/` has
the frozen working tree, profile, server log and exact requests/responses in
`receipt.json`. The standalone replay script is retained alongside it. The
historical full-gate and focused-repair evidence was not overwritten.

## Fallback Verification

Re-rendering the exact oversized saved N07 input with the bounded catalog gives
**17 tools / 3,545 tokens**, down from **51 / 7,236**. The original offline count
matches the serving tokenizer's saved count exactly. Both the 6,000 input budget
and the 8,192 context window with 1,024 output reserve are now satisfied.
See `fallback-token-count.json` in the same dated evidence root.

The controller/rendered-prompt regression was first run as `xfail(strict=True)`;
it then flipped to XPASS after the repair. Its marker was removed after that
verification, retaining all assertions as a normal passing regression.
118 focused backend tests and all 127 wrapper tests pass. These include request
flags, unchanged Ollama handling, untouched ordinary generation, actual fake-server
launcher expansion, retained workflow tools and no tool grants from fallback.
Additional overlapping runs passed 141 controller/workflow tests and 44
classifier/usage-log tests, including the separately configured classifier's
outbound whitespace flag. These totals are separate runs, not additive coverage.

## vLLM 0.13 Compatibility

Upstream v0.13.0 uses the same request field in
[StructuredOutputsParams](https://github.com/vllm-project/vllm/blob/v0.13.0/vllm/sampling_params.py)
and the same runtime field in
[StructuredOutputsConfig](https://github.com/vllm-project/vllm/blob/v0.13.0/vllm/config/structured_outputs.py).
Its [XGrammar backend](https://github.com/vllm-project/vllm/blob/v0.13.0/vllm/v1/structured_output/backend_xgrammar.py)
also reads runtime-wide configuration. This verifies upstream spelling and
mechanism, not amms06's installed/running behavior. amms06 was not accessed; it
still needs its pending reboot and an isolated runtime check before deployment.

## Jeff Category Experiment

Stopped and deferred after Alex identified competing workloads. The owned process
exited (130, interrupted); no other jobs were changed. One CPU-only weight load
of existing Jeff v1.3 took 6.1 seconds. Inputs and labels for 60 deduplicated
recorded requests were frozen before inference, with 13 independent yes/no
category-relevance questions per request.

Six varied timing probes completed at one and four CPU threads: medians **33.7 s**
and **73.4 s** per 13-category request. The eight-thread trial was interrupted.
Host load averages immediately afterward were 35.13/30.65/27.91. These are
contention-affected wall times, not evidence that four threads are intrinsically
slower or that Jeff is unsuitable on an idle CPU. The full accuracy comparison and
threshold curve were not completed; the runner had not yet persisted category
scores. `partial-timing.json` records the completed timing summaries and limits.
No approval thresholds changed and no Jeff controller integration was made.

The installed implementation creates one full input sequence per category,
tokenizes/pads batches, runs the backbone with `use_cache=False`, and scores
options from the final hidden state. At batch size eight, 13 categories use
two batched forward passes, repeating the source context across the 13 rows.
There is no autoregressive answer generation or cross-question prefix reuse.
Backbone computation is the likely dominant cost, but preparation versus forward
time has not been measured separately. The next small timing check should separate
those stages with one resident model, rather than rerunning the full comparison.

These are synthetic evaluation conversations with recorded model outputs, kept
distinct from real collected Slack examples. They are not training data. Only
projects, people, experiments, schedule and no-tool greeting are represented as
expected categories; scoring all 13 does not validate all 13. The saved router
usually saw only the latest user request, so contextual follow-ups lack preceding
drafts. Report that limitation rather than attributing all misses to Jeff.

The original binary-question runner is retained as `runner.py` in that interrupted
run's evidence directory. The current runner follows Alex's subsequent direction:
one context-bearing question, cumulative selection, and staff/general-user passes.
See [the contextual replay](jeff-contextual-router-20261113.md) for its distinct
inputs, selection rules and reproduction command. The old timings are not the
speed of the new single-question design.
