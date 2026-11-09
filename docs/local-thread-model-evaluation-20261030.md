# Local thread-context evaluation, 2026-10-30

## Decision

Local FP16 evaluation now works. Keep production and the existing fallback
unchanged. Neither the shorter thread prompt nor joining the quoted thread to the
current request fixes the important failures. Do not promote these experiments.

This extends the existing behavioral runner, not a new evaluation framework.
There were no Slack reads/sends, production writes, installations, driver changes,
or model downloads. The already-downloaded Instruct checkpoint was copied from
amms06 to local `/mnt/e/ispec-model-cache/olmo3-instruct-6e5971d9` (about 14 GiB).
Think was not rerun in this slice; its earlier results are not new local evidence.

## Runtime and compatibility

- Local Quadro RTX 6000, 24 GiB, SM 7.5; driver 595.112.07. Initially about
  1.5 GiB desktop use and no compute jobs. Sandbox GPU invisibility was not a
  driver failure; host CUDA allocation succeeded before testing.
- Existing Python 3.12.3, vLLM 0.32.0, torch 2.13.0+cu130, Transformers 5.19.0,
  Triton 3.7.1. No packages changed.
- Tulu: `allenai/Llama-3.1-Tulu-3-8B`, revision
  `666943798adbde0b1aff34626007e26986a3c107`, `llama3_json` parser.
- OLMo: `allenai/Olmo-3-7B-Instruct`, revision
  `6e5971d9eba42665f5bd5a0fcf047f299ce1dccc`, `olmo3` parser. BF16 source weights
  cast to FP16, not quantized. No claim of BF16 execution on this GPU.
- Both: loopback port 18030, 8192 context, one sequence, 0.80 GPU fraction,
  eager execution, Triton attention, temperature zero, 512 output tokens.
  OLMo reported 4.99 GiB KV capacity / 10,201 tokens before the 8192 context cap.

Two launch differences mattered. vLLM 0.32 accepts `--no-enable-log-requests`,
not the older `--disable-log-requests`. Its Anthropic endpoint introspection also
renders templates without a `tools` argument. OLMo's pinned template treated
Undefined as not-None and failed JSON serialization **after weights loaded**.

The evaluation-only `olmo_instruct_tools_defined.jinja` adds exactly
`tools | default(none)` before the pinned upstream body. Tests verify the body's
hash and identical rendering with defined tools (None, empty, or populated).
The checkpoint and installed packages are unchanged. This is a startup fix,
not evidence that tool selection or note quality is solved.

## Controlled comparison

Four synthetic cases: one development case (uncaptured parent with a project
number and readiness update), and three frozen held-out cases (sample receipt
mixed with printer chatter, missing project identity, and an old quoted approval
with a current no-action request). Baseline uses the current development prompt;
candidate uses `experimental.slack.thread_context.concise`. Neither is a claim
about the deployed production prompt.

The real `support.chat` controller, policy, ticket machinery and tools run against
temporary fixture databases. Exact-thread retrieval is stubbed. Tool execution is
restricted to fixture project reads/comments, and HTTP to loopback inference.
The comment tool is allowed by the fixture deployment settings, so zero writes
are not achieved merely by setting the deployment read-only. These four requests
must nevertheless issue no write ticket and create no comment.

Optional review, compaction, queueing and response contracts are disabled to
isolate the core path. This does not test HTTP authentication, real Slack delivery,
or the complete production review configuration.

| Model / prompt | Case checks | Median turn | Range |
| --- | --- | --- | --- |
| Tulu / baseline | 2/4 | 3.95 s | 3.25-8.43 s |
| Tulu / shorter candidate | 2/4 | 4.03 s | 2.88-12.52 s |
| OLMo Instruct / baseline | 1/4 | 9.51 s | 7.06-12.94 s |
| OLMo Instruct / shorter candidate | 2/4 | 8.44 s | 7.34-13.63 s |

Single sequential trials, different native templates/parsers, and some cold
startup work: these are observed turn times, not a general model ranking or
cross-platform equivalence. Fragment checks require manual draft review.

Both models fail the known parent-reference case. Tulu adds unsupported assignment
or title details; its candidate ultimately reaches the delayed-response fallback
after a false-save retry and a rejected unavailable tool. OLMo asks for the ID
despite the thread carrying it. Both include the unrelated printer in the mixed
receipt draft; OLMo candidate even calls it a printer assigned to the project.
OLMo baseline also fails to ask which project when the ID is genuinely absent.
All four arms honor the context-only/quoted-approval control without writing.

Afterward, one separate OLMo development diagnostic joined quoted context and
current request into one user message, keeping all policies and evidence labels.
The exact quote was present in the controller request; it still asked for the ID
(8.01 s). This does not justify changing production message layout.

All **17 thread trials: zero comment writes and zero issued tickets**. There are
no positive approval cases in this new slice, so this is not an approval benchmark
or promotion proof. The older approval corpus remains necessary.

One rubric omission was corrected after inspection: "not yet started" is a valid
pending-status phrase. Every affected mixed case still fails the independent
irrelevant-printer check. No model was rerun or promoted on that correction;
original run/source hashes remain in the receipts.

## Capability versus integration

OLMo's minimal native automatic tool test fails, while forced `required` completes
the tool/result round trip correctly (1.39 s). Reusing the actual iSPEC lookup
controls passes **2/2**: correct automatic `get_project(id=2)` and stored title
(2.96 s), and no guessed lookup when the ID is absent (3.38 s). Those controls
expose a narrower tool set than the thread test, so they do not isolate the cause
to one prompt sentence. This agrees with the earlier prompt/template sensitivity
report; no new parser permissiveness or blanket preamble was enabled.

The loaded thread appears in controller metadata, and the joined diagnostic
records its literal presence in model-request messages. This is no longer merely
the missing-parent transport bug. Reference resolution, automatic tool choice and
relevant-fact selection remain separate behavior failures.

## Code size and next steps

The new retrieval module is only about 84 lines. Much of the surrounding dirty
diff is tests, fixtures, documentation and deployment preflight rather than a new
runtime subsystem. This pass removes a duplicated transport-bounding loop:
local and API-derived thread context now share `bound_thread_context`.

Keep the remaining cleanup targeted:

1. Reuse the behavioral corpus for deployment semantic smoke tests instead of
   maintaining another growing set of embedded conversations. Deployment health,
   credential isolation and activation checks should stay deployment-specific.
2. Inspect the serialized model request and tool-schema budget before adding more
   prompt clauses. OLMo's stock template repeats schemas at each system message;
   the 4000-token controller estimate is not the final templated token count
   (one observed planner request was 5158 tokens). Measure before redesigning.
3. Next offline experiment: relevant-fact selection with source-linked fields,
   then a concise draft, versus a single-pass draft. Keep absent identity,
   mixed chatter, timing, attribution and quoted approvals as negative controls.
4. Try Think separately with explicit reasoning/output budgets and its template
   limitations recorded. Keep the existing backend as fallback; any false approval
   remains a promotion blocker. Do not use parser repair to improve scores.

## Reproduction and evidence

Raw logs, runtime snapshot, synthetic fixture databases, JUnit results and
source-hash receipts stay ignored under
`tests/behavioral/local/thread-models-20261030/` (about 14 MiB).
Subdirectories: `tulu-dev`, `tulu-held`, `olmo-instruct-dev`,
`olmo-instruct-held`, `olmo-instruct-joined-dev`, `olmo-instruct-native`,
`olmo-instruct-lookup`. Server logs include the failed startup attempts.

With an explicitly isolated server already running, use a new output directory:

```bash
.venv/bin/python scripts/run_instruct_refinement.py --suite thread \
  --trial baseline candidate --split dev --url http://127.0.0.1:18030 \
  --model olmo-instruct-local-eval --tool-parser olmo3 \
  --output-dir tests/behavioral/local/NEW-olmo-dev
```

Repeat with `--split held`; for Tulu use its served name and `llama3_json`.
`--trial joined --split dev` is the layout diagnostic, not a default behavior.
The runner starts no services, reads no credentials/env files, and keeps failed
model assertions as nonzero exits. Use the cached model paths and launch flags
above; OLMo needs the explicit evaluation template on this vLLM version.

Cleanup verified: owned OLMo API PID 704941 exited cleanly; port 18030 refused
connections and the GPU compute list was empty. Owned Tulu was stopped before
OLMo started. No forwards, model promotions or production changes occurred.

Final focused verification: **158 deterministic tests passed**, covering the
runner, template guard, thread fetch/context/inbox/participants, Slack handling,
Bolt integration, behavioral context routing, and approval policy/evaluation.
`git diff --check` passed. Existing FastAPI lifespan deprecation warnings remain;
the full repository suite was not run. Model failures above remain real failures,
separate from this passing infrastructure test count. Frontend stayed clean and
unrelated backend/wrapper edits were preserved. Nothing was committed.
