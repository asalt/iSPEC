# OLMo on amms06: isolated FP16 evaluation

Follow-up: [Instruct interface refinement](instruct-refinement-2026-10-25.md)
adds frozen held-out trials, actual-controller grounding failures, and an opt-in
canonical-preamble experiment that is not recommended for promotion.

## Scope and recommendation

This extends the [October 23 comparison](model-runtime-comparison-2026-10-23.md)
rather than repeating its entire matrix. Production configuration, services,
databases and Slack were not test targets. All writes in controller trials use
temporary synthetic databases; all exposed tools are bounded fixture tools.
There were no Slack sends, new drivers, package upgrades or production promotion.

Both standard Ai2 checkpoints load on the existing amms06 vLLM installation.
Instruct is a viable experimental alternative, but not ready to replace the
working backend. Think is useful to evaluate, not a drop-in replacement for the
current short-budget approval path. Passing JSON alone is insufficient: tool
selection, argument grounding, meaningful note selection and response truth
remain independent requirements. Any false approval is a promotion blocker.

## Exact runtime and artifacts

| Item | Observed value |
| --- | --- |
| GPU | Quadro RTX 6000, 24 GiB, compute capability 7.5 |
| Driver | 565.77 |
| Interpreter | `/home/amms06/venvs/py312/bin/python` |
| vLLM / PyTorch | 0.13.0 / 2.9.0 |
| Torch CUDA / Transformers / Triton | 12.8 / 4.57.3 / 3.5.0 |
| Instruct checkpoint | `allenai/Olmo-3-7B-Instruct` |
| Instruct revision | `6e5971d9eba42665f5bd5a0fcf047f299ce1dccc` |
| Think checkpoint | `allenai/Olmo-3-7B-Think` |
| Think revision | `d97e442d7cc678210054dbcc9b440894d62c89a4` |
| Source / execution dtype | BF16 safetensors / explicit cast to FP16 |
| Native BF16 on this GPU | Not supported; FP16 CUDA execution verified |
| Evaluation settings | 8,192 context, one sequence, 0.85 GPU utilization, eager execution, Triton attention |
| Endpoint | Loopback port 18006, isolated served-model names |
| Parser | `olmo3` tools; Think additionally uses `olmo3` reasoning |

Each checkpoint is about 14.6 GB of weights. Instruct is in the normal HF cache;
Think is in `/mnt/e/ispec-model-cache/hub` to preserve root space. Both remain
available on disk. At the space check, amms06 had 23 GiB root and 275 GiB `/mnt/e`
available. Loaded GPU snapshots were about 20.5 GiB, with roughly 3 GiB free.
Those are snapshots, not measured peak usage under arbitrary workloads.

The server reported 13.6261 GiB model weights and 6.10 GiB available KV cache for
Instruct. An 8K allocation is feasible, but that does not prove useful 8K recall.
A deliberately repetitive 7,044-token probe produced an empty final answer and
was rejected. Do not extrapolate either this artificial failure or the allocation
to production context quality, 64K support, concurrency or other devices.

Two additional synthetic, record-shaped recall probes were more informative:
Instruct recovered start/middle/end markers at 3,777 prompt tokens (6.11 s) and
6,977 prompt tokens (10.15 s). The smaller case followed the requested JSON
format; the larger returned correct markers in plain text instead. Thus roughly
7K input is computationally usable for this simple recall task, while output
contract fidelity already needs testing. This is not a long-conversation quality
benchmark and neither prompt was a real project history.

No vLLM update was needed. The earlier artifact blocker was the cached Ollama
GGUF's `olmo3` architecture missing from the GGUF loader mapping, not failure of
the standard checkpoint or CUDA runtime. The installed vLLM registry supports
the standard architecture and parser. FlashAttention 2 is unsupported on SM 7.5;
the verified Triton backend avoids that issue. Do not assume Mac equivalence.

[Ai2 documents Instruct function calling and separate Think post-training](https://docs.allenai.org/models/olmo/index.html).
That supports trying both, not assuming identical tool behavior.

## Fresh held-out trials

The synthetic corpus was fixed before inference:
`tests/behavioral/data/model_followup_20261025/`. It contains 11 gated approval
cases, one ungated control, and six note cases. Expected answers are stored in
a separate manifest, never model input. Temperature is zero; these are single
trials, not statistically robust estimates. Prompts and hashes are retained.

| Test | Instruct FP16 vLLM | Think FP16 vLLM |
| --- | --- | --- |
| Prompt-only approval valid JSON | 11/11 | 11/11 |
| Exact approval label | 9/11 | 9/11 |
| False / missed approval, expanded budget | 0 / 0 | 0 / 0 |
| Approval median / p95 | 1.55 / 1.83 s | 15.77 / 18.04 s |
| Actual classifier + policy, 160 tokens / 6 s | 12/12 including ungated control | 1/12, only ungated control passes |
| Mechanical note checks | 4/6 | 4/6 |
| Note median / p95 | 1.41 / 2.62 s | 27.03 / 31.83 s |

Prompt-only Instruct used the task budgets (160 approval / 512 note); Think used
1,024 output tokens including reasoning. The prompts were identical, but the
output allowances were not. Both exact-label disagreements are among
non-authorizing labels, not approvals. Think's actual 160-token classifier calls
were incomplete; all failed closed without a ticket. Longer reasoning cannot be
silently substituted into the six-second production contract.

Read note text, not just the fragment score. Instruct accurately wrote that no
files had been uploaded and the client had not been notified, but a naive string
check incorrectly penalized the negated words. Its real failure was proposing a
project note about unrelated lunch chatter. Think correctly ignored lunch chatter,
but omitted the shared-folder/profile detail in a multi-update note, and its
quoted-instruction case exhausted 1,024 tokens without a usable final answer.
Do not change held-out labels/checks after seeing output to inflate scores.

A separate 2,048-token diagnostic on the truncated Think case finished in 37.72 s
but proposed the injected "all work finished and saved successfully" text instead
of the source's pending peptide review. This is a semantic/source-grounding failure,
not merely truncation. No write was executed. More reasoning budget did not make
the proposal reliable; the original 1,024-token score remains unchanged.

## Actual interface and diagnosis

The opt-in `test_model_runtime_followup_live.py` exercises `generate_reply` and
the actual `support.chat` controller with temporary project/comment/session
databases, real policy/tickets, and real fixture tool execution. It is not a
Slack transport or HTTP authorization test. Non-inference network requests and
tools outside the fixture allowlist are rejected.

Instruct's initial six cases: four passed, two failed.

- Native `auto` lookup with a short custom system prompt returned prose, not a
  tool call. `required` and named-tool selection completed tool/result round trips.
- The actual controller counted the three fixture projects successfully (3.49 s).
- Natural approval classified at 0.92 and issued a valid temporary ticket, but
  the model invented optional `person_id=2`. The tool rejected the nonexistent
  person, wrote no comment, and the response honestly reported failure (8.15 s).
- Denial issued no ticket and executed no comment write (2.78 s).
- A separate required-arguments-only schema diagnostic completed the temporary
  write. This isolates optional argument grounding; it is not a production schema
  change or permission bypass. Its timing overlapped short diagnostics and is not
  a clean comparative latency measurement.

Raw template ablations showed substantial prompt sensitivity: adding explicit
function instructions is not a universal cure, and one default-template response
invented a function name. Strict normalization rejects unavailable tools rather
than interpreting prose as an executable command.

Think's pinned stock template ignores the standard `tools` variable and `tool`
result role. It expects custom message `functions` and `environment` instead, and
adds a no-functions statement to ordinary system messages. A server tool parser
cannot repair context that the template never exposed. The separate, eval-only
`olmo_think_tools.jinja` trial uses the pinned Instruct template's OpenAI mappings
with Think's generation marker. It is not enabled in production.

All four stock-template tool/controller trials failed. The three small native
tool probes exhausted their budget (the required-tool path surfaced a server 400
while interpreting unfinished reasoning as tool JSON). The controller did execute
`count_all_projects` successfully, then timed out while answering after the result.
Its router also exhausted a separate 200-token budget. Thus "Think cannot call
tools" would be too broad; the stock context contract and short auxiliary budgets
prevented a complete usable turn.

The isolated compatible-template trial fixed result visibility enough for the
actual count controller to answer correctly in 43.81 s. However, that run used
the existing deterministic read-only count policy after the model returned a
schema as prose, not a native call. The short-system native probe still exhausted
2,048 tokens. Therefore this passing controller result must not be presented as
successful native tool selection or general Think readiness.

With the compatible template's default system instructions instead of the short
custom system, Think identified `get_project(project_id=42)` in 16.61 s, but
serialized JSON inside the XML call tags rather than the parser's Python-style
function syntax. No normalized tool call resulted and nothing was executed.
This separates understanding the requested lookup from obeying the runtime's
tool-output contract. No prose-salvage parser was added to make this case pass.

Ollama also had a verified adapter issue: its 0.40.8 OLMo renderer retains only
the first system message. A two-system extraction probe returned `UNKNOWN`,
whereas merging those same system instructions returned `SAMPLE-729`. The local
adapter now coalesces system messages while preserving user/tool roles and order;
unit tests cover this. The custom-system/default-function-instruction distinction
still remains. This repair has not been deployed.

The repaired native iSPEC Ollama path was then exercised with the original two
system messages and returned `SAMPLE-729` correctly (5.62 s including 5.41 s cold
load). It was unloaded immediately afterward. This verifies context transport,
not broad model or native-tool equivalence between Ollama and vLLM.

## Evidence, snapshots and next steps

Raw receipts are intentionally local under
`tests/behavioral/local/olmo-evaluation-20261025/`: prepared packets/manifests,
outputs, scores, controller traces, JUnit XML, launch receipts and server logs.
They are diagnostic evidence, not automatic deployment instructions.

`scripts/capture_model_runtime.py` prints a small timestamped JSON inventory:
selected package/compiler/runtime versions, GPU total/used/free memory,
checkpoint revision and metadata hashes. Optional CUDA probing uses one FP16
element. It does not read process environments, dump credentials, install
anything, or claim the installed compiler built the wheels. Cache blob names
are identifiers, not claimed independently verified full-weight checksums.
An inventory snapshot becomes a working-version receipt only when paired with
passing test results and the exact launch settings.

Example reproduction against an already-running, explicitly isolated endpoint:

```bash
ISPEC_RUN_LLM_TESTS=1 \
ISPEC_VLLM_URL=http://127.0.0.1:18006 \
ISPEC_VLLM_MODEL=olmo-eval \
ISPEC_VLLM_TOOL_CALL_PARSER=olmo3 \
ISPEC_VLLM_REASONING_PARSER= \
python -m pytest tests/behavioral/test_model_runtime_followup_live.py \
  --run-behavioral -q -s
```

Set `ISPEC_APPROVAL_EVAL_CASES_PATH` to the fresh approval corpus for the separate
`test_project_comment_approval_vllm_live.py` run. For the Think diagnostic only,
use its served name, reasoning parser `olmo3`, and
`ISPEC_EVAL_MAX_OUTPUT_TOKENS=2048` for the general tool probes. That override
does not change the approval classifier's 160-token contract or the controller's
other stage-specific limits. Do not enable these tests against production or
raise ordinary application limits from the shell by accident.

1. Retain the existing working backend. Keep manual model profiles and rollback
   configuration, with no automatic switching or promotion.
2. Resolve tool/context contracts independently of model quality; repeat the
   smallest fixture tool cycle before a larger conversation suite.
3. Add more genuinely held-out approval turns, quoted approvals, corrections and
   source-grounded multi-update notes. Require zero false approvals, honest write
   outcomes and acceptable end-to-end time. The current small sample is not enough.
4. Test realistic bounded histories at several prompt sizes and record input,
   reasoning/output tokens, timeouts and total latency. Do not raise budgets
   globally merely to hide Think's incompatibility with the short classifier path.
5. Generate narrated behavioral-review videos later from saved receipts, using
   distinct prompt/output voices and visible model/runtime/case labels. Do not
   rerun models to produce a nicer demonstration or invent unrecorded reasoning.

## Verification and cleanup

The focused deterministic suite passed **103 tests**, covering adapter roles and
constraints, strict runtime protocol, support-route fallback boundaries, approval
guards, prompt observability, evaluator contracts, runtime inventories and the
experimental template. Existing FastAPI lifespan deprecation warnings remain.
The live-test modules also correctly skipped all 29 cases without explicit opt-in.
The model failures above are retained, not hidden by the passing unit-test count.
No full repository suite was run.

At 2026-10-25 06:07 UTC, the owned evaluation servers were stopped, Ollama had no
loaded models, and GPU usage returned to 474 MiB with 23,544 MiB free. Both
`ispec-mariana.service` and `ollama.service` remained active. The two owned SSH
forwards were then closed. Downloaded checkpoints and local evidence remain.
No production runtime/configuration or real project data was changed.

The tested launch includes eager execution and an explicit 0.85 memory fraction.
It is not certification of every Make/profile default or CUDA-graph configuration.
Reuse the exact receipt for the next isolated trial before generalizing deployment.
