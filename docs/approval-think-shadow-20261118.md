# Think as a save-intent classifier

**Useful offline comparison, not a live approval check.** Every leading yes/no
label matched the expected action, including P07. Only 19/24 final messages
were a single valid response; none arrived within the existing 6 s deadline.
No label was used to authorize a write.

|Policy|Actual: TP / FN / TN / FP|Hypothetical: TN / FP|
|---|---|---|
|Baseline Instruct a|5 / 2 / 2 / 1|13 / 1|
|Instruct y1 or r1|6 / 1 / 3 / 0|14 / 0|
|Think, strict eventual response|6 / 1 / 3 / 0|14 / 0|
|Think, strict and timely|0 / 7 / 3 / 0|14 / 0|

These are allow/block counts: invalid or late responses block, even when the
model has not given a usable "no". Ten **recorded calls** and fourteen **hypothetical**
pending follow-ups, all from synthetic conversations. y1 is the direct comparison:
same system question, messages, original assistant presentation and yes/no +
confidence schema. Think used a 2,048-token total output cap; the Instruct
references are saved scores. Temperature 0 and confidence cutoff 0.75.

**What it understood.** P03 ("file it with this project's notes") returned
`yes 0.95`; Think's diagnostic output says "the user's reply is a command to
file it". N13 ("once the PI has verified the count") returned `no 0.95`,
observing that "the user explicitly stated the condition for saving".
P04 returned `yes 0.95`, N02's wording praise `no 0.95`. P07 returned
`yes 0.99` in 14.61 s, whereas y1 and r1 both blocked. P02 instead lost a
usable save: its final started with the correct yes but continued after the
closing JSON brace. N07, N08, N12 and N15 did likewise after their correct no.
N15's appended explanation also drifted into *whether a draft was already
saved*, not just whether this user requested saving it. Inspecting a leading label
is diagnostic only; no production parser or acceptance rule was relaxed.

**Budget and runtime.** All 24 finished with `stop`, none with `length`. Maximum
total completion was 655 tokens. Reasoning tokens per check were 241–639
(median 336.5); wall time was 9.94–22.83 s (median 13.16 s, model load excluded).
The server's usage breakdown reported zero reasoning tokens on all 24 despite
returning separate reasoning and final content. Counts here use emitted token IDs
before the first closing tag, excluding the boundary tokens. The paired N02
request with `chat_template_kwargs.enable_thinking=false` had identical input
IDs, 290 reasoning tokens and identical final JSON. No independent off arm was run.

**Parser evidence.** All 24 emitted `[524,27963,1363]` for `</think>\n\n`.
The installed vLLM 0.32.0 OLMo parser's token-level `is_reasoning_end` returned
false on those actual boundary IDs, and true for the bare tag
`[524,27963,29]`. String-level final-content separation succeeded on all 24.
This directly reproduces the earlier boundary mismatch on emitted IDs and
explains how unconstrained prose can reach the final field. No parser patch,
timeout change, or claim that a parser fix would make the model faster.

[Each turn, token counts, timings and confusion matrices](../tests/behavioral/local/approval-think-shadow-4ebd366/comparison.md)
and [runtime receipt](../tests/behavioral/local/approval-think-shadow-4ebd366/receipt.json).
Stock Think checkpoint `d97e442d7cc678210054dbcc9b440894d62c89a4`,
FP16 on the local Quadro RTX 6000; only the checkpoint files came from the
existing mount. No Instruct process was stopped, no Slack or DB connections,
no changes to service config. The owned loopback server exited and its port closed.

Reproduce from the saved round-two y1 inputs with
`python -m scripts.replay_think_approval tests/behavioral/local/<fresh-name>`
and `python -m scripts.report_think_approval tests/behavioral/local/<fresh-name>`.
