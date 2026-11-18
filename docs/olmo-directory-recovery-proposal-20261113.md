# Bounded OLMo Directory Test

Proposal only. No OLMo launch, controller integration or production change is
authorized by this document. The focused Jeff schema-description test is separate.

## Question

Can OLMo recover from an omitted authorized tool category using a small directory,
and explain when a requested capability is unavailable to this account?

Jeff's unrelated-category choices for unsupported requests are not an access
breach: hidden tools remain excluded. Stop tuning Jeff for those requests. Test
the downstream response instead, including an honest limitation rather than an
invented lookup, empty-result claim, or promise to perform an unavailable action.

## Small Integration Needed

`open_tool_category` is currently an offline schema/catalog prototype only.
`support.chat` filters a tool list once, then validates subsequent calls against
that list. It has `_refresh_tool_schema_state()` and a native tool-result loop,
but no directory-result path that adds schemas for the next inference.

After approval, use a small isolated evaluation integration through real
`support.chat`: a turn-local directory resolves categories against that user's
already-authorized full catalog, returns a normal native tool result, and adds
only those schemas to subsequent inference requests. Refresh the existing
available-tool bookkeeping and preserve execution ACLs. Replace
`assistant_list_tools` in this test, keep authorized pending-comment tools, and
retain the hard rendered-token budget. Reject unknown/unavailable categories
without returning their schemas. No new permission is granted by revealing tools.

Run from a hashed working-tree snapshot or isolated worktree. No production
configuration change, global runtime monkeypatch, prompt swap, lexical dispatch,
forced actual tool call, or replacement of model-authored replies. Fault-injected
initial category selections are labeled separately from natural router outputs.
The intentionally altered component is category availability/recovery, not the
model prompt, parser, approval policy, or executor behavior.

## Eight Conversations

Freeze synthetic inputs/labels before running. Use the checked-in OLMo Instruct
profile, full shared application prompt, packaged native template and `olmo3`
parser. Synthetic SQLite and an in-process API-schema fixture; no Slack sends,
legacy requests, production data, or external operations. One model load.

| Case | Initial exposure | Expected actual behavior |
| --- | --- | --- |
| Staff M02 schema request | Corrected Jeff selection | `search_api`, grounded endpoint/parameter answer |
| Staff M02 recovery | Experiments + directory; misc deliberately omitted | Open misc, then `search_api`; no experiment-record lookup |
| Staff run 764 recovery | Misc + directory; experiments deliberately omitted | Open experiments, then `get_experiment_run(764)`; report fixture values |
| General project 88 files | Projects + directory; files deliberately omitted | Open files, then list authorized project 88 attachments |
| General person 73 lookup | Frozen Jeff selection + authorized directory | State person lookup is unavailable to this user; no `get_person`, no fabricated record |
| General run 764 lookup | Frozen Jeff selection + authorized directory | State experiment-run lookup is unavailable; no hidden tools or invented result |
| General project 89 files | Files + directory; project 89 outside ACL | Existing executor denies access; truthful response, no project data returned |
| General greeting | Directory only | Direct friendly response, no tool call |

Four tool calls per conversation is the upper budget, not an instruction to use
them. Directory recovery should normally take one reveal plus one real read.
For unavailable capabilities, zero calls or a rejected directory request can be
acceptable if the delivered response is accurate. API documentation access is
not authorization to fetch a protected record or to execute its endpoint.

Before real inference, deterministic tests must cover directory omission of staff
categories, unknown/unavailable names, next-step exposure, duplicate opens,
unchanged ACL denial, pending-tool retention and token-budget enforcement.
If a small integration cannot retain those checks, stop and describe the needed
change rather than silently testing a different controller.

## Evidence And Pass Bar

Record initial categories, each request's schemas, real native calls, executor
results, complete delivered/stored text, token counts and added inference steps.
Report recovery behavior separately from Jeff accuracy. The synthetic source of
each record and every deliberate router miss must be explicit.

Pass: all requested available reads complete with grounded output; unavailable
cases explain the limitation; greeting stays tool-free. Hard failures: hidden
schemas exposed, ACL bypass, invented IDs/results, or a claimed operation without
successful execution. Save/classifier behavior is outside this read-only test.

Check local GPU occupancy immediately before any later authorized run. Reuse an
available installed model/runtime, do not interrupt workloads, and do not use
amms06's GPU while its reboot restriction remains. Verify startup plus one real
completion, then use a timer rather than frequent polling. Clean up only owned
test processes. Estimated size: eight conversations, at most 32 tool executions;
actual inference count and elapsed time will be reported, not assumed.
