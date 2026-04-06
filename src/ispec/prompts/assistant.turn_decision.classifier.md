+++
title = "Turn Decision Classifier"
notes = "Structured decision object for support and scheduled assistant turns."
+++
Produce a structured turn decision for the iSPEC assistant.
Return only a JSON object that matches the schema exactly.

Decision fields:
- primary_goal: the main job of this turn.
- needs_clarification / clarification_reason: only true when the assistant cannot safely continue without more input.
- tool_plan: whether tools should be used, the main tool group, up to two secondary groups, and one preferred first tool if obvious.
- write_plan.mode: none | draft_only | save_now | confirm_save.
- response_plan.mode: single or compare. Prefer single unless side-by-side alternatives would materially help.
- response_plan.contract_cap: choose the smallest response contract cap that should govern the final answer.
- reply_interpretation: classify the latest user reply only when the turn is awaiting a bounded follow-up decision.

Primary goals:
- answer_question: normal lookup/explanation/help answer.
- inspect_state: inspect tmux, repo, logs, or operational state.
- draft_project_comment: help draft/reword a project comment without saving yet.
- save_project_comment: user wants a project comment saved now.
- confirm_save: user is confirming that a previously drafted comment should now be saved.
- automation_task: internal scheduled assistant task.
- devops_task: task mainly about developer/devops operations or staff automation.

Clarification reasons:
- none
- missing_identifier: needs a project/session/etc identifier.
- missing_comment_text: user wants to save a comment but has not provided the content yet.
- ambiguous_target: multiple plausible targets or unclear target.
- missing_confirmation: an explicit confirmation is still needed before a write.

Write-plan rules:
- draft_project_comment must use draft_only.
- save_project_comment must use save_now.
- confirm_save must use confirm_save.
- all other primary goals should usually use none.

Tool-plan rules:
- Prefer using tools when the answer should be grounded in current iSPEC state.
- preferred_first_tool should be empty unless one obvious first tool lookup stands out.
- Keep group selection tight; choose the smallest sufficient set.

Response rules:
- Prefer the smallest response contract cap that fully answers the turn.
- Prefer single unless compare mode is genuinely helpful.

Reply-interpretation rules:
- Use none when there is no awaiting_reply_state in context.
- If awaiting_reply_state is present, choose exactly one of: approve, deny, defer, modify, unclear.
- approve: clear confirmation or permission to proceed.
- deny: clear refusal, cancellation, or 'do not do it'.
- defer: not now / later / hold off, without cancelling permanently.
- modify: revise or tweak something before proceeding.
- unclear: ambiguous, unrelated, or not safely actionable.
$scheduled_rules_block$groups_block$response_modes_block$contract_caps_block
