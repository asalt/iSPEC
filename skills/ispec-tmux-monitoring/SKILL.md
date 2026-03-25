---
name: ispec-tmux-monitoring
description: Observe allowed iSPEC tmux panes with hidden internal tmux-monitoring assistant tools. Use when Codex needs to list allowed panes, capture a read-only pane snapshot, or compare whether a pane changed over a short interval for staged Slack/chatbot observability workflows gated behind `ISPEC_ASSISTANT_TMUX_TOOLS_ENABLED=1`.
---

# iSPEC Tmux Monitoring

Use this skill for hidden internal tmux-pane observability during staged Slack/chatbot workflows. It is for inspection only, not tmux control.

## Preconditions

- The tools are gated behind `ISPEC_ASSISTANT_TMUX_TOOLS_ENABLED=1`.
- The tmux-monitoring tools are read-only for now.
- The default allowlist file is `configs/tmux-pane-allowlist.local.txt`.
- Authenticated chat users also need to be listed in `configs/assistant-code-tool-users.local.txt`.
- If `ISPEC_ASSISTANT_TMUX_TARGET_ALLOWLIST` is set, only matching panes are readable.
- `ISPEC_ASSISTANT_TMUX_TARGET_ALLOWLIST_PATH` can point at a different allowlist file.
- Only allowed panes are exposed. If the pane you need is not listed, stop and ask rather than guessing.

## Tools

- `assistant_list_tmux_panes`
- `assistant_capture_tmux_pane`
- `assistant_compare_tmux_pane`

## Workflow

1. Start with `assistant_list_tmux_panes` to see which panes are allowed, optionally narrow by `session_name`, and get the exact pane identifier or session-group context you can use.
2. Use `assistant_capture_tmux_pane` when you need a point-in-time view of one allowed pane.
3. Use `assistant_compare_tmux_pane` when you need to know whether a pane changed over a short interval, such as checking whether logs are still moving.
4. Keep comparisons short and targeted. This is meant for lightweight observability checks, not continuous monitoring or deep log collection.
5. If you need to restart services or manipulate tmux, switch to `ispec-dev-make` or use a human/operator path instead.

## Notes

- These tools are intended for staged Slack/chatbot observability workflows where an assistant needs a narrow operational peek without shell access.
- A hidden send helper may exist for internal plumbing/tests, but it is not exposed as an assistant tool in this phase.
- Treat snapshot results as transient operational context; summarize the findings instead of assuming they are complete logs.
- Do not treat `changed` or `unchanged` as a health verdict by itself. Pair it with the pane snapshot and the surrounding task context.
