+++
title = "Tmux Capture Guidance"
notes = "Guidance for summarizing an already-resolved tmux pane capture."
+++
The tmux resolver already selected the best real allowlisted pane for this request.
Use the capture result directly.

By default, summarize the pane's current state in 1-3 concise sentences or a few short bullets.
Prefer structured fields like activity_summary, last_nonempty_line, current_command, pane_title,
and preferred_alias over dumping raw pane content.

Quote or paste raw pane text only when the user explicitly asks for exact output, raw text, logs,
traceback, or a transcript.

Do not rename the session, do not invent another pane handle, and do not claim you inspected a different pane.
