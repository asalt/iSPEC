+++
title = "Tmux Capture Guidance"
notes = "Guidance for summarizing an already-resolved tmux pane capture."
+++
The tmux resolver already selected the best real allowlisted pane for this request.
Use the capture result directly.

By default, summarize the pane's current state in 1-3 concise sentences or a few short bullets.
For "what is going on", "status", "is it done", or similar questions, base the answer on
recent_tail.text / content, which is the bounded trailing pane capture, normally the last
40 lines for status requests. Use activity_summary, current_command, pane_title, and
preferred_alias as metadata.

Use last_nonempty_line only as a secondary clue. Do not summarize only the final line when
recent_tail.text / content contains useful recent context.

Quote or paste raw pane text only when the user explicitly asks for exact output, raw text, logs,
traceback, or a transcript.

Use the original session name, pane title, and preferred alias when referring to the capture, and do not rename the session or substitute another pane handle.
