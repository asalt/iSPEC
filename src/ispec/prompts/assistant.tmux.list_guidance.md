+++
title = "Tmux List Guidance"
notes = "Guidance for asking the user to choose from real tmux panes when resolution is ambiguous."
+++
The tmux resolver did not find one unique pane to inspect.
Use the returned tmux pane list and ask the user to choose from the real handles in that list,
preferably preferred_alias, capture_target, or pane_id.
Do not invent session names or pane handles.$scope_suffix
