+++
title = "Tmux Target Choice"
notes = "Choose among real tmux pane candidates only."
+++
Choose the best tmux pane candidate for the user's request.
Return only a JSON object that matches the schema exactly.
Select candidate_key='none' when no listed candidate is clearly correct.
Do not invent session names, pane ids, or tmux handles.
