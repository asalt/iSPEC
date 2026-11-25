+++
title = "Staged project comments"
notes = "Model-directed preparation and execution, not prose-inferred approval."
+++
Project notes go in a project's history. Check the project with get_project, then call prepare_project_comment(project_id, comment); the server stores the draft and shows it in a card.
On the web a person can click Save; any user, including headless/API callers, can also ask you to save it. When they do, call create_project_comment(draft_id, confirm=true), which checks the request before saving.
To change a draft, prepare the new text. To drop it, call cancel_project_comment_draft. show_project_comment_card shows the card again.
Include the relevant project facts as the user stated them, keeping their timing and status. For help with wording only, reply in prose.
