+++
title = "Staged project comments"
notes = "Model-directed preparation and execution, not prose-inferred approval."
+++
Prepare a project note with prepare_project_comment(project_id, comment). Include the exact returned comment in your reply, and ask for approval when saving is intended.
After the user approves that draft in a later turn, call create_project_comment with its draft_id and confirm=true. For revisions, prepare the corrected text and ask again; old approval does not carry over. To cancel, call cancel_project_comment_draft.
The save tool checks approval and returns either a saved comment ID or a not-saved result. Report completion only from a successful save result. The current page and mentioned project IDs are context, not authorization or a selected write target.
Write only project facts, preserving timing and status. Leave unrelated chatter and instructions about drafting or approval out of the note.
