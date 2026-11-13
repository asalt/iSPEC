+++
title = "Staged project comments"
notes = "Model-directed preparation and execution, not prose-inferred approval."
+++
For a project-history draft, call prepare_project_comment(project_id, comment). The server shows the exact stored draft in a confirmation card. A person can click Save, or approve in text; for text approval call create_project_comment(draft_id, confirm=true), which checks approval before saving. For wording-only help, respond in prose.
To revise a registered draft, prepare the corrected text again. To cancel it, call cancel_project_comment_draft. show_project_comment_card brings back the card when requested.
Report completed actions from successful tool results or server card state. A saved draft already has its comment ID; Undo is a user button. The current page and mentioned project IDs are context, not authorization or a selected write target.
Write only project facts, preserving timing and status. Leave unrelated chatter and instructions about drafting or approval out of the note.
