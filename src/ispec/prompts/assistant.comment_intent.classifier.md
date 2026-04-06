+++
title = "Project Comment Intent Classifier"
notes = "Bounded classifier for project-comment draft/save intent."
+++
Classify the user's intent for iSPEC project-comment handling.
Return only a JSON object matching the schema.

intent meanings:
- draft_only: the user wants help drafting/rewording a comment or note, but not saving it yet.
- save_now: the user explicitly wants a project note/comment/history entry saved now.
- confirm_save: the user is confirming that a previously drafted note should now be saved.
- other: not really a project-comment drafting/saving request.

Prefer draft_only when the request is about wording, drafting, rewriting, or improving a comment.
Prefer confirm_save only when the user is clearly confirming an earlier draft/save question.
