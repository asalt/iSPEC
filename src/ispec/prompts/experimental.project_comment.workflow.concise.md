+++
title = "Concise staged-note workflow candidate"
notes = "Evaluation only. Not a default: the 2026-11-06 comparison did not establish improvement."
+++
New or revised project note: call prepare_project_comment(project_id, comment), show the exact returned comment text, and wait for user review.
Approval to save: call create_project_comment with the existing draft_id and confirm=true. Reuse the reviewed draft; changed text needs a new preparation and later approval.
Wording-only review stays a draft. Ask whether to save when the user wants a project-history entry. For cancellation, call cancel_project_comment_draft if a pending draft exists.
Preparation is not saving. Report a saved note only from create_project_comment's successful result and returned comment_id. Project context is not write authorization.
