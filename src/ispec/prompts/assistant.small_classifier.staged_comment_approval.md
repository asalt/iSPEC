+++
title = "Staged comment approval backstop"
notes = "Called only after the model requests saving a registered draft."
+++
Is the user asking to save this draft to the project's history now?
Read the latest reply in the context of the original request and the draft.
"Put it in the notes", "add it as shown", and "yes, save it" ask to record this version: approve_save.
"The text reads well", "thanks", and "looks good" acknowledge the draft: draft_only.
Choose approve_save, deny_save, draft_only, revise_draft, requires_explicit_confirmation, or unrelated_or_unclear.
Return JSON with label, confidence (0 to 1), and a short reason.
