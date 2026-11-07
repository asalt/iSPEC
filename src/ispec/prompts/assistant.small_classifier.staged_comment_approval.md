+++
title = "Staged comment approval backstop"
notes = "Called only after the model requests saving a registered draft."
+++
Does the user's latest reply approve saving this pending project comment unchanged now?
Use the initial request, presented draft and latest reply as conversation context.
Choose approve_save, deny_save, draft_only, revise_draft, requires_explicit_confirmation, or unrelated_or_unclear.
Return JSON with label, confidence (0 to 1), and a short reason. Agreement to wording alone, quoted permission, pending conditions, or requested edits are not approval to save this version now.
