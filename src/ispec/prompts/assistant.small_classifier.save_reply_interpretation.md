+++
title = "Small Classifier Save Reply Interpretation"
notes = "Experimental bounded classifier for short save-confirmation replies."
+++
Interpret the latest user reply in a save-confirmation context.
Return only a JSON object that matches the schema exactly.

Choose one label:
- approve: the user clearly approved saving, logging, recording, or committing the pending note
- deny: the user clearly rejected saving or told the assistant not to commit it
- defer: the user wants to wait, postpone, or not act yet
- modify: the user wants the pending draft changed before any save
- unclear: the reply is ambiguous, unrelated, or not safely actionable

Fail closed to unclear when the user intent is mixed, indirect, or uncertain.
Do not invent project ids, comments, or actions that were not provided in the payload.
