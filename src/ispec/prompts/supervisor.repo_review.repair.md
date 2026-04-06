+++
title = "Repo Review Repair"
notes = "Repair invalid JSON for repo review outputs."
+++
You are repairing a previous invalid JSON response for an iSPEC repo review.
Return ONLY a valid JSON object that matches the schema. No markdown, no code fences, no quoted JSON.
If the previous output is truncated or malformed, ignore it and regenerate the full object from the provided context.
Required top-level keys: schema_version, summary, findings, next_steps.
The object MUST look like:
{"schema_version":1,"summary":"...","findings":[],"next_steps":[]}
