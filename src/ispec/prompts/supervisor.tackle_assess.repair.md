+++
title = "Tackle Assess Repair"
notes = "Repair invalid JSON for tackle assessment outputs."
+++
You are repairing a previous invalid JSON response for an iSPEC tackle results assessment.
Return ONLY a valid JSON object that matches the schema. No markdown, no code fences, no quoted JSON.
If the previous output is truncated or malformed, ignore it and regenerate the full object from the provided context.
Required top-level keys: schema_version, project_id, summary, findings, next_steps, questions.
The object MUST look like:
{"schema_version":1,"project_id":null,"summary":"...","findings":[],"next_steps":[],"questions":[]}
