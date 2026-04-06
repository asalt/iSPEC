+++
title = "Support Digest Repair"
notes = "Repair invalid JSON for support digest outputs."
+++
You are repairing a previous invalid JSON response for an iSPEC support digest.
Return ONLY a valid JSON object that matches the schema. No markdown, no code fences, no quoted JSON.
If the previous output is truncated or malformed, ignore it and regenerate the full object from the provided context.
Required top-level keys: schema_version, from_review_id, to_review_id, summary, highlights, followups, sessions.
The object MUST look like:
{"schema_version":1,"from_review_id":0,"to_review_id":0,"summary":"...","highlights":[],"followups":[],"sessions":[]}
