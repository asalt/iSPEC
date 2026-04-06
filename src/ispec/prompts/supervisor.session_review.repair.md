+++
title = "Session Review Repair"
notes = "Repair invalid JSON for session review outputs."
+++
You are repairing a previous invalid JSON response for an iSPEC session review.
Return ONLY a valid JSON object that matches the schema. No markdown, no code fences, no quoted JSON.
If the previous output is truncated or malformed, ignore it and regenerate the full object from the provided context.
Required top-level keys: schema_version, session_id, target_message_id, summary, issues, repo_search_queries, followups.
The object MUST look like:
{"schema_version":1,"session_id":"...","target_message_id":123,"summary":"...","issues":[],"repo_search_queries":[],"followups":[]}
