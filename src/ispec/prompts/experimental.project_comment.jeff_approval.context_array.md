+++
title = "Jeff approval context: message array"
notes = "JSON data for Jeff, not native chat roles. Three messages only."
+++
[
  {"role": "user", "content": $initial_request_json},
  {"role": "assistant", "content": $assistant_draft_json},
  {"role": "user", "content": $latest_reply_json}
]
