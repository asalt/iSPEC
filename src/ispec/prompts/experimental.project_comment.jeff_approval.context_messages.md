+++
title = "Jeff context: nested message array"
notes = "JSON body. Historical control: the entire messages array is Jeff v1.3's last field."
+++
{
  "pending_project_id": $project_id,
  "messages": [
    {"role": "user", "content": $initial_request_json},
    {"role": "assistant", "content": $assistant_draft_json},
    {"role": "user", "content": $latest_reply_json}
  ]
}
