# scratch test to ensure loaded model can call tools openai style
# requires vllm loaded with appropriate flags

# example test:
# curl -s http://localhost:8000/v1/chat/completions \
#         -H "Content-Type: application/json" \
#         -d '{
#       "model": "allenai/Llama-3.1-Tulu-3-8B",
#       "temperature": 0,
#       "messages": [
#         {"role": "user", "content": "What is 19 * 23? Use the calculator tool."}
#       ],
#       "tools": [
#         {
#           "type": "function",
#           "function": {
#             "name": "calc",
#             "description": "Evaluate a simple arithmetic expression.",
#             "parameters": {
#               "type": "object",
#               "properties": {
#                 "expression": { "type": "string", "description": "e.g. 19*23" }
#               },
#               "required": ["expression"]
#             }
#           }
#         }
#       ],
#       "tool_choice": "required"
#     }'


# example output:
# {
#   "id": "chatcmpl-8779bb609d1e33f3",
#   "object": "chat.completion",
#   "created": 1768154661,
#   "model": "allenai/Llama-3.1-Tulu-3-8B",
#   "choices": [
#     {
#       "index": 0,
#       "message": {
#         "role": "assistant",
#         "content": "",
#         "refusal": null,
#         "annotations": null,
#         "audio": null,
#         "function_call": null,
#         "tool_calls": [
#           {
#             "id": "chatcmpl-tool-836053cd8221d9e5",
#             "type": "function",
#             "function": {
#               "name": "calc",
#               "arguments": "{\"expression\": \"19 * 23\"}"
#             }
#           }
#         ],
#         "reasoning": null,
#         "reasoning_content": null
#       },
#       "logprobs": null,
#       "finish_reason": "tool_calls",
#       "stop_reason": null,
#       "token_ids": null
#     }
#   ],
#   "service_tier": null,
#   "system_fingerprint": null,
#   "usage": {
#     "prompt_tokens": 23,
#     "total_tokens": 45,
#     "completion_tokens": 22,
#     "prompt_tokens_details": null
#   },
#   "prompt_logprobs": null,
#   "prompt_token_ids": null,
#   "kv_transfer_params": null
# }
# 
