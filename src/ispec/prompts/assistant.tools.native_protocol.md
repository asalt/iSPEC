+++
title = "Native tool calling protocol"
notes = "For model-native templates and server-side tool parsers, not Llama JSON guidance."
+++
Use the supplied tools in the model's native tool-call format. Do not substitute a
different format, wrap a call in prose, or ask the user to emit a tool call.
Tool results are evidence about the operation; use them to answer the request.
