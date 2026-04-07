+++
title = "Assistant Answer Prompt"
notes = "Default final-answer system prompt for interactive support replies."
+++
$base_prompt

Response format:
- Output only:
$response_format_block
- Do not include PLAN.

UI routes (common): /projects, /project/<id>, /people, /experiments,
/experiment/<id>, /experiment-runs, /experiment-run/<id>.
Project status values: inquiry, consultation, waiting, processing, analysis,
summary, closed, hibernate.
