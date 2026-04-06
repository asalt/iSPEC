+++
title = "Tool Router Classifier"
notes = "Choose the best current tool groups for a user request."
+++
Select the best tool group(s) for the user request.
Return only a JSON object that matches the provided schema.
If you cannot follow the schema exactly, return: {"groups":["<primary>","<optional-secondary>",...]}

Groups:
$groups_block
