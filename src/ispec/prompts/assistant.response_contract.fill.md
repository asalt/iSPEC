+++
title = "Response Contract Slot Fill"
notes = "Fill bounded slots for a selected response contract."
+++
Fill slots for the response contract: $contract_name.
Intent: $contract_intent
Required slots: $required_slots.
Optional slots: $optional_slots.
Use at most $max_optional optional slot(s).
Use the provided draft answer as source material, but make each slot compact.
Keep slots independent. Do not let one slot contain the whole essay.
If detail does not fit, omit it instead of spilling into extra prose.
$points_rule
Return JSON only.
Omit optional slots that are unused.
Allowed slots: $allowed_slots.
