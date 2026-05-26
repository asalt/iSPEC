from __future__ import annotations

from typing import Any


_NONE_LABEL_ALIASES = {
    "",
    "0",
    "0.0",
    "none",
    "nan",
    "na",
    "n/a",
    "labelnone",
    "label_none",
    "label-none",
    "label=none",
    "label:none",
    "no_label",
    "no-label",
    "nolabel",
    "unlabeled",
    "unlabelled",
}


def normalize_legacy_label(value: Any) -> str:
    """Return the canonical local label string for legacy LabelFLAG values.

    Legacy data accepts both numeric label flags and textual aliases like
    ``none``/``labelnone``. Locally we keep the label in canonical numeric string
    form so ``recno_runno_searchno_label`` keys remain stable.
    """

    if value is None:
        return "0"

    text = str(value).strip()
    normalized = text.lower().replace(" ", "").replace("\t", "")
    if normalized in _NONE_LABEL_ALIASES:
        return "0"

    if normalized.startswith("label") and len(normalized) > len("label"):
        suffix = normalized[len("label") :].lstrip("_-=:#")
        if suffix in _NONE_LABEL_ALIASES:
            return "0"
        if suffix:
            text = suffix

    try:
        numeric = float(text)
    except Exception:
        return text or "0"

    if numeric.is_integer():
        return str(int(numeric))
    return str(numeric)


def experiment_run_legacy_key(
    *,
    experiment_id: Any,
    run_no: Any,
    search_no: Any,
    label: Any = None,
) -> str:
    """Build the legacy-style run/sample key EXPRecNo_run_search_label."""

    return "_".join(
        [
            str(int(float(experiment_id))),
            str(int(float(run_no))),
            str(int(float(search_no))),
            normalize_legacy_label(label),
        ]
    )
