"""Derived display status for legacy ExperimentRun processing metadata."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def _has_text(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def derive_legacy_grouping_status(values: Mapping[str, Any]) -> str:
    """Summarize raw legacy grouper flags without persisting another status."""

    started = values.get("legacy_grouper_started")
    completed = values.get("legacy_grouper_completed")
    failed = values.get("legacy_grouper_failed")

    if failed is True:
        return "failed"
    if completed is True:
        return "complete"
    if started is True:
        return "running"
    if all(value is None for value in (started, completed, failed)):
        return "unknown"
    return "pending"


def derive_legacy_import_status(values: Mapping[str, Any]) -> str:
    """Summarize raw legacy E2G import evidence with conservative precedence."""

    started = values.get("legacy_e2g_import_started")
    completed = values.get("legacy_e2g_import_completed")
    fix_requested = values.get("legacy_e2g_import_fix_requested")
    imported_at = values.get("legacy_e2g_imported_at")
    error = values.get("legacy_e2g_import_error")

    if _has_text(error):
        return "failed"
    if fix_requested is True:
        return "needs_fix"
    if completed is True or imported_at is not None:
        return "imported"
    if started is True:
        return "running"
    if all(value is None for value in (started, completed, fix_requested, imported_at)):
        return "unknown"
    return "pending"


def legacy_experiment_run_display_status(values: Mapping[str, Any]) -> dict[str, str]:
    return {
        "legacy_grouping_status": derive_legacy_grouping_status(values),
        "legacy_import_status": derive_legacy_import_status(values),
    }
