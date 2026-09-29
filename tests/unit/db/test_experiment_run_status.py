from __future__ import annotations

import pytest

from ispec.db.experiment_run_status import (
    derive_legacy_grouping_status,
    derive_legacy_import_status,
)


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ({}, "unknown"),
        ({"legacy_grouper_started": False}, "pending"),
        ({"legacy_grouper_started": True}, "running"),
        ({"legacy_grouper_completed": True}, "complete"),
        (
            {"legacy_grouper_completed": True, "legacy_grouper_failed": True},
            "failed",
        ),
    ],
)
def test_derive_legacy_grouping_status(values, expected):
    assert derive_legacy_grouping_status(values) == expected


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ({}, "unknown"),
        ({"legacy_e2g_import_started": False}, "pending"),
        ({"legacy_e2g_import_started": True}, "running"),
        ({"legacy_e2g_import_completed": True}, "imported"),
        ({"legacy_e2g_imported_at": "2026-09-04T10:00:00"}, "imported"),
        (
            {
                "legacy_e2g_import_completed": True,
                "legacy_e2g_import_fix_requested": True,
            },
            "needs_fix",
        ),
        (
            {
                "legacy_e2g_import_fix_requested": True,
                "legacy_e2g_import_error": "ODBC import failed",
            },
            "failed",
        ),
    ],
)
def test_derive_legacy_import_status(values, expected):
    assert derive_legacy_import_status(values) == expected
