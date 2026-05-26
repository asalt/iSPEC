from __future__ import annotations

import pytest

from ispec.omics.labels import experiment_run_legacy_key, normalize_legacy_label


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, "0"),
        ("", "0"),
        ("0", "0"),
        ("0.0", "0"),
        ("none", "0"),
        ("labelnone", "0"),
        ("label=none", "0"),
        ("label_0", "0"),
        (126, "126"),
        (126.0, "126"),
        ("126.0", "126"),
        ("127N", "127N"),
    ],
)
def test_normalize_legacy_label(raw, expected):
    assert normalize_legacy_label(raw) == expected


def test_experiment_run_legacy_key_uses_numeric_label_alias():
    assert (
        experiment_run_legacy_key(
            experiment_id="56774.0",
            run_no="1",
            search_no="7",
            label="labelnone",
        )
        == "56774_1_7_0"
    )
