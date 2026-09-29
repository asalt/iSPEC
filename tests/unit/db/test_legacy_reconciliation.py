from datetime import UTC, datetime
import json

import pytest

from ispec.db.legacy_reconciliation import reconcile_source_fields


SOURCE = "http://legacy.example/api/v2/legacy/tables/iSPEC_ExperimentRuns/rows"


def snapshot(values):
    return json.dumps({"version": 1, "source": SOURCE, "values": values})


def reconcile(local, incoming, baseline=None, conflicts=None, **kwargs):
    return reconcile_source_fields(
        current=local,
        incoming=incoming,
        baseline_json=snapshot(baseline) if baseline is not None else None,
        conflicts_json=conflicts,
        source=SOURCE,
        **kwargs,
    )


@pytest.mark.parametrize(
    "previous,local,incoming,updates,conflicted",
    [
        ("A", "A", "B", {"field": "B"}, False),
        ("A", "B", "A", {}, False),
        ("A", "B", "B", {}, False),
        ("A", "B", "C", {}, True),
        ("A", "A", "A", {}, False),
        ("A", "A", None, {"field": None}, False),
        ("A", None, "A", {}, False),
        (None, None, "B", {"field": "B"}, False),
        (None, "B", None, {}, False),
    ],
)
def test_three_way_rules(previous, local, incoming, updates, conflicted):
    result = reconcile({"field": local}, {"field": incoming}, {"field": previous})
    assert result.updates == updates
    assert bool(result.conflict_fields) is conflicted
    assert json.loads(result.baseline_json)["values"] == {"field": incoming}


def test_divergent_conflict_survives_repeat_and_resolves_on_agreement():
    first = reconcile({"field": "local"}, {"field": "remote"}, {"field": "base"})
    second = reconcile(
        {"field": "local"},
        {"field": "remote"},
        {"field": "remote"},
        conflicts=first.conflicts_json,
    )
    assert second.conflicts_json == first.conflicts_json
    evidence = json.loads(second.conflicts_json)["values"]["field"]
    assert evidence == {
        "reason": "divergent_changes",
        "previous_source_known": True,
        "previous_source": "base",
        "local": "local",
        "incoming_source": "remote",
    }
    agreed = reconcile(
        {"field": "remote"},
        {"field": "remote"},
        {"field": "remote"},
        conflicts=second.conflicts_json,
    )
    assert agreed.conflict_fields == ()


def test_missing_field_preserves_baseline_and_conflict():
    first = reconcile({"field": "local"}, {"field": "remote"}, {"field": "base"})
    result = reconcile(
        {"field": "local", "other": 1},
        {"other": 2},
        {"field": "remote", "other": 1},
        conflicts=first.conflicts_json,
    )
    assert result.updates == {"other": 2}
    assert json.loads(result.baseline_json)["values"]["field"] == "remote"
    assert result.conflicts_json == first.conflicts_json


@pytest.mark.parametrize(
    "baseline",
    [
        None,
        "{bad",
        "[]",
        '{"version": 2}',
        snapshot({"field": "local"}).replace("legacy.example", "another-source"),
    ],
)
def test_untrusted_baseline_never_overwrites_nonempty_local_values(baseline):
    result = reconcile_source_fields(
        current={"field": "local"},
        incoming={"field": "incoming"},
        baseline_json=baseline,
        conflicts_json=None,
        source=SOURCE,
        backfill_missing=True,
    )
    assert not result.updates
    assert result.conflict_fields == ("field",)
    assert (
        json.loads(result.conflicts_json)["values"]["field"]["reason"]
        == "unknown_baseline"
    )


@pytest.mark.parametrize("backfill", [False, True])
def test_unknown_baseline_backfill_requires_explicit_option(backfill):
    result = reconcile({"field": None}, {"field": "remote"}, backfill_missing=backfill)
    assert result.updates == ({"field": "remote"} if backfill else {})
    assert bool(result.conflict_fields) is not backfill


@pytest.mark.parametrize("local", [None, ""])
def test_backfill_does_not_restore_a_known_local_deletion(local):
    result = reconcile(
        {"field": local},
        {"field": "source"},
        {"field": "source"},
        backfill_missing=True,
    )
    assert not result.updates
    assert not result.conflict_fields


def test_unknown_baseline_can_be_established_by_agreement():
    first = reconcile({"field": "A"}, {"field": "A"})
    assert not first.conflict_fields
    second = reconcile({"field": "A"}, {"field": "B"}, {"field": "A"})
    assert second.updates == {"field": "B"}


def test_datetime_baseline_compares_normalized_values():
    instant = datetime(2026, 1, 1, tzinfo=UTC)
    first = reconcile({"field": instant}, {"field": instant})
    result = reconcile_source_fields(
        current={"field": instant.replace(tzinfo=None)},
        incoming={"field": datetime(2026, 2, 1)},
        baseline_json=first.baseline_json,
        conflicts_json=first.conflicts_json,
        source=SOURCE,
    )
    assert result.updates == {"field": datetime(2026, 2, 1)}
    assert not result.conflict_fields
