import random

from ispec.cli.agent import BackoffPolicy, _next_interval_seconds


def test_backoff_advances_after_three_idle_rounds():
    policy = BackoffPolicy(
        levels_seconds=[30, 60, 120, 180],
        idle_rounds_per_level=3,
        jitter_fraction=0.0,
        random_probe_probability=0.0,
    )
    rng = random.Random(0)

    idle = 0
    interval, idle, reason = _next_interval_seconds(policy=policy, idle_rounds=idle, activity=False, rng=rng)
    assert (interval, idle, reason) == (30, 1, "idle_level_0")

    interval, idle, reason = _next_interval_seconds(policy=policy, idle_rounds=idle, activity=False, rng=rng)
    assert (interval, idle, reason) == (30, 2, "idle_level_0")

    interval, idle, reason = _next_interval_seconds(policy=policy, idle_rounds=idle, activity=False, rng=rng)
    assert (interval, idle, reason) == (30, 3, "idle_level_0")

    interval, idle, reason = _next_interval_seconds(policy=policy, idle_rounds=idle, activity=False, rng=rng)
    assert (interval, idle, reason) == (60, 4, "idle_level_1")


def test_backoff_resets_on_activity():
    policy = BackoffPolicy(levels_seconds=[30, 60], idle_rounds_per_level=3, jitter_fraction=0.0)
    rng = random.Random(0)
    interval, idle, reason = _next_interval_seconds(policy=policy, idle_rounds=99, activity=True, rng=rng)
    assert (interval, idle, reason) == (30, 0, "activity")


def test_backoff_random_probe_can_shortcut_when_idle():
    policy = BackoffPolicy(
        levels_seconds=[30, 60, 120],
        idle_rounds_per_level=3,
        jitter_fraction=0.0,
        random_probe_probability=1.0,
        random_probe_min_seconds=5,
        random_probe_max_seconds=6,
    )
    rng = random.Random(0)
    interval, idle, reason = _next_interval_seconds(policy=policy, idle_rounds=3, activity=False, rng=rng)
    assert interval in {5, 6}
    assert idle == 4
    assert reason.endswith(":probe")
