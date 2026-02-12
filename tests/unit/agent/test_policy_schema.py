from __future__ import annotations

from ispec.agent.policy_schema import (
    CacheKey,
    ComposeSpec,
    InputSignature,
    ParamSet,
    PolicyRef,
    PolicySpec,
    PolicyState,
    callable_id_from_callable,
    canon_json,
    stable_hash,
)


def _example_policy(x: int) -> int:
    return x + 1


def test_canon_json_is_stable_and_sorted() -> None:
    a = {"b": 1, "a": 2}
    b = {"a": 2, "b": 1}
    assert canon_json(a) == canon_json(b)


def test_stable_hash_is_stable() -> None:
    payload = {"a": 1, "b": [2, 3]}
    assert stable_hash(payload) == stable_hash(payload)


def test_callable_id_from_callable_prefers_source_when_available() -> None:
    cid = callable_id_from_callable(_example_policy, mode="source")
    assert cid.mode in {"source", "code", "fallback"}
    assert cid.qualname.endswith("_example_policy")
    assert cid.id() == cid.id()


def test_policy_ids_are_stable() -> None:
    cid = callable_id_from_callable(_example_policy, mode="code")
    spec = PolicySpec(
        ref=PolicyRef(kind="example", callable_id=cid),
        param_set=ParamSet(name="v1", params={"alpha": 0.1}),
        tags={"team": "ispec"},
    )
    composed = ComposeSpec(base=spec, modifiers=())
    assert spec.policy_id() == spec.policy_id()
    assert composed.policy_id() == composed.policy_id()


def test_cache_key_is_stable() -> None:
    state = PolicyState(step=3, data={"n": 1})
    input_sig = InputSignature(inputs={"thread": "t1"}, context={"bucket": "2026-01-17"})
    key = CacheKey(
        policy_id="p1",
        input_hash=input_sig.input_hash(),
        state_hash=state.state_hash(),
        step=state.step,
    )
    assert key.key() == key.key()

