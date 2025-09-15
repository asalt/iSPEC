import numpy as np

from ispec.io import column_matching


def test_match_columns_without_transformer(monkeypatch):
    # Simulate environment without sentence_transformers by forcing get_default_model
    # to raise ImportError and ensuring score_matches is never called.

    def raise_import_error():  # pragma: no cover - behaviour validated in tests
        raise ImportError

    def fail(*args, **kwargs):
        raise AssertionError(
            "score_matches should not be called when transformer is unavailable"
        )

    monkeypatch.setattr(column_matching, "get_default_model", raise_import_error)
    monkeypatch.setattr(column_matching, "score_matches", fail)

    res = column_matching.match_columns(["foo"], ["foo", "bar"])
    assert res == {"foo": "foo"}


def test_match_columns_with_transformer(monkeypatch):
    # Simulate presence of transformer by providing a dummy model and scoring function
    called = {"flag": False}

    def fake_score(src_cols, tgt_cols, model=None):
        called["flag"] = True
        return np.array([[0.9]])

    monkeypatch.setattr(column_matching, "score_matches", fake_score)
    monkeypatch.setattr(column_matching, "get_default_model", lambda: object())

    res = column_matching.match_columns(["foo"], ["foo"])
    assert res == {"foo": "foo"}
    assert called["flag"]


def test_partial_matches_leave_unmatched(monkeypatch):
    # Simulate transformer producing scores where only the first column clears the threshold

    def fake_score(src_cols, tgt_cols, model=None):
        return np.array([[0.9, 0.1], [0.5, 0.4]])

    monkeypatch.setattr(column_matching, "score_matches", fake_score)
    monkeypatch.setattr(column_matching, "get_default_model", lambda: object())

    res = column_matching.match_columns(
        ["good_match", "partial"],
        ["good_match_target", "other_target"],
        threshold=0.6,
        fallback=False,
    )

    assert res == {"good_match": "good_match_target", "partial": None}


def test_score_matches_called_with_multiple_columns(monkeypatch):
    called = {"count": 0, "args": None}

    def fake_score(src_cols, tgt_cols, model=None):
        called["count"] += 1
        called["args"] = (src_cols, tgt_cols)
        return np.array([[0.9, 0.1], [0.2, 0.8]])

    monkeypatch.setattr(column_matching, "score_matches", fake_score)
    monkeypatch.setattr(column_matching, "get_default_model", lambda: object())

    res = column_matching.match_columns(["a", "b"], ["a", "b"])

    assert res == {"a": "a", "b": "b"}
    assert called["count"] == 1
    assert called["args"] == (["a", "b"], ["a", "b"])
