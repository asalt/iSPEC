import numpy as np

from ispec.io import column_matching


def test_match_columns_without_transformer(monkeypatch):
    # Simulate environment without sentence_transformers
    monkeypatch.setattr(column_matching, "_default_model", None)

    def fail(*args, **kwargs):
        raise AssertionError("score_matches should not be called when transformer is unavailable")

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
    monkeypatch.setattr(column_matching, "_default_model", object())

    res = column_matching.match_columns(["foo"], ["foo"])
    assert res == {"foo": "foo"}
    assert called["flag"]
