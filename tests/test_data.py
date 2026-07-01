"""The split is the methodological backbone: it must not leak and must stay stratified."""

import numpy as np

from fraud_detection.data import TARGET, split_data


def test_split_sizes_sum_to_total(toy_df, cfg):
    s = split_data(toy_df, cfg)
    total = len(toy_df)
    assert len(s.y_train) + len(s.y_val) + len(s.y_test) == total
    # Roughly 60 / 20 / 20.
    assert abs(len(s.y_test) - 0.2 * total) <= 1
    assert abs(len(s.y_val) - 0.2 * total) <= 1
    assert abs(len(s.y_train) - 0.6 * total) <= 1


def test_split_has_no_overlap(toy_df, cfg):
    s = split_data(toy_df, cfg)
    train_idx = set(s.X_train.index)
    val_idx = set(s.X_val.index)
    test_idx = set(s.X_test.index)
    assert train_idx.isdisjoint(val_idx)
    assert train_idx.isdisjoint(test_idx)
    assert val_idx.isdisjoint(test_idx)
    assert len(train_idx | val_idx | test_idx) == len(toy_df)


def test_split_preserves_fraud_rate(toy_df, cfg):
    overall = toy_df[TARGET].mean()
    s = split_data(toy_df, cfg)
    for y in (s.y_train, s.y_val, s.y_test):
        assert np.isclose(y.mean(), overall, atol=0.02)


def test_split_is_deterministic(toy_df, cfg):
    a = split_data(toy_df, cfg)
    b = split_data(toy_df, cfg)
    assert list(a.X_test.index) == list(b.X_test.index)
