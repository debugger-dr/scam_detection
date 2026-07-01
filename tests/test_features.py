"""The scaler must fit on training data only -- no peeking at validation or test."""

import numpy as np
import pandas as pd
import pytest

from fraud_detection.features import AmountScaler


def test_scaler_standardizes_training_amount():
    X_train = pd.DataFrame({"Amount": [10.0, 20.0, 30.0, 40.0], "V1": [0, 0, 0, 0]})
    scaler = AmountScaler(["Amount"]).fit(X_train)
    out = scaler.transform(X_train)
    assert np.isclose(out["Amount"].mean(), 0.0, atol=1e-9)
    assert np.isclose(out["Amount"].std(ddof=0), 1.0, atol=1e-9)


def test_scaler_fits_on_train_only():
    X_train = pd.DataFrame({"Amount": [10.0, 20.0, 30.0, 40.0]})
    X_test = pd.DataFrame({"Amount": [1000.0, 2000.0]})
    scaler = AmountScaler(["Amount"]).fit(X_train)
    # The learned mean must come from train, not test.
    assert np.isclose(scaler.scaler.mean_[0], X_train["Amount"].mean())
    # Test data transformed with train stats is far from zero-mean.
    transformed_test = scaler.transform(X_test)
    assert transformed_test["Amount"].mean() > 10


def test_scaler_does_not_mutate_input():
    X = pd.DataFrame({"Amount": [10.0, 20.0, 30.0]})
    original = X.copy()
    AmountScaler(["Amount"]).fit_transform(X)
    pd.testing.assert_frame_equal(X, original)


def test_transform_before_fit_raises():
    with pytest.raises(RuntimeError):
        AmountScaler(["Amount"]).transform(pd.DataFrame({"Amount": [1.0]}))
