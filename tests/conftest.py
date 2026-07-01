"""Shared fixtures: a small synthetic, imbalanced dataset shaped like the real one."""

import numpy as np
import pandas as pd
import pytest

from fraud_detection.config import Config


@pytest.fixture
def cfg() -> Config:
    return Config()  # defaults: seed=42, test_size=0.2, val_size=0.2


@pytest.fixture
def toy_df() -> pd.DataFrame:
    """1,000 rows, ~5% fraud, columns V1..V5 + Amount + Class (mirrors the ULB schema)."""
    rng = np.random.default_rng(0)
    n = 1000
    n_fraud = 50
    y = np.zeros(n, dtype=int)
    y[:n_fraud] = 1
    rng.shuffle(y)

    data = {f"V{i}": rng.normal(size=n) for i in range(1, 6)}
    # Give fraud a slightly different signal so models are learnable.
    for i in range(1, 6):
        data[f"V{i}"] += y * rng.normal(0.8, 0.1)
    data["Amount"] = rng.gamma(2.0, 50.0, size=n)
    data["Class"] = y
    return pd.DataFrame(data)
