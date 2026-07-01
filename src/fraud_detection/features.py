"""Feature preprocessing.

The ULB features ``V1``..``V28`` are already PCA components (decorrelated and scaled); only
``Amount`` needs rescaling. The scaler is fit on the training split *only* so that no
statistic from validation or test leaks backward into preprocessing.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
from sklearn.preprocessing import StandardScaler


class AmountScaler:
    """Standard-scale a subset of columns, fitting on training data only."""

    def __init__(self, columns: Sequence[str]) -> None:
        self.columns = list(columns)
        self.scaler = StandardScaler()
        self._fitted = False

    def fit(self, X: pd.DataFrame) -> "AmountScaler":
        self.scaler.fit(X[self.columns])
        self._fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("AmountScaler.transform called before fit")
        X = X.copy()
        X[self.columns] = self.scaler.transform(X[self.columns])
        return X

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X).transform(X)
