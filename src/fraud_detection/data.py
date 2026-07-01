"""Dataset fetching, loading, and the stratified three-way split.

The split is the methodological backbone of the project: train (fit + tune), validation
(calibration + threshold selection), test (the single honest report). Keeping it here, in one
seeded function, is what prevents the test set from leaking into any decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .config import Config

TARGET = "Class"


@dataclass
class Splits:
    """A stratified train/validation/test split of features and target."""

    X_train: pd.DataFrame
    y_train: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series

    def summary(self) -> str:
        def line(name: str, y: pd.Series) -> str:
            return f"{name:5}: {len(y):>7,} rows, {int(y.sum()):>4} frauds ({100 * y.mean():.3f}%)"

        return "\n".join(
            [line("Train", self.y_train), line("Val", self.y_val), line("Test", self.y_test)]
        )


def fetch_dataset(cfg: Config, force: bool = False) -> Path:
    """Download the ULB credit-card dataset from OpenML into ``cfg.data.raw_csv``.

    No Kaggle login required. Returns the path to the CSV. Skips the download if the file
    already exists unless ``force`` is set.
    """
    out = Path(cfg.data.raw_csv)
    if out.exists() and not force:
        return out

    from sklearn.datasets import fetch_openml

    frame = fetch_openml(
        cfg.data.openml_name, version=cfg.data.openml_version, as_frame=True
    ).frame
    frame[TARGET] = frame[TARGET].astype(int)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    return out


def load_data(cfg: Config) -> pd.DataFrame:
    """Load the dataset CSV into a DataFrame with an integer ``Class`` column."""
    path = Path(cfg.data.raw_csv)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Run `fraud-detection fetch-data` first."
        )
    df = pd.read_csv(path)
    df[TARGET] = df[TARGET].astype(int)
    return df


def split_data(df: pd.DataFrame, cfg: Config) -> Splits:
    """Stratified train/validation/test split.

    ``test_size`` and ``val_size`` are fractions of the *full* dataset; the remainder is the
    training set. Stratification on ``Class`` preserves the ~0.17% fraud rate in every split.
    """
    from sklearn.model_selection import train_test_split

    X = df.drop(columns=TARGET)
    y = df[TARGET]

    test_size = cfg.split.test_size
    val_size = cfg.split.val_size

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=cfg.seed
    )
    # val_size is a fraction of the full set; convert to a fraction of the remaining temp set.
    val_relative = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_relative, stratify=y_temp, random_state=cfg.seed
    )
    return Splits(X_train, y_train, X_val, y_val, X_test, y_test)
