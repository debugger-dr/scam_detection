"""Metrics, bootstrap confidence intervals, and the model comparison table.

PR-AUC (average precision) is the headline metric because ROC-AUC stays optimistically high
even for useless models on a 0.17%-positive problem. Bootstrap CIs on PR-AUC turn "model A
beats model B" into a claim that can actually be checked for overlap.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

METRIC_COLUMNS = ["pr_auc", "roc_auc", "precision", "recall", "f1"]


def metric_row(y_true, scores, threshold: float = 0.5) -> dict[str, float]:
    """Threshold-independent (PR-AUC, ROC-AUC) and thresholded (P/R/F1) metrics."""
    scores = np.asarray(scores)
    preds = (scores >= threshold).astype(int)
    return {
        "pr_auc": float(average_precision_score(y_true, scores)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
        "f1": float(f1_score(y_true, preds, zero_division=0)),
    }


def bootstrap_pr_auc_ci(
    y_true, scores, n_resamples: int = 1000, ci: float = 0.95, seed: int = 42
) -> tuple[float, float, float]:
    """Bootstrap percentile CI for PR-AUC.

    Returns ``(point_estimate, low, high)``. Resamples that happen to contain no positives are
    skipped, since average precision is undefined without a positive class.
    """
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    n = len(y_true)
    rng = np.random.default_rng(seed)

    point = float(average_precision_score(y_true, scores))
    estimates: list[float] = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        y_b = y_true[idx]
        if y_b.sum() == 0:
            continue
        estimates.append(average_precision_score(y_b, scores[idx]))

    if not estimates:
        return point, float("nan"), float("nan")

    alpha = (1.0 - ci) / 2.0
    low, high = np.quantile(estimates, [alpha, 1.0 - alpha])
    return point, float(low), float(high)


def comparison_table(results: dict[str, dict[str, float]]) -> pd.DataFrame:
    """Build a sorted comparison DataFrame from a name -> metric-row mapping."""
    available = [c for c in METRIC_COLUMNS if any(c in row for row in results.values())]
    extra = ["pr_auc_low", "pr_auc_high"]
    cols = available + [c for c in extra if any(c in row for row in results.values())]
    df = pd.DataFrame(results).T
    df = df.reindex(columns=[c for c in cols if c in df.columns])
    return df.sort_values("pr_auc", ascending=False)
