"""Metrics and bootstrap CIs must match sklearn and bracket the point estimate."""

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from fraud_detection.evaluation import (
    bootstrap_pr_auc_ci,
    comparison_table,
    metric_row,
)


def _scores():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, size=400)
    # Scores correlated with y so the metrics are non-degenerate.
    scores = 0.3 * rng.random(400) + 0.5 * y
    return y, scores


def test_metric_row_matches_sklearn():
    y, scores = _scores()
    row = metric_row(y, scores, threshold=0.5)
    assert np.isclose(row["pr_auc"], average_precision_score(y, scores))
    assert np.isclose(row["roc_auc"], roc_auc_score(y, scores))
    assert set(row) == {"pr_auc", "roc_auc", "precision", "recall", "f1"}


def test_bootstrap_ci_brackets_point_estimate():
    y, scores = _scores()
    point, low, high = bootstrap_pr_auc_ci(y, scores, n_resamples=300, ci=0.95, seed=42)
    assert np.isclose(point, average_precision_score(y, scores))
    assert low <= point <= high
    assert 0.0 <= low <= high <= 1.0


def test_bootstrap_ci_is_reproducible():
    y, scores = _scores()
    a = bootstrap_pr_auc_ci(y, scores, n_resamples=200, seed=7)
    b = bootstrap_pr_auc_ci(y, scores, n_resamples=200, seed=7)
    assert a == b


def test_comparison_table_sorts_by_pr_auc():
    results = {
        "A": {"pr_auc": 0.5, "roc_auc": 0.9, "precision": 0.1, "recall": 0.9, "f1": 0.2},
        "B": {"pr_auc": 0.8, "roc_auc": 0.95, "precision": 0.8, "recall": 0.7, "f1": 0.75},
    }
    table = comparison_table(results)
    assert list(table.index) == ["B", "A"]
    assert "pr_auc" in table.columns
