"""Cost-based threshold selection must respect the asymmetric FN/FP cost."""

import numpy as np

from fraud_detection.threshold import (
    cost_curve,
    cost_optimal_threshold,
    expected_cost,
)


def test_expected_cost_counts_errors():
    y = [0, 0, 1, 1]
    scores = [0.1, 0.4, 0.6, 0.9]
    # Perfect separation at 0.5.
    assert expected_cost(y, scores, 0.5, c_fn=100, c_fp=1) == 0.0
    # Threshold too high: both frauds missed.
    assert expected_cost(y, scores, 0.95, c_fn=100, c_fp=1) == 200.0
    # Threshold too low: both legit flagged.
    assert expected_cost(y, scores, 0.05, c_fn=100, c_fp=1) == 2.0


def test_optimal_threshold_separates_clean_data():
    y = [0, 0, 1, 1]
    scores = [0.1, 0.4, 0.6, 0.9]
    grid = np.linspace(0.0, 1.0, 101)
    curve = cost_curve(y, scores, grid, c_fn=100, c_fp=1)
    assert curve.best_cost == 0.0
    assert 0.4 < curve.best_threshold <= 0.6


def test_high_fn_cost_pushes_threshold_down():
    # The 0.50 sample is fraud; the 0.55 sample is legit (overlap).
    y = [0, 0, 1, 1]
    scores = [0.2, 0.55, 0.50, 0.9]
    grid = np.linspace(0.0, 1.0, 101)
    # With a heavy false-negative cost, the optimum catches the 0.50 fraud (threshold below it),
    # accepting the false alarm on the 0.55 legit transaction.
    t = cost_optimal_threshold(y, scores, c_fn=100, c_fp=1, grid=grid)
    assert t <= 0.50
    assert expected_cost(y, scores, t, c_fn=100, c_fp=1) == 1.0


def test_curve_arrays_are_aligned():
    y = [0, 1, 0, 1]
    scores = [0.1, 0.8, 0.3, 0.6]
    grid = np.linspace(0.0, 1.0, 50)
    curve = cost_curve(y, scores, grid, c_fn=10, c_fp=1)
    assert len(curve.costs) == len(curve.recalls) == len(curve.precisions) == len(grid)
    assert (curve.recalls >= 0).all() and (curve.recalls <= 1).all()
