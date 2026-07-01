"""Cost-based decision-threshold selection.

A missed fraud (false negative) costs far more than a false alarm (false positive). Rather
than defaulting to 0.5, we sweep thresholds and pick the one that minimises expected cost
``C_FN * FN + C_FP * FP`` -- and we do this on the *validation* split, never on test.

All functions here are pure and deterministic, which is what makes them unit-testable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import confusion_matrix

from .config import Config


@dataclass
class ThresholdCurve:
    thresholds: np.ndarray
    costs: np.ndarray
    recalls: np.ndarray
    precisions: np.ndarray
    best_threshold: float
    best_cost: float


def expected_cost(y_true, scores, threshold: float, c_fn: float, c_fp: float) -> float:
    """Expected cost of classifying at ``threshold`` given asymmetric error costs."""
    preds = (np.asarray(scores) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
    return float(c_fn * fn + c_fp * fp)


def threshold_grid(cfg: Config) -> np.ndarray:
    t = cfg.threshold
    return np.linspace(t.grid_low, t.grid_high, t.grid_points)


def cost_curve(y_true, scores, grid, c_fn: float, c_fp: float) -> ThresholdCurve:
    """Evaluate cost, recall, and precision across a grid of thresholds."""
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    grid = np.asarray(grid, dtype=float)

    costs, recalls, precisions = [], [], []
    for t in grid:
        tn, fp, fn, tp = confusion_matrix(
            y_true, (scores >= t).astype(int), labels=[0, 1]
        ).ravel()
        costs.append(c_fn * fn + c_fp * fp)
        recalls.append(tp / (tp + fn) if (tp + fn) else 0.0)
        precisions.append(tp / (tp + fp) if (tp + fp) else 0.0)

    costs = np.asarray(costs, dtype=float)
    best_idx = int(costs.argmin())
    return ThresholdCurve(
        thresholds=grid,
        costs=costs,
        recalls=np.asarray(recalls),
        precisions=np.asarray(precisions),
        best_threshold=float(grid[best_idx]),
        best_cost=float(costs[best_idx]),
    )


def cost_optimal_threshold(y_true, scores, c_fn: float, c_fp: float, grid) -> float:
    """Return the threshold on ``grid`` that minimises expected cost."""
    return cost_curve(y_true, scores, grid, c_fn, c_fp).best_threshold
