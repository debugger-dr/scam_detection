"""Probability calibration and calibration diagnostics.

GBMs trained with ``scale_pos_weight`` emit distorted probabilities, which matters because the
decision threshold is chosen against a *cost* expressed in those probabilities. We calibrate
the prefit headline model on the validation split and report a reliability curve + Brier score.
"""

from __future__ import annotations

import numpy as np
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss

from .config import Config


def _freeze(estimator):
    """Wrap an already-fitted estimator so CalibratedClassifierCV won't refit it.

    Uses ``FrozenEstimator`` (scikit-learn >= 1.6); falls back to the legacy ``cv="prefit"``.
    """
    try:
        from sklearn.frozen import FrozenEstimator

        return FrozenEstimator(estimator), {}
    except ImportError:  # pragma: no cover - depends on sklearn version
        return estimator, {"cv": "prefit"}


def calibrate(estimator, X_val, y_val, cfg: Config):
    """Return a calibrated classifier fit on the validation split."""
    base, extra = _freeze(estimator)
    calibrated = CalibratedClassifierCV(base, method=cfg.calibration.method, **extra)
    calibrated.fit(X_val, y_val)
    return calibrated


def reliability(y_true, prob, n_bins: int = 10):
    """Return ``(fraction_of_positives, mean_predicted_value)`` for a reliability curve."""
    return calibration_curve(y_true, prob, n_bins=n_bins, strategy="quantile")


def brier(y_true, prob) -> float:
    """Brier score (lower is better); a proper scoring rule for calibrated probabilities."""
    return float(brier_score_loss(y_true, np.asarray(prob)))
