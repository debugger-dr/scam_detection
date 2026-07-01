"""Hyper-parameter search for the gradient-boosted models.

A small, fixed ``RandomizedSearchCV`` over a stratified CV on the *training* split only,
optimising average precision (PR-AUC). Kept deliberately light so a full run stays in the
minutes, not hours. Non-tunable models are simply fit as-is.
"""

from __future__ import annotations

import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.base import clone

from .config import Config
from .models import TUNABLE_MODELS

SEARCH_SPACES: dict[str, dict] = {
    "xgboost": {
        "n_estimators": randint(200, 600),
        "max_depth": randint(3, 8),
        "learning_rate": loguniform(1e-2, 3e-1),
        "subsample": uniform(0.6, 0.4),          # 0.6 .. 1.0
        "colsample_bytree": uniform(0.6, 0.4),
        "min_child_weight": randint(1, 10),
    },
    # No subsample (row bagging) for LightGBM: it starves trees of the ~315 training frauds
    # and destabilises ranking. We search depth/leaves/learning rate/column subsampling instead.
    "lightgbm": {
        "n_estimators": randint(400, 1000),
        "num_leaves": randint(15, 64),
        "learning_rate": loguniform(1e-2, 1e-1),
        "colsample_bytree": uniform(0.7, 0.3),
        "min_child_samples": randint(10, 60),
    },
}


def tune(name: str, estimator, X: pd.DataFrame, y: pd.Series, cfg: Config):
    """Return a fitted estimator: tuned when eligible and enabled, else fit as-is.

    The returned estimator is always fit on ``(X, y)`` and ready to predict.
    """
    if not cfg.tuning.enabled or name not in TUNABLE_MODELS:
        fitted = clone(estimator)
        fitted.fit(X, y)
        return fitted

    from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

    # Parallelise across CV fits only; keep each estimator single-threaded so the two layers
    # of parallelism don't oversubscribe the CPU (nested n_jobs=-1 thrashes badly).
    base = clone(estimator)
    if "n_jobs" in base.get_params():
        base.set_params(n_jobs=1)

    cv = StratifiedKFold(n_splits=cfg.tuning.cv_folds, shuffle=True, random_state=cfg.seed)
    search = RandomizedSearchCV(
        base,
        SEARCH_SPACES[name],
        n_iter=cfg.tuning.n_iter,
        scoring="average_precision",
        cv=cv,
        random_state=cfg.seed,
        n_jobs=-1,
        refit=True,
    )
    search.fit(X, y)
    best = search.best_estimator_
    if "n_jobs" in best.get_params():
        best.set_params(n_jobs=-1)  # restore full parallelism for inference/refit
    return best
