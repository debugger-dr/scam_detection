"""Model factory.

Five models, each handling the 0.17% imbalance a different way:

* ``logreg``        - Logistic Regression with ``class_weight="balanced"`` (cost-sensitive).
* ``logreg_smote``  - Logistic Regression with SMOTE applied *inside* CV (no leakage).
* ``lightgbm``      - LightGBM with ``scale_pos_weight`` (replaces the old Random Forest).
* ``xgboost``       - XGBoost with ``scale_pos_weight`` (the headline model).
* ``isolation_forest`` - unsupervised contrast, trained on legitimate transactions only.
"""

from __future__ import annotations

import pandas as pd

from .config import Config

# Supervised models that produce calibratable probabilities via predict_proba.
SUPERVISED_MODELS = ("logreg", "logreg_smote", "lightgbm", "xgboost")
# Models eligible for randomized hyper-parameter search.
TUNABLE_MODELS = ("lightgbm", "xgboost")

# Human-readable labels for reports and plots.
MODEL_LABELS = {
    "logreg": "LogReg (class_weight)",
    "logreg_smote": "LogReg + SMOTE (in-CV)",
    "lightgbm": "LightGBM",
    "xgboost": "XGBoost (scale_pos_weight)",
    "isolation_forest": "Isolation Forest (unsup.)",
}


def scale_pos_weight(y: pd.Series) -> float:
    """Negative/positive ratio used to reweight the rare positive class in the GBMs."""
    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    if pos == 0:
        raise ValueError("No positive (fraud) samples in y; cannot compute scale_pos_weight")
    return neg / pos


def make_logreg(cfg: Config):
    from sklearn.linear_model import LogisticRegression

    params = cfg.models.get("logreg", {})
    return LogisticRegression(
        max_iter=params.get("max_iter", 1000),
        class_weight="balanced",
        random_state=cfg.seed,
    )


def make_logreg_smote(cfg: Config):
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    from sklearn.linear_model import LogisticRegression

    params = cfg.models.get("logreg", {})
    return ImbPipeline(
        [
            ("smote", SMOTE(random_state=cfg.seed)),
            ("clf", LogisticRegression(max_iter=params.get("max_iter", 1000),
                                       random_state=cfg.seed)),
        ]
    )


def make_lightgbm(cfg: Config):
    """LightGBM classifier.

    Unlike XGBoost, LightGBM's leaf-wise growth is destabilised by the extreme
    ``scale_pos_weight`` (~578) this dataset implies: heavy positive weighting wrecks its
    high-precision ranking (PR-AUC collapses while ROC-AUC stays high). It ranks best with
    default weighting, so ``scale_pos_weight`` defaults to 1 here and is only applied if a
    value is set in config. The right imbalance handling is model-specific.
    """
    from lightgbm import LGBMClassifier

    params = dict(cfg.models.get("lightgbm", {}))
    params.setdefault("scale_pos_weight", 1.0)
    return LGBMClassifier(
        n_jobs=-1,
        random_state=cfg.seed,
        verbose=-1,
        **params,
    )


def make_xgboost(cfg: Config, spw: float):
    from xgboost import XGBClassifier

    params = dict(cfg.models.get("xgboost", {}))
    return XGBClassifier(
        scale_pos_weight=spw,
        eval_metric="aucpr",
        n_jobs=-1,
        random_state=cfg.seed,
        **params,
    )


def make_isolation_forest(cfg: Config, contamination: float):
    from sklearn.ensemble import IsolationForest

    params = dict(cfg.models.get("isolation_forest", {}))
    return IsolationForest(
        contamination=contamination,
        random_state=cfg.seed,
        n_jobs=-1,
        **params,
    )


def build_supervised_models(cfg: Config, spw: float) -> dict[str, object]:
    """Return the four supervised estimators keyed by name (unfitted)."""
    return {
        "logreg": make_logreg(cfg),
        "logreg_smote": make_logreg_smote(cfg),
        "lightgbm": make_lightgbm(cfg),
        "xgboost": make_xgboost(cfg, spw),
    }
