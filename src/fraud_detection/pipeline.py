"""End-to-end training pipeline.

Wires the modules together into one honest run:

    load -> split -> scale (fit on train) -> tune+fit supervised -> isolation forest
         -> pick headline by validation PR-AUC -> calibrate on validation
         -> choose cost-optimal threshold on validation -> report once on test (+ bootstrap CIs)
         -> persist models, metrics.json, and figures.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless; the notebook overrides with its own inline backend

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import calibration, evaluation, plotting, threshold  # noqa: E402
from .config import Config  # noqa: E402
from .data import load_data, split_data  # noqa: E402
from .features import AmountScaler  # noqa: E402
from .models import (  # noqa: E402
    MODEL_LABELS,
    build_supervised_models,
    make_isolation_forest,
    scale_pos_weight,
)
from .tuning import tune  # noqa: E402


@dataclass
class RunResult:
    results: dict[str, dict[str, float]]
    headline: str
    headline_threshold: float
    headline_test_metrics_at_threshold: dict[str, float]
    brier: float
    smote_cv: dict[str, float]
    comparison: pd.DataFrame
    artifacts_dir: Path
    figures: dict[str, Path] = field(default_factory=dict)


def _ci_row(y_true, scores, cfg: Config, with_thresholded: bool) -> dict[str, float]:
    if with_thresholded:
        row = evaluation.metric_row(y_true, scores, threshold=0.5)
    else:
        row = {
            "pr_auc": float(evaluation.average_precision_score(y_true, scores)),
            "roc_auc": float(evaluation.roc_auc_score(y_true, scores)),
            "precision": float("nan"),
            "recall": float("nan"),
            "f1": float("nan"),
        }
    point, low, high = evaluation.bootstrap_pr_auc_ci(
        y_true, scores, cfg.bootstrap.n_resamples, cfg.bootstrap.ci, cfg.seed
    )
    row["pr_auc"] = point
    row["pr_auc_low"] = low
    row["pr_auc_high"] = high
    return row


def run(cfg: Config) -> RunResult:
    df = load_data(cfg)
    splits = split_data(df, cfg)

    scaler = AmountScaler(cfg.features.scale_columns).fit(splits.X_train)
    X_train = scaler.transform(splits.X_train)
    X_val = scaler.transform(splits.X_val)
    X_test = scaler.transform(splits.X_test)
    y_train, y_val, y_test = splits.y_train, splits.y_val, splits.y_test

    spw = scale_pos_weight(y_train)

    results: dict[str, dict[str, float]] = {}
    fitted: dict[str, Any] = {}
    val_scores: dict[str, np.ndarray] = {}
    test_scores: dict[str, np.ndarray] = {}

    for name, estimator in build_supervised_models(cfg, spw).items():
        model = tune(name, estimator, X_train, y_train, cfg)
        fitted[name] = model
        val_scores[name] = model.predict_proba(X_val)[:, 1]
        test_scores[name] = model.predict_proba(X_test)[:, 1]
        results[MODEL_LABELS[name]] = _ci_row(y_test, test_scores[name], cfg, with_thresholded=True)

    # Unsupervised contrast: learn "normal" only, score anomalies on the test set.
    iso = make_isolation_forest(cfg, contamination=float(y_train.mean()))
    iso.fit(X_train[y_train == 0])
    iso_test = -iso.score_samples(X_test)
    fitted["isolation_forest"] = iso
    results[MODEL_LABELS["isolation_forest"]] = _ci_row(
        y_test, iso_test, cfg, with_thresholded=False
    )

    # SMOTE-in-CV sanity report on training data (no leakage).
    smote_cv = _smote_cv_report(cfg, X_train, y_train)

    # Pick the headline model by validation PR-AUC (never by test).
    supervised = ["logreg", "logreg_smote", "lightgbm", "xgboost"]
    headline = max(
        supervised,
        key=lambda n: evaluation.average_precision_score(y_val, val_scores[n]),
    )

    # Calibrate the headline on validation, then choose a cost-optimal threshold on validation.
    calibrated = calibration.calibrate(fitted[headline], X_val, y_val, cfg)
    cal_val = calibrated.predict_proba(X_val)[:, 1]
    cal_test = calibrated.predict_proba(X_test)[:, 1]

    grid = threshold.threshold_grid(cfg)
    curve = threshold.cost_curve(y_val, cal_val, grid, cfg.cost.c_fn, cfg.cost.c_fp)
    best_t = curve.best_threshold
    brier = calibration.brier(y_test, cal_test)
    at_threshold = evaluation.metric_row(y_test, cal_test, threshold=best_t)

    comparison = evaluation.comparison_table(results)

    artifacts_dir = cfg.artifacts_path
    figures = _save_artifacts(
        cfg, artifacts_dir, fitted, scaler, calibrated, headline, best_t,
        y_test, test_scores, cal_test, curve, brier,
    )

    result = RunResult(
        results=results,
        headline=MODEL_LABELS[headline],
        headline_threshold=best_t,
        headline_test_metrics_at_threshold=at_threshold,
        brier=brier,
        smote_cv=smote_cv,
        comparison=comparison,
        artifacts_dir=artifacts_dir,
        figures=figures,
    )
    _write_metrics(artifacts_dir, result, headline_key=headline)
    return result


def _smote_cv_report(cfg: Config, X_train, y_train) -> dict[str, float]:
    from sklearn.model_selection import StratifiedKFold, cross_validate

    from .models import make_logreg_smote

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=cfg.seed)
    res = cross_validate(
        make_logreg_smote(cfg), X_train, y_train, cv=cv,
        scoring=["average_precision", "roc_auc"], n_jobs=-1,
    )
    return {
        "pr_auc_mean": float(res["test_average_precision"].mean()),
        "pr_auc_std": float(res["test_average_precision"].std()),
        "roc_auc_mean": float(res["test_roc_auc"].mean()),
        "roc_auc_std": float(res["test_roc_auc"].std()),
    }


def _save_artifacts(
    cfg, artifacts_dir, fitted, scaler, calibrated, headline, best_t,
    y_test, test_scores, cal_test, curve, brier,
) -> dict[str, Path]:
    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = artifacts_dir / "figures"

    joblib.dump(
        {
            "scaler": scaler,
            "models": fitted,
            "calibrated_headline": calibrated,
            "headline": headline,
            "threshold": best_t,
            "feature_names": list(scaler.scaler.feature_names_in_)
            if hasattr(scaler.scaler, "feature_names_in_") else None,
        },
        artifacts_dir / "models.joblib",
    )

    figures: dict[str, Path] = {}
    figures["pr_roc"] = plotting.save_fig(
        plotting.plot_pr_roc(y_test, cal_test), fig_dir / "pr_roc.png"
    )
    figures["cost_curve"] = plotting.save_fig(
        plotting.plot_cost_curve(curve), fig_dir / "cost_curve.png"
    )
    frac_pos, mean_pred = calibration.reliability(y_test, cal_test)
    figures["reliability"] = plotting.save_fig(
        plotting.plot_reliability(frac_pos, mean_pred, brier), fig_dir / "reliability.png"
    )
    preds = (cal_test >= best_t).astype(int)
    figures["confusion"] = plotting.save_fig(
        plotting.plot_confusion(y_test, preds, f"{MODEL_LABELS[headline]} @ {best_t:.3f}"),
        fig_dir / "confusion.png",
    )
    importances = _feature_importance(fitted.get(headline))
    if importances is not None:
        figures["importance"] = plotting.save_fig(
            plotting.plot_feature_importance(importances), fig_dir / "importance.png"
        )
    return figures


def _feature_importance(model):
    if model is None or not hasattr(model, "feature_importances_"):
        return None
    names = getattr(model, "feature_names_in_", None)
    if names is None:
        names = [f"f{i}" for i in range(len(model.feature_importances_))]
    return pd.Series(model.feature_importances_, index=names)


def _write_metrics(artifacts_dir: Path, result: RunResult, headline_key: str) -> None:
    payload = {
        "headline": result.headline,
        "headline_threshold": result.headline_threshold,
        "headline_test_metrics_at_threshold": result.headline_test_metrics_at_threshold,
        "brier": result.brier,
        "smote_cv": result.smote_cv,
        "results": result.results,
    }
    out = Path(artifacts_dir) / "metrics.json"
    out.write_text(json.dumps(payload, indent=2, default=float))
