"""Plotting helpers.

Each function builds and returns a matplotlib Figure so the caller decides whether to show it
(notebook) or save it (headless pipeline). The pipeline selects a non-interactive backend
before importing this module.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from .threshold import ThresholdCurve


def save_fig(fig, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    return path


def plot_pr_roc(y_true, scores):
    import matplotlib.pyplot as plt

    prec, rec, _ = precision_recall_curve(y_true, scores)
    fpr, tpr, _ = roc_curve(y_true, scores)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].plot(rec, prec, color="#C44E52")
    ax[0].set_xlabel("Recall")
    ax[0].set_ylabel("Precision")
    ax[0].set_title(f"Precision-Recall  (AP = {average_precision_score(y_true, scores):.3f})")
    ax[1].plot(fpr, tpr, color="#4C72B0")
    ax[1].plot([0, 1], [0, 1], "--", color="gray")
    ax[1].set_xlabel("False Positive Rate")
    ax[1].set_ylabel("True Positive Rate")
    ax[1].set_title(f"ROC  (AUC = {roc_auc_score(y_true, scores):.3f})")
    fig.tight_layout()
    return fig


def plot_cost_curve(curve: ThresholdCurve, default_threshold: float = 0.5):
    import matplotlib.pyplot as plt

    fig, ax1 = plt.subplots(figsize=(9, 4.5))
    ax1.plot(curve.thresholds, curve.costs, color="black", label="expected cost")
    ax1.axvline(curve.best_threshold, color="#C44E52", ls="--",
                label=f"cost-opt = {curve.best_threshold:.3f}")
    ax1.axvline(default_threshold, color="gray", ls=":",
                label=f"default = {default_threshold:.2f}")
    ax1.set_xlabel("decision threshold")
    ax1.set_ylabel("expected cost")
    ax2 = ax1.twinx()
    ax2.plot(curve.thresholds, curve.recalls, color="#4C72B0", alpha=0.6, label="recall")
    ax2.plot(curve.thresholds, curve.precisions, color="#55A868", alpha=0.6, label="precision")
    ax2.set_ylabel("recall / precision")
    ax1.legend(loc="center right")
    ax2.legend(loc="upper right")
    ax1.set_title("Cost-based threshold selection")
    fig.tight_layout()
    return fig


def plot_reliability(frac_pos, mean_pred, brier: float | None = None):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="perfectly calibrated")
    ax.plot(mean_pred, frac_pos, "o-", color="#4C72B0", label="model")
    ax.set_xlabel("mean predicted probability")
    ax.set_ylabel("observed fraction of frauds")
    title = "Reliability curve"
    if brier is not None:
        title += f"  (Brier = {brier:.5f})"
    ax.set_title(title)
    ax.legend(loc="upper left")
    fig.tight_layout()
    return fig


def plot_feature_importance(importances: pd.Series, top: int = 15):
    import matplotlib.pyplot as plt

    top_imp = importances.sort_values(ascending=False).head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(top_imp.index, top_imp.values, color="#4C72B0")
    ax.set_xlabel("importance")
    ax.set_title(f"Top {top} features")
    fig.tight_layout()
    return fig


def plot_confusion(y_true, preds, title: str = "Confusion matrix"):
    import matplotlib.pyplot as plt

    cm = confusion_matrix(y_true, preds, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], labels=["legit", "fraud"])
    ax.set_yticks([0, 1], labels=["legit", "fraud"])
    ax.set_xlabel("predicted")
    ax.set_ylabel("actual")
    ax.set_title(title)
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, f"{v:,}", ha="center", va="center",
                color="white" if v > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig
