"""Credit-card fraud detection on highly imbalanced data.

A small, defensible pipeline: stratified train/validation/test split, cost-sensitive and
SMOTE-in-CV models, probability calibration, validation-chosen decision thresholds, and
bootstrap confidence intervals on PR-AUC.
"""

from .config import Config

__all__ = ["Config", "__version__"]
__version__ = "0.1.0"
