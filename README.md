# Credit-Card Fraud Detection on Highly Imbalanced Data

A machine-learning study of credit-card fraud detection that treats the **0.17% class
imbalance as the central problem**, not an afterthought. It reproduces a common
methodological mistake seen in many fraud-detection notebooks and shows what defensible
numbers actually look like.

> **TL;DR** — On the public ULB credit-card dataset, an XGBoost model with cost-sensitive
> weighting reaches **PR-AUC 0.88** on a held-out, genuinely imbalanced test set, catching
> **~84% of fraud at ~85% precision** — and up to **90% of fraud** once the decision
> threshold is tuned to a business cost. A popular shortcut (applying SMOTE before the
> train/test split) instead reports a fake **~0.99 on every metric**; this project shows why.

---

## Why this matters

Card fraud is a needle-in-a-haystack problem. In this dataset only **492 of 284,807
transactions (0.173%)** are fraudulent — roughly **1 in 578**. That extreme rarity breaks
the usual ML habits:

- **Accuracy is meaningless.** "Always predict legitimate" scores 99.83% accuracy and catches
  zero fraud. A useful metric must focus on the rare positive class.
- **The wrong validation lies to you.** The single most common mistake in fraud-detection
  notebooks is oversampling (SMOTE) the *entire* dataset and *then* splitting into
  train/test. Synthetic copies of test points leak into training, and the metrics become
  fiction.

This repository is a worked, end-to-end answer to both pitfalls.

## Two common approaches this addresses

Public fraud-detection notebooks tend to fall into two camps:

| Approach | Typical models | Imbalance handling | Observation |
|----------|----------------|--------------------|-------------|
| **Methodologically sound** | Logistic Regression, Decision Tree, Random Forest | SMOTE + random undersampling | Evaluates on the imbalanced test set with ROC-AUC — defensible, though ROC-AUC flatters rare-event performance. |
| **The leaky shortcut** | XGBoost | SMOTE | SMOTE applied to the **full set before the split**, producing reported scores of **~0.99 on every metric** — too good to be real. |

This project keeps the strengths of the first and uses the second as a cautionary baseline:
it reproduces the leak, then fixes the methodology.

## What this implementation does differently

1. **Reproduces the leakage** — applies SMOTE before the split and recovers the same inflated
   ~0.99 scores, then shows the honest numbers side by side.
2. **Handles imbalance correctly** — cost-sensitive learning (`class_weight` /
   `scale_pos_weight`) and SMOTE applied **strictly inside cross-validation folds** via an
   `imblearn` pipeline.
3. **Uses the right metric** — **PR-AUC (Average Precision)** as the headline number, not
   accuracy and not ROC-AUC (which stays optimistically high even for a useless model here).
4. **Tunes the decision threshold to a business cost** (a missed fraud costed at 100× a false
   alarm) instead of blindly using 0.5.
5. **Compares four models** — Logistic Regression, Random Forest, XGBoost, and an
   unsupervised Isolation Forest.

## Results (held-out, imbalanced test set — 98 frauds in 56,962 transactions)

| Model | PR-AUC | ROC-AUC | Precision | Recall | F1 |
|-------|:------:|:-------:|:---------:|:------:|:--:|
| **XGBoost** (`scale_pos_weight`) | **0.879** | 0.976 | 0.854 | 0.837 | 0.845 |
| Random Forest (`class_weight`) | 0.858 | 0.952 | 0.961 | 0.755 | 0.846 |
| LogReg + SMOTE (correct, in-CV) | 0.723 | 0.970 | 0.056 | 0.918 | 0.105 |
| LogReg (`class_weight`) | 0.720 | 0.971 | 0.059 | 0.918 | 0.111 |
| Isolation Forest (unsupervised) | 0.137 | 0.954 | — | — | — |

*Metrics at the default 0.5 threshold; PR-AUC / ROC-AUC are threshold-independent.*

**The leaky shortcut for comparison:** a plain logistic regression trained after
SMOTE-before-split scores **ROC-AUC 0.992, PR-AUC 0.993, recall 0.94** — on a test set that
has been synthetically inflated to 50% fraud. None of it transfers to reality.

### Three things the numbers prove

- **Honest SMOTE is not a magic boost.** Done correctly (inside CV), SMOTE + LogReg scores
  PR-AUC **0.723** — statistically the same as cost-weighted LogReg (**0.720**). The huge
  gains people report come from leakage, not from SMOTE.
- **ROC-AUC misleads on rare events.** *Every* model scores ROC-AUC ≥ 0.95 — including the
  Isolation Forest whose PR-AUC is a near-useless **0.137**. PR-AUC separates the real
  performers; ROC-AUC does not.
- **The threshold is a business lever.** Moving XGBoost's cut-off from 0.50 to the
  cost-optimal **0.006** lifts recall from **0.84 → 0.90** (fewer missed frauds) at the cost
  of precision (**0.85 → 0.47**) — a deliberate, quantified trade, not an accident.

## The dataset

**ULB Credit Card Fraud Detection** (Worldline & Université Libre de Bruxelles, 2013):
284,807 European card transactions over two days, 492 fraudulent.

- `V1`–`V28` — anonymised **PCA components** (already decorrelated and scaled).
- `Amount` — transaction value (the only feature this project rescales).
- `Class` — target: `1` = fraud, `0` = legitimate.

The CSV is **not committed** (≈148 MB). It is fetched reproducibly — **no Kaggle login
required** — via OpenML:

```python
from sklearn.datasets import fetch_openml
df = fetch_openml("creditcard", version=1, as_frame=True).frame
df["Class"] = df["Class"].astype(int)
df.to_csv("data/creditcard.csv", index=False)
```

## Repository layout

```
fraud-detection/
├── credit-card-fraud-detection.ipynb   # the analysis (run top-to-bottom, outputs embedded)
├── build_notebook.py                   # script that generates the notebook from source
├── requirements.txt
├── data/
│   └── creditcard.csv                  # fetched locally, not version-controlled
└── README.md
```

## How to run

```bash
python3 -m pip install -r requirements.txt

# 1. fetch the dataset into data/creditcard.csv (snippet above), then:
jupyter notebook credit-card-fraud-detection.ipynb
# or execute headless:
jupyter nbconvert --to notebook --execute --inplace credit-card-fraud-detection.ipynb
```

Everything is seeded (`RANDOM_STATE = 42`); a full run takes ~3 minutes on a laptop CPU.

## Notebook contents

1. Setup & reproducible data load
2. EDA — imbalance, `Amount` distribution, per-feature fraud correlation, top discriminative features
3. **The leakage trap** — reproducing the inflated ~0.99 scores
4. An honest experimental setup (split → scale → evaluate on real test only)
5. Cost-sensitive models (LogReg, Random Forest, XGBoost)
6. **SMOTE done right** (inside cross-validation)
7. Model comparison on the imbalanced test set
8. Precision-Recall & ROC curves
9. **Cost-based threshold selection** with confusion matrices
10. Feature importance
11. Unsupervised Isolation Forest contrast
12. Final scoreboard & takeaways

## Possible next steps

Hyperparameter search, probability calibration, SHAP explanations for investigator-facing
reason codes, and **time-aware validation** (train on the past, test on the future) to mirror
real deployment.

## Acknowledgements

- Dataset: ULB Machine Learning Group — *Credit Card Fraud Detection* (via OpenML).
