"""Command-line interface: ``fraud-detection {fetch-data,train,evaluate,report}``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import Config

DEFAULT_CONFIG = "config/default.yaml"


def _load_config(path: str) -> Config:
    return Config.from_yaml(path)


def cmd_fetch_data(args: argparse.Namespace) -> int:
    from .data import fetch_dataset

    cfg = _load_config(args.config)
    print(f"Fetching '{cfg.data.openml_name}' (v{cfg.data.openml_version}) from OpenML...")
    path = fetch_dataset(cfg, force=args.force)
    print(f"Dataset ready at {path}")
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    from .pipeline import run

    cfg = _load_config(args.config)
    if args.no_tuning:
        cfg.tuning.enabled = False
    print("Running pipeline (this may take a few minutes)...\n")
    result = run(cfg)
    _print_report(result)
    print(f"\nArtifacts written to {result.artifacts_dir}/ (models.joblib, metrics.json, figures/)")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    cfg = _load_config(args.config)
    metrics_path = cfg.artifacts_path / "metrics.json"
    if not metrics_path.exists():
        print(f"No metrics found at {metrics_path}. Run `fraud-detection train` first.",
              file=sys.stderr)
        return 1
    payload = json.loads(metrics_path.read_text())
    _print_metrics_payload(payload)
    print(f"\nFigures: {cfg.artifacts_path / 'figures'}/")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    import subprocess

    nb = Path(args.notebook)
    if not nb.exists():
        print(f"Notebook not found: {nb}", file=sys.stderr)
        return 1
    print(f"Executing {nb} headless...")
    cmd = [
        sys.executable, "-m", "nbconvert", "--to", "notebook", "--execute",
        "--inplace", str(nb),
    ]
    return subprocess.call(cmd)


def _print_report(result) -> None:
    import pandas as pd

    with pd.option_context("display.float_format", lambda v: f"{v:.4f}"):
        print(result.comparison.to_string())
    print(f"\nHeadline model      : {result.headline}")
    print(f"Cost-optimal thresh : {result.headline_threshold:.4f} (chosen on validation)")
    print(f"Brier score (test)  : {result.brier:.5f}")
    m = result.headline_test_metrics_at_threshold
    print(f"At that threshold   : recall={m['recall']:.3f}  precision={m['precision']:.3f}  "
          f"f1={m['f1']:.3f}")
    cv = result.smote_cv
    print(f"SMOTE-in-CV LogReg  : PR-AUC {cv['pr_auc_mean']:.4f} +/- {cv['pr_auc_std']:.4f}")


def _print_metrics_payload(payload: dict) -> None:
    print(f"Headline model      : {payload['headline']}")
    print(f"Cost-optimal thresh : {payload['headline_threshold']:.4f}")
    print(f"Brier score (test)  : {payload['brier']:.5f}\n")
    print(f"{'model':30} {'PR-AUC':>8} {'95% CI':>18} {'ROC-AUC':>8}")
    rows = sorted(payload["results"].items(), key=lambda kv: kv[1]["pr_auc"], reverse=True)
    for name, row in rows:
        ci = f"[{row.get('pr_auc_low', float('nan')):.3f}, {row.get('pr_auc_high', float('nan')):.3f}]"
        print(f"{name:30} {row['pr_auc']:>8.4f} {ci:>18} {row['roc_auc']:>8.4f}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fraud-detection", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_fetch = sub.add_parser("fetch-data", help="Download the dataset from OpenML")
    p_fetch.add_argument("--config", default=DEFAULT_CONFIG)
    p_fetch.add_argument("--force", action="store_true", help="Re-download even if present")
    p_fetch.set_defaults(func=cmd_fetch_data)

    p_train = sub.add_parser("train", help="Run the full pipeline and save artifacts")
    p_train.add_argument("--config", default=DEFAULT_CONFIG)
    p_train.add_argument("--no-tuning", action="store_true", help="Skip hyper-parameter search")
    p_train.set_defaults(func=cmd_train)

    p_eval = sub.add_parser("evaluate", help="Print the saved comparison table + CIs")
    p_eval.add_argument("--config", default=DEFAULT_CONFIG)
    p_eval.set_defaults(func=cmd_evaluate)

    p_report = sub.add_parser("report", help="Execute the narrative notebook headless")
    p_report.add_argument("--config", default=DEFAULT_CONFIG)
    p_report.add_argument(
        "--notebook", default="notebooks/credit-card-fraud-detection.ipynb"
    )
    p_report.set_defaults(func=cmd_report)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
