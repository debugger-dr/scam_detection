#!/usr/bin/env python3
"""Thin wrapper to run the full pipeline. Equivalent to `fraud-detection train`."""

from fraud_detection.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["train"]))
