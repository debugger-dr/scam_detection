#!/usr/bin/env python3
"""Thin wrapper to download the dataset. Equivalent to `fraud-detection fetch-data`."""

from fraud_detection.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["fetch-data"]))
