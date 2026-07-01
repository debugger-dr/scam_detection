.PHONY: help setup data train evaluate test report clean

PYTHON ?= python3
PKG := fraud_detection

help:
	@echo "Targets:"
	@echo "  setup     Install the package and dependencies (editable)"
	@echo "  data      Download the ULB credit-card dataset into data/"
	@echo "  train     Run the full pipeline: split, tune, fit, calibrate, threshold, save artifacts"
	@echo "  evaluate  Load artifacts and print the comparison table + bootstrap CIs"
	@echo "  report    Execute the narrative notebook headless"
	@echo "  test      Run the unit tests"
	@echo "  clean     Remove caches and run artifacts"

setup:
	$(PYTHON) -m pip install -e ".[notebook,dev]"

data:
	$(PYTHON) -m $(PKG).cli fetch-data

train:
	$(PYTHON) -m $(PKG).cli train

evaluate:
	$(PYTHON) -m $(PKG).cli evaluate

report:
	$(PYTHON) -m $(PKG).cli report

test:
	$(PYTHON) -m pytest

clean:
	rm -rf .pytest_cache .ruff_cache **/__pycache__ artifacts/*.joblib artifacts/*.json artifacts/figures
