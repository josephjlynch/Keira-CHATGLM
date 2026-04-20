#!/usr/bin/env bash
# Phase 5: rebuild windowed features from all CSVs under data/raw_labeled, then train
# the hub SVM and write models/classifier.joblib. Run from repo root after adding
# venue calibration sessions (see train_hub_svm / collect_labeled_data docstrings).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
python -m scentsation_ml.build_custom_6d \
  --input-dir data/raw_labeled \
  --output datasets/custom_6d.csv
python scentsation_ml/train_hub_svm.py \
  --data datasets/custom_6d.csv \
  --results-dir results/venue
