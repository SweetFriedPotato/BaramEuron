#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/content/Baram}"
ARCHIVE="${ARCHIVE:-/content/drive/MyDrive/Baram/cache/baseline_features_3ea4bc4.tar.gz}"

if [[ -f "$REPO/baseline/cache/features/train_features_raw.parquet" ]]; then
  echo "baseline feature cache already exists"
  exit 0
fi

test -f "$ARCHIVE"
tar -xzf "$ARCHIVE" -C "$REPO"
test -f "$REPO/baseline/cache/features/train_features_raw.parquet"
echo "restored baseline cache from $ARCHIVE"
