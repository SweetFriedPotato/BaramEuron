#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/content/Baram}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
SOURCE="$REPO/experiments/exp02_daily_tcn_scada_aux/outputs"
DEST="/content/drive/MyDrive/Baram/runs/exp02_daily_tcn_scada_aux/$RUN_ID"

mkdir -p "$DEST"
tar -czf "$DEST/outputs.tar.gz" -C "$SOURCE" .
cp "$SOURCE/run_manifest.json" "$DEST/"
cp "$SOURCE/report.md" "$DEST/"
test -s "$DEST/outputs.tar.gz"
echo "$DEST"
