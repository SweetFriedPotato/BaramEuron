#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/content/Baram}"
REFERENCE_ROOT="${REFERENCE_ROOT:-/content/drive/MyDrive/Baram/reference}"
DRIVE_RUN_ROOT="${DRIVE_RUN_ROOT:-/content/drive/MyDrive/Baram/runs/exp02_daily_tcn_scada_aux}"

cd "$REPO"
git fetch origin
git switch exp/02-tcn-aux
git pull --ff-only origin exp/02-tcn-aux

bash experiments/exp02_daily_tcn_scada_aux/scripts/restore_colab_cache.sh
if [[ ! -f open/train/train_labels.csv ]]; then
  mkdir -p open
  cp -a /content/drive/MyDrive/data/. open/
fi
test -f open/train/train_labels.csv
python -m pip install -r experiments/exp02_daily_tcn_scada_aux/requirements.txt

python - <<'PY'
import torch
print("PyTorch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("GPU available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
PY

PYTHONPATH=baseline/src:. python -m experiments.exp02_daily_tcn_scada_aux.src.run_experiment \
  --config-dir experiments/exp02_daily_tcn_scada_aux/configs \
  --catboost-reference "$REFERENCE_ROOT/catboost_selected_validation.csv" \
  --catboost-test "$REFERENCE_ROOT/exp01_catboost_selected_test.csv" \
  --drive-root "$DRIVE_RUN_ROOT"
