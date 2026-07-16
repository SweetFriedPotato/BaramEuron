# exp05 cross-group transfer v2

This experiment improves the Exp04 Exp03/raw ensemble without fitting the Public leaderboard. It first searches leakage-safe group-specific weights and then trains small residual stackers exclusively on rolling OOF predictions. A raw cross-group attention retrain is permitted only if the cheap OOF stages fail the documented acceptance gate.

The OOF contract reproduces the Exp04 global 0.40 blend before any search. Every outer quarter is predicted using only earlier-quarter labels. Full/test predictions are never stacker training rows, and Public metrics appear only in the report.

```bash
PYTHONPATH=baseline/src:. python -m pytest experiments/exp05_cross_group_transfer/tests -q
PYTHONPATH=baseline/src:. python -m experiments.exp05_cross_group_transfer.src.ensemble
```

Colab runtimes and Drive mount credentials are ephemeral. After mounting Drive on a replacement VM,
restore the branch, raw data, and baseline feature cache with:

```bash
python /content/drive/MyDrive/Baram/bootstrap_exp05_colab.py
```

The persistent raw archive is `MyDrive/Baram/cache/baram_open.tar.gz`; completed run artifacts are
written under `MyDrive/Baram/runs/exp05_cross_group_transfer/`.
