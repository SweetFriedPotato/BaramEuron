# exp05 cross-group transfer v2

This experiment improves the Exp04 Exp03/raw ensemble without fitting the Public leaderboard. It first searches leakage-safe group-specific weights and then trains small residual stackers exclusively on rolling OOF predictions. A raw cross-group attention retrain is permitted only if the cheap OOF stages fail the documented acceptance gate.

The OOF contract reproduces the Exp04 global 0.40 blend before any search. Every outer quarter is predicted using only earlier-quarter labels. Full/test predictions are never stacker training rows, and Public metrics appear only in the report.

```bash
PYTHONPATH=baseline/src:. python -m pytest experiments/exp05_cross_group_transfer/tests -q
PYTHONPATH=baseline/src:. python -m experiments.exp05_cross_group_transfer.src.ensemble
```
