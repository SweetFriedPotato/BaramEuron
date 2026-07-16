# exp03 — official score calibration

This experiment vendors DACON's BARAM 2026 `metric.ipynb` byte-for-byte, wraps it without changing the scoring rules, rescales all exp01/exp02 OOF predictions, and selects prediction-only calibration with chronological separation.

The official thresholds are read from the published code: only truth at or above 10% capacity is evaluated; hourly normalized errors at most 6% and 8% receive unit prices 4 and 3, respectively; larger errors receive zero.

Run the prediction-only phase locally:

```bash
python -m experiments.exp03_official_score_calibration.src.run_experiment
```

Public leaderboard scores are report-only and never enter any search objective. Outputs are ignored by Git. The TCN training configs retain exp02 inputs: SCADA is auxiliary-label-only, never an input feature, and no target lag is used.
