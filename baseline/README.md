# Baram shared preprocessing baseline

This folder owns the team-shared raw feature pipeline. It creates leak-aware hourly train/test feature tables that every model branch can reuse. It does not implement deep learning, CatBoost, Optuna, ensembling, or final performance tuning.

`open/` is original competition data and must not be modified. `open/`, `outputs/`, and `baseline/cache/` are generated or local data paths and must not be committed to Git. SCADA is excluded from shared model features because no SCADA exists for test.

## Commands

Run from the repository root:

```bash
python3 baseline/scripts/inspect_data.py \
  --config baseline/configs/preprocessing.yaml

python3 baseline/scripts/build_features.py \
  --config baseline/configs/preprocessing.yaml

python3 baseline/scripts/smoke_test_rf.py \
  --config baseline/configs/preprocessing.yaml

PYTHONPATH=baseline/src python3 -m pytest baseline/tests -q
```

## Generated Files

Data checks:

- `outputs/checks/data_contract.json`
- `outputs/checks/time_semantics.json`
- `outputs/checks/validation_split_summary.json`

Raw shared features:

- `baseline/cache/features/train_features_raw.parquet`
- `baseline/cache/features/test_features_raw.parquet`
- `baseline/cache/features/train_labels.parquet`
- `baseline/cache/features/feature_metadata.json`

Smoke test:

- `outputs/baseline_preprocessing_smoke/metrics.json`
- `outputs/baseline_preprocessing_smoke/validation_predictions.csv`
- `outputs/baseline_preprocessing_smoke/submission_smoke.csv`

## Feature Tables

The raw feature tables have exactly one row per `forecast_kst_dtm`. They do not include targets, SCADA, imputation, or scaling. Train and test column order is identical.

Common features include:

- Time: `hour`, `dayofweek`, `month`, `dayofyear`, cyclic encodings, `lead_time_h`
- LDAPS wind: `ws10`, `ws50_mid`, `ws50_maxcomp`, `ws50_mincomp`
- GFS wind: `ws10`, `ws80`, `ws100`, `ws_pbl`, `ws850`, `ws700`, `ws500`, `gust`
- Weather: temperature, dew point, relative humidity, surface pressure, mean sea level pressure
- Grid summaries: mean, max, min, std per selected dynamic weather variable

Group nearest-grid features use explicit prefixes, for example:

- `group_1__ldaps_nearest__ws50_mid`
- `group_2__gfs_nearest__ws100`
- `group_3__ldaps_nearest__surface_pressure`

Use `baram.feature_builder.get_features_for_group(feature_table, group_id)` to select common columns plus only that group's nearest-grid columns. This keeps train/test schemas identical even when group models use different selected columns.

`feature_metadata.json` records each feature's source, formula, unit, scope, train/test missing counts, constant flag, and config hash. It also notes that `ws50_maxcomp` and `ws50_mincomp` combine component-wise extrema and are not observed maximum/minimum wind speeds.

## Labels

Labels are stored separately in `train_labels.parquet`. Weather features and labels join by:

`train_labels.kst_dtm == features.forecast_kst_dtm`

Each target has its own valid mask:

- `kpx_group_1`: rows where group 1 label exists
- `kpx_group_2`: rows where group 2 label exists
- `kpx_group_3`: rows where group 3 label exists

Targets are never interpolated. A missing target for one group does not remove the row for the other groups.

## Validation

Random split is not used.

- Group 1/2 train: 2022-2023 labels, represented by `2022-01-01 01:00:00` through `2024-01-01 00:00:00`
- Group 3 train: 2023 labels, represented by `2023-01-01 01:00:00` through `2024-01-01 00:00:00`
- Validation: 2024 labels, represented by `2024-01-01 01:00:00` through `2025-01-01 00:00:00`

The `2025-01-01 00:00:00` label is the end timestamp of the final 2024 hourly interval.

## Model Preprocessing Interfaces

Raw shared features are intentionally unscaled and unimputed.

Tree preprocessing:

- `SimpleImputer(strategy="median")`
- no scaler
- fit only on the current fold's training rows

Neural preprocessing:

- `SimpleImputer(strategy="median")`
- `StandardScaler`
- both fit only on the current fold's training rows

Use `fit_tree_preprocessor(...)` or `fit_neural_preprocessor(...)` from `baram.preprocessing`.

## RandomForest Smoke Test

`smoke_test_rf.py` trains one RandomForest per group only to verify preprocessing, splitting, imputation, metric, and submission contracts. It uses the official notebook parameters where available, but the resulting score is not a final performance experiment.

## Adding Future Experiments

Model branches should depend on the raw parquet files or on `build_feature_tables(config)`. Copy `baseline/configs/preprocessing.yaml` to an experiment config instead of editing the shared config directly. Add new features behind config flags and keep them out of `open/`. Fit imputers, scalers, encoders, feature selection, or model-specific transforms inside the fold/model pipeline, not in the shared raw feature cache.
