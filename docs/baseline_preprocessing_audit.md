# Baseline preprocessing audit

## Scope

This audit covers the repository state before and during the shared preprocessing pipeline cleanup. The goal is not a new high-performance model. The goal is a reusable train/test-safe raw feature table, target masks, time validation, and a RandomForest smoke test.

## Existing `baseline/` audit

Existing files before this cleanup included:

- `baseline/README.md`
- `baseline/requirements.txt`
- configs: `random_forest.yaml`, `mlp_smoke.yaml`, `gru_smoke.yaml`
- scripts: `build_features.py`, `check_data.py`, `validate.py`, `train_full.py`, `make_submission.py`
- source modules: `config.py`, `constants.py`, `data.py`, `preprocessing.py`, `submission.py`, `validation.py`, `utils.py`, `inference.py`, `metrics.py`
- feature modules: `features/common.py`, `features/weather.py`, `features/spatial.py`, `features/sequence.py`
- model/training modules: `models/*`, `trainer.py`
- tests: `test_data.py`, `test_features.py`, `test_sequence.py`, `test_submission.py`, `test_validation.py`
- existing cache: hashed `train_features_*.parquet`, `test_features_*.parquet`, `feature_metadata_*.json`

Findings:

- Data loader: present. It loaded LDAPS/GFS, labels, metadata, and sample submission, and already checked grid counts and duplicate timestamp-grid pairs.
- Feature generation: present but implemented in `features/common.py` with hashed cache paths. It already had time features, weather summaries, derived wind speeds, and nearest-grid features.
- Parquet/cache: present, but not in the requested `baseline/cache/features/*_raw.parquet` contract.
- Train/test feature generation: present and schema-checked.
- Time-based validation: present in a small `time_split` helper, but not yet producing the requested validation summary artifact.
- Official baseline reproduction: partially reflected through `random_forest.yaml` and `docs/official_baseline_summary.md`; no dedicated official-baseline reproduction mode was kept in the shared preprocessing layer.
- Incomplete or duplicated code: preprocessing-oriented feature generation lived beside broader MLP/GRU/trainer infrastructure. For this task, the deep-learning/trainer code was left untouched and the shared preprocessing path was made explicit through new modules and scripts.

## Official RandomForest notebook analysis

Notebook: `[Baseline]_기상 예보 데이터 기반 RandomForest 풍력발전량 예측.ipynb`. The notebook was read but not modified.

- LDAPS/GFS use: reads train/test LDAPS and GFS CSVs directly. SCADA and `info.xlsx` are not used.
- Forecast-label join: labels are renamed from `kst_dtm` to `forecast_kst_dtm` and merged by exact timestamp equality. `kst_dtm` is the end timestamp of the one-hour generation interval.
- Grid handling: all numeric weather columns are averaged over grids per `forecast_kst_dtm`; spatial coordinates and grid identity are discarded.
- Features: calendar features (`month`, `day`, `hour`, `dayofweek`, `is_weekend`, hour/month sin-cos) plus LDAPS/GFS grid mean columns.
- Missing handling: a single `SimpleImputer(strategy="median")` is fit on the full training feature matrix and applied to test. It does not combine train and test for imputer fitting.
- Model input: one tabular matrix for all groups, with target-specific non-null label masks.
- Submission: preserves `sample_submission` key order, fills the three target columns, clips predictions to `[0, capacity]`, and writes `baseline_submit.csv`.
- Time sorting: mostly inherited from input order after timestamp merge; no explicit validation split.
- Leakage risk: train and test are not combined for imputation, but there is no time-based validation. The official notebook fits imputation on the whole training period before final test prediction, which is fine for final training but not suitable for fold validation.

## Implemented shared preprocessing path

Added or updated:

- `baseline/configs/preprocessing.yaml`
- `baseline/src/baram/feature_builder.py`
- `baseline/src/baram/metadata.py`
- `baseline/src/baram/features/time.py`
- `baseline/src/baram/features/wind.py`
- updated `features/weather.py`, `features/spatial.py`, `preprocessing.py`, `validation.py`, `submission.py`, `data.py`
- `baseline/scripts/inspect_data.py`
- `baseline/scripts/build_features.py`
- `baseline/scripts/smoke_test_rf.py`
- contract tests under `baseline/tests/`

The old broader experiment code remains available, but the public preprocessing contract now runs through `preprocessing.yaml` and the three scripts above.

## Shared pipeline versus official baseline

The official notebook is intentionally simple: grid means only, calendar features, one median imputer fitted on the full training matrix, no validation split, and direct final-test prediction.

The shared pipeline differs in these ways:

- It builds one explicit raw feature table per forecast timestamp and stores labels separately.
- It computes lead time from the actual `data_available_kst_dtm` relation.
- It adds derived LDAPS/GFS wind-speed features and weather summaries using mean, max, min, and std.
- It adds group-specific nearest-grid weather features from `info.xlsx`.
- It keeps raw shared features unimputed and unscaled.
- It fits imputer/scaler only inside each model/fold preprocessing step.
- It uses target-specific label masks and a time-based validation split.
- It writes data-contract, validation, smoke metric, and submission-contract artifacts.

## Data contract results

Generated files:

- `outputs/checks/data_contract.json`
- `outputs/checks/time_semantics.json`

Observed from actual data:

- LDAPS train: 420,864 rows, 26,304 forecast timestamps, exactly 16 grids per timestamp.
- GFS train: 236,736 rows, 26,304 forecast timestamps, exactly 9 grids per timestamp.
- LDAPS test: 140,160 rows, 8,760 forecast timestamps, exactly 16 grids per timestamp.
- GFS test: 78,840 rows, 8,760 forecast timestamps, exactly 9 grids per timestamp.
- Sample submission: 8,760 rows.
- Train weather and labels: `2022-01-01 01:00:00` through `2025-01-01 00:00:00`.
- Test weather and submission: `2025-01-01 01:00:00` through `2026-01-01 00:00:00`.
- Weather timestamps are timezone-naive KST strings.
- Lead times are 12 through 35 hours, computed from `forecast_kst_dtm - data_available_kst_dtm`.
- LDAPS test has missing source rows at 48 timestamp-grid locations; the missing forecast timestamps are represented in `time_semantics.json`.

## Feature table contract

Generated files:

- `baseline/cache/features/train_features_raw.parquet`
- `baseline/cache/features/test_features_raw.parquet`
- `baseline/cache/features/train_labels.parquet`
- `baseline/cache/features/feature_metadata.json`

Feature table results:

- Train features: 26,304 rows x 166 columns, including `forecast_kst_dtm`.
- Test features: 8,760 rows x 166 columns, including `forecast_kst_dtm`.
- Feature columns excluding timestamp: 165.
- Train/test schemas and column order match exactly.
- Test timestamps are unique.
- SCADA columns are absent.
- Target columns are absent from raw feature tables.
- Train feature missing values: 0.
- Test feature missing values: 126, from LDAPS test source missingness.
- No constant columns were removed in this run.

Feature families:

- Time: hour, dayofweek, month, dayofyear, cyclic encodings, lead time.
- LDAPS wind: `ws10`, `ws50_mid`, `ws50_maxcomp`, `ws50_mincomp`.
- GFS wind: `ws10`, `ws80`, `ws100`, `ws_pbl`, `ws850`, `ws700`, `ws500`, `gust`.
- Thermodynamic/weather: temperature, dew point, relative humidity, surface pressure, mean sea level pressure.
- Grid summary: mean, max, min, std for every selected dynamic weather variable.
- Group nearest-grid features: `group_1__...`, `group_2__...`, `group_3__...` columns.

Important metadata note: `ws50_maxcomp = sqrt(50MUmax^2 + 50MVmax^2)` combines component-wise maxima and is not necessarily an observed maximum wind speed.

## Label and validation contract

Labels are stored separately in `baseline/cache/features/train_labels.parquet` with `forecast_kst_dtm` plus targets. Target masks are target-specific; missing one group target does not remove the row for other groups.

Generated file:

- `outputs/checks/validation_split_summary.json`

Validation split:

- Group 1/2 train: `2022-01-01 01:00:00` through `2024-01-01 00:00:00`.
- Group 3 train: `2023-01-01 01:00:00` through `2024-01-01 00:00:00`.
- All groups valid: `2024-01-01 01:00:00` through `2025-01-01 00:00:00`.
- Group 1 train/valid rows: 17,422 / 8,778.
- Group 2 train/valid rows: 17,423 / 8,778.
- Group 3 train/valid rows: 8,760 / 8,778.
- Group 3 train has no 2022 target rows.
- Validation timestamps are strictly after train timestamps for all groups.

## RandomForest smoke test

Generated files:

- `outputs/baseline_preprocessing_smoke/metrics.json`
- `outputs/baseline_preprocessing_smoke/validation_predictions.csv`
- `outputs/baseline_preprocessing_smoke/submission_smoke.csv`

This is a preprocessing smoke test only. It uses the official notebook RandomForest parameters where available:

`n_estimators=120`, `max_depth=14`, `min_samples_leaf=8`, `max_features="sqrt"`, `random_state=42`, `n_jobs=-1`.

Results:

- Group 1: MAE 2247.9361, nMAE 0.104071, features 121.
- Group 2: MAE 2203.6350, nMAE 0.102020, features 121.
- Group 3: MAE 2206.5800, nMAE 0.105075, features 121.
- Macro MAE: 2219.3837.
- Macro nMAE: 0.103722.

Submission smoke contract:

- 8,760 rows.
- Same columns and key order as `sample_submission.csv`.
- No NaN or inf predictions.
- Numeric predictions.
- No duplicate timestamps.

## Tests

Command:

```bash
PYTHONPATH=baseline/src python3 -m pytest baseline/tests -q
```

Result:

- 22 passed.

## Time-ordering notes

The label timestamp `2025-01-01 00:00:00` belongs to the last one-hour interval of the 2024 validation period, so the validation window ends there. This is why train/valid boundaries use `2024-01-01 00:00:00` and `2024-01-01 01:00:00` rather than calendar-year midnight in the naive way.

## Known limitations and next-experiment cautions

- The RandomForest results are smoke-test metrics only and should not be treated as tuned competition performance.
- LDAPS test source data has missing values; model branches must handle them with train-fold-fitted imputers.
- SCADA remains excluded from public prediction features because test has no SCADA.
- Target-derived lag features are not included because they would require careful availability checks.
- Future feature/model branches should copy `baseline/configs/preprocessing.yaml` and keep shared raw preprocessing stable unless the team intentionally updates the contract.
