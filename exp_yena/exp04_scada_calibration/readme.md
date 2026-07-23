# Exp04: leak-safe SCADA calibration ablations

Exp02의 rolling validation과 CatBoost pipeline을 기준으로 기능을 flag로 관리하고,
각 변경을 독립 run으로 검증한다. Test에는 SCADA가 없으므로 validation과 test에
실제 SCADA를 feature로 넣지 않는다. SCADA 보조 feature는 outer-fold train에서만
학습하고 validation에는 예측값만 전달한다.

## Run order

1. `R0_baseline`: 수정된 exp02 기준선
2. `R1_lead`: lead-time interaction
3. `R2_direction`: raw U/V, direction sin/cos, 12-sector
4. `R3_lead_direction`: R1 + R2
5. `S0_analysis`: SCADA hourly aggregation and forecast offset audit
6. `S1_offset`: cross-fitted forecast-to-SCADA offset map
7. `S2_predicted_scada`: cross-fitted auxiliary SCADA CatBoost
8. `S3_power_curve`: S2 + empirical turbine-group power curve
9. `M1_weighted`: S3 + low-generation sample weighting
10. `M2_two_stage`: S3 + high-generation classifier/conditional regressor

M1과 M2는 서로 다른 실험이며 동시에 켤 수 없다.

## S0 analysis

```powershell
python -m experiments.exp04_scada_calibration.src.analyze_scada `
  --config experiments/exp04_scada_calibration/configs/r0_baseline.yaml `
  --output-root experiments/exp04_scada_calibration/outputs/S0_analysis
```

S0는 `scada_quality_summary.csv`, group별 hourly table과
`forecast_scada_offset_map.csv`를 생성한다. SCADA power는 turbine 정격범위 밖의
센서 이상치를 결측 처리하고 유효 turbine 수로 보정한다.

## Validation run

```powershell
python -m experiments.exp04_scada_calibration.src.run_experiment `
  --config experiments/exp04_scada_calibration/configs/r1_lead.yaml `
  --output-root experiments/exp04_scada_calibration/outputs/R1_lead `
  --no-finalize
```

Smoke test에서는 `--iterations 50`을 함께 사용한다. S2 이후에는 SCADA auxiliary
model도 여러 번 cross-fit하므로 `base.yaml`의 `auxiliary_model_params.iterations`도
50 정도로 낮춘 별도 smoke config를 쓰는 것이 좋다.

## Colab validation example

```powershell
$RunId = "$(Get-Date -Format yyyyMMdd_HHmmss)_R1_lead"
$Output = "experiments/exp04_scada_calibration/outputs/$RunId"

.\colab\windows\Invoke-ColabPython.ps1 -Distro $Distro -Session $Session `
  -ScriptPath .\colab\run_and_sync.py `
  --source $Output `
  --experiment exp04_scada_calibration `
  --run-id $RunId -- `
  python -m experiments.exp04_scada_calibration.src.run_experiment `
  --config experiments/exp04_scada_calibration/configs/r1_lead.yaml `
  --output-root $Output `
  --no-finalize
```

## Comparison

```powershell
python -m experiments.exp04_scada_calibration.src.compare_runs
```

모든 run에서 `oof_predictions.csv`, `fold_group_metrics.csv`,
`best_iterations.csv`, `feature_importances_by_fold.csv`, `val_results.yaml`을 저장한다.
