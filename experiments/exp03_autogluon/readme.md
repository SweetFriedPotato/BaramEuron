# Exp03: AutoGluon

Exp02의 raw feature cache, leak-safe `FeatureBlockPipeline`, tree preprocessor와
tracked feature-drop 목록을 사용하고 모델만 AutoGluon `TabularPredictor`로 교체한 실험이다.

## Install

```powershell
pip install -r experiments/exp03_autogluon/requirements.txt

.\colab\windows\Invoke-Colab.ps1 -Distro $Distro `
  install -s $Session `
  -r experiments/exp03_autogluon/requirements.txt
```

## Validation only

```powershell
python -m experiments.exp03_autogluon.src.run_experiment `
  --config experiments/exp03_autogluon/configs/autogluon_gpu.yaml `
  --no-finalize


$RunId = "$(Get-Date -Format yyyyMMdd_HHmmss)_autogluon_val"
$Output = "experiments/exp03_autogluon/outputs/gpu/$RunId"

.\colab\windows\Invoke-ColabPython.ps1 -Distro $Distro -Session $Session `
  -ScriptPath .\colab\run_and_sync.py `
  --source $Output `
  --experiment exp03_autogluon `
  --run-id $RunId -- `
  python -m experiments.exp03_autogluon.src.run_experiment `
  --config experiments/exp03_autogluon/configs/autogluon_gpu.yaml `
  --output-root $Output `
  --time-limit 600 `
  --no-finalize
```

## Full run

```powershell
python -m experiments.exp03_autogluon.src.run_experiment `
  --config experiments/exp03_autogluon/configs/autogluon_gpu.yaml
```

`--time-limit`과 `--presets`로 YAML 값을 실행 시 덮어쓸 수 있다.

고유한 `--output-root`에서 생성한 importance로 drop 목록을 갱신하려면 보고서
경로를 명시한다.

```powershell
python experiments/exp03_autogluon/src/feature_drop.py `
  --report experiments/exp03_autogluon/outputs/gpu/<run-id>/feature_importances_report.csv
```
