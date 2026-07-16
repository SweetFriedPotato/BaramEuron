# exp02_daily_tcn_scada_aux

exp01 selected feature(`baseline + spatial + wind_physics + thermodynamic`)를 세 그룹 공통/전용 union으로 재구성하고, 24시간 issue block에서 pointwise MLP와 daily TCN을 비교한다. SCADA hub wind는 입력이 아니라 auxiliary target으로만 사용한다.

## 핵심 계약

- train/test issue는 각각 1,096/365개이며 모든 issue가 24시간, 1시간 간격, lead 12–35시간이다.
- non-causal convolution은 같은 issue의 24시간 예보가 issue time에 모두 사용 가능할 때만 활성화한다.
- Fold A: 2022 train → 2023 valid, group 3 power/aux mask 제외.
- Fold B: 2022–2023 train → 2024 valid. group 3의 2022 mask는 0이다.
- neural preprocessing과 SCADA scaling/extreme mask는 fold train에서만 fit한다.
- target은 capacity factor, loss는 그룹별 masked MAE의 평균이다.
- SCADA, target lag, forecast disagreement는 input feature가 아니다.

## 로컬 테스트와 smoke

```bash
PYTHONPATH=baseline/src:. .venv/bin/python -m pytest \
  baseline/tests \
  experiments/exp01_catboost_physics/tests \
  experiments/exp02_daily_tcn_scada_aux/tests -q

PYTHONPATH=baseline/src:. .venv/bin/python -m \
  experiments.exp02_daily_tcn_scada_aux.src.run_experiment \
  --smoke \
  --output-root experiments/exp02_daily_tcn_scada_aux/outputs/smoke
```

smoke는 MLP와 plain TCN을 seed 42, 3 epochs로 실행하고 full train/submission은 만들지 않는다.

## Colab A100 full run

검증된 exp01 reference 두 파일이 Drive의 `MyDrive/Baram/reference/`에 있어야 한다.

```bash
colab console -s baram
cd /content/Baram
bash experiments/exp02_daily_tcn_scada_aux/scripts/run_colab.sh
```

각 seed checkpoint와 preprocessing object는 즉시 Drive run 폴더로 복사된다. 종료 시 전체 outputs archive, report, manifest, submission을 다시 Drive에 저장한다.
