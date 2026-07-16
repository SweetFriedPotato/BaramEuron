# exp01_catboost_physics

첫 성능 실험으로 공식 RandomForest와 CatBoost를 시간 기반 validation에서 비교하고, spatial → wind physics → thermodynamic → LDAPS/GFS disagreement feature를 누적 ablation한다. SCADA, target lag, random split은 사용하지 않는다.

## 재사용하는 공용 계약

- `baram.feature_builder.load_raw_feature_artifacts`: 공용 26,304/8,760행 raw parquet
- `baram.feature_builder.get_features_for_group`: 해당 그룹 nearest-grid feature만 선택
- `baram.validation.split_labeled_table`: Fold A/B 모두 공용 시간 split 함수로 생성
- `baram.preprocessing.fit_tree_preprocessor`: RF fold-train median imputer
- `baram.constants.CAPACITY_KWH`: capacity factor 변환
- `baram.submission.create_submission`: 제출 key/order/dtype/finite 계약

공식 notebook의 RF 설정은 `n_estimators=120`, `max_depth=14`, `min_samples_leaf=8`, `max_features=sqrt`, `random_state=42`, `n_jobs=-1`이다. 이 설정을 A에 고정하고 튜닝하지 않는다.

## 실행

저장소 루트에서 기존 `.venv`를 사용한다.

```bash
.venv/bin/python -m ensurepip --upgrade
.venv/bin/python -m pip install -r baseline/requirements.txt
.venv/bin/python -m pip install -r experiments/exp01_catboost_physics/requirements.txt

PYTHONPATH=baseline/src:. .venv/bin/python -m pytest baseline/tests experiments/exp01_catboost_physics/tests -q

# submission을 만들지 않는 300-iteration pipeline smoke
PYTHONPATH=baseline/src:. .venv/bin/python -m experiments.exp01_catboost_physics.src.run_experiment \
  --iterations 300 \
  --output-root experiments/exp01_catboost_physics/outputs/smoke \
  --no-finalize

# A-F full ablation, selected 조합 재검증, full train, submission
PYTHONPATH=baseline/src:. .venv/bin/python -m experiments.exp01_catboost_physics.src.run_experiment
```

CatBoost는 지원 GPU가 감지되면 GPU를 먼저 사용하고 오류 시 CPU로 재시도한다. CatBoost의 target은 capacity factor이며 metric과 submission 전에 kWh로 되돌린다. alpha clipping과 high-wind p90은 각 fold train에서만 계산된다.

## Fold

- Fold A, group 1/2: 2022 train → 2023 valid. group 3은 평가하지 않는다.
- Fold B, group 1/2: 2022–2023 train → 2024 valid.
- Fold B, group 3: 2023 train → 2024 valid.

label timestamp는 집계 종료시각이므로 각 연도의 마지막 구간은 다음 해 1월 1일 00:00까지다.

## Feature block

- Spatial: grid dispersion, 터빈 그룹 중심과 grid 사이 Haversine 거리의 역거리 가중 wind, nearest와 grid mean 차이/비율
- Wind physics: GFS 80/100 m 및 LDAPS 10/50 m shear, 117 m extrapolation, 선택 wind 제곱/세제곱, 설정에 기록된 wind bin
- Thermodynamic: Celsius, dew-point depression, 검증된 습공기 밀도, MSLP-surface pressure
- Disagreement: 의미가 가까운 LDAPS/GFS wind의 차이, 절댓값 차이, 안전한 비율

`outputs/` 전체는 gitignore 대상이다. 코드, config, tests와 최종 요약 `report.md`만 commit한다. 생성된 submission을 대회에 자동 제출하지 않는다.
