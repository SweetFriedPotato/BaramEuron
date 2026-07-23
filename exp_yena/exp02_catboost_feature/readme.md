# Exp02: CatBoost 기상·물리 피처 실험

이 디렉터리는 공용 전처리 파이프라인이 만든 시간별 기상 피처에 추가 파생 피처를 적용하고, 발전 단지별 CatBoost 회귀 모델을 학습하는 실험을 담고 있다. 목표는 단순 풍속 통계뿐 아니라 예보 모델 간 차이, 대기 상태, 풍력 발전의 비선형성을 모델에 전달하고 피처 중요도가 낮은 변수를 후속 실행에서 제거하는 것이다.

## 실험 요약

- 모델: `CatBoostRegressor`
- 대상: `kpx_group_1`, `kpx_group_2`, `kpx_group_3`
- 검증: 과거 구간 학습, 2024년 구간 검증의 시간 순서 분할
- 기본 파라미터: 2,000 iterations, depth 8, learning rate 0.03, Bernoulli subsampling 0.8
- 결측치 처리: 각 학습 구간에서 중앙값 imputer를 적합하며 스케일링은 하지 않음
- 조기 종료: 검증 실행에서 150 rounds
- 평가 지표: 대회 지표인 `1 - NMAE`와 `FICR`의 결합 점수
- 후처리: 음수 예측을 0으로 clipping하며 상한 clipping은 기본적으로 비활성화

현재 저장된 [`outputs/val_results.txt`](outputs/val_results.txt)의 결과는 다음과 같다.

| 지표 | 값 |
| --- | ---: |
| Total score | 0.574725 |
| 1 - NMAE | 0.861143 |
| FICR | 0.288306 |

이 값은 저장된 산출물의 기록이며, 실행 환경·CatBoost 버전·설정 파일·피처 제거 목록이 달라지면 재현 결과도 달라질 수 있다.

## 검증 구성

무작위 분할은 사용하지 않는다. 라벨의 시각은 해당 발전 시간 구간의 종료 시각이다.

- Group 1·2 학습: `2022-01-01 01:00:00` ~ `2024-01-01 00:00:00`
- Group 3 학습: `2023-01-01 01:00:00` ~ `2024-01-01 00:00:00`
- 전체 그룹 검증: `2024-01-01 01:00:00` ~ `2025-01-01 00:00:00`

실행기는 Group 1·2를 Fold A와 Fold B에서 각각 평가하고, Group 3을 Fold B에서 평가한 뒤 예측을 합쳐 최종 지표를 계산한다. 라벨이 없는 행은 그룹별로 제외한다.

## 파생 피처

[`src/feature_blocks.py`](src/feature_blocks.py)는 학습 구간에서만 상태를 적합하는 `FeatureBlockPipeline`을 제공한다.

- 풍력 물리: 높이별 풍속으로 vertical shear 계수와 117 m 허브 높이 풍속을 추정하고, 풍속의 제곱·세제곱 및 구간 피처를 생성
- 열역학: 섭씨 온도, 이슬점 편차, 습윤 공기 밀도, 해면기압과 지면기압 차이
- 예보 불일치: LDAPS와 GFS 풍속의 차이, 절댓값 차이, 비율
- 고급 기상: 돌풍 계수, 난류 강도, 순복사와 구름 상호작용, PBL 높이 대비 연직 시어, 착빙 위험, `공기 밀도 × 풍속³`

주의: 현재 [`src/run_experiment.py`](src/run_experiment.py)의 설정 연결 기준으로 `forecast_disagreement`는 `features.weather_summary`, 풍력 물리 블록은 `features.power_curve_features`를 읽는다. 따라서 제공된 두 YAML에서는 예보 불일치와 고급 기상 블록은 활성화되지만, YAML에 적힌 `wind_physics` 및 `thermodynamic` 키만으로는 같은 이름의 블록이 활성화되지 않는다. 실험을 비교할 때는 YAML의 의도보다 실행 코드가 실제로 읽는 키를 기준으로 해석해야 한다.

## 실행 준비

저장소 루트에서 실행한다. 원본 데이터는 `open/` 아래에 있어야 하며, 공용 raw feature cache를 먼저 생성해야 한다.

```powershell
python -m pip install -r baseline/requirements.txt
python -m pip install -r experiments/exp02_catboost_feature/requirements.txt

$env:PYTHONPATH = "baseline/src"
python baseline/scripts/build_features.py --config baseline/configs/preprocessing.yaml
```

필요한 주요 입력은 다음과 같다.

- `open/train/train_labels.csv`
- `open/sample_submission.csv`
- `baseline/cache/features/` 아래의 공용 train/test 피처 cache

## 실행 방법

먼저 짧은 validation-only smoke run을 권장한다.

```powershell
$env:PYTHONPATH = "baseline/src"
python -m experiments.exp02_catboost_feature.src.run_experiment `
  --config experiments/exp02_catboost_feature/configs/catboost_basic.yaml `
  --iterations 10 `
  --no-finalize `
  --output-root experiments/exp02_catboost_feature/outputs/smoke
```

전체 검증과 submission 생성:

```powershell
python -m experiments.exp02_catboost_feature.src.run_experiment `
  --config experiments/exp02_catboost_feature/configs/catboost_basic.yaml
```

주요 CLI 옵션은 다음과 같다.

- `--config`: 설정 파일 경로, 필수
- `--iterations`: YAML의 iteration 수를 임시로 덮어씀
- `--output-root`: 산출물 디렉터리를 덮어씀
- `--no-finalize`: 검증과 중요도 저장까지만 수행하고 전체 재학습 및 submission 생성을 생략

## GPU 설정

두 설정은 GPU 실패 처리 방식이 다르다.

- `catboost_basic.yaml`: 단조 제약을 사용한다. GPU 사전 검증이 실패하면 CPU로 자동 전환한다.
- `catboost_gpu_unconstrained.yaml`: CatBoost GPU에서 지원하지 않는 단조 제약을 제거하고 `require_gpu: true`로 설정한다. GPU를 사용할 수 없으면 실패하며 CPU로 전환하지 않는다.

GPU smoke run 예시:

```powershell
python -m experiments.exp02_catboost_feature.src.run_experiment `
  --config experiments/exp02_catboost_feature/configs/catboost_gpu_unconstrained.yaml `
  --iterations 10 `
  --no-finalize
```

## 피처 제거 실험

검증 실행은 fold별 CatBoost 중요도를 평균 내어 `feature_importances_report.csv`를 만든다. 그 다음 아래 명령으로 중요도 `0.05` 미만 피처 목록을 생성할 수 있다.

```powershell
python -m experiments.exp02_catboost_feature.src.feature_drop
```

생성된 [`configs/dropped_features_list.txt`](configs/dropped_features_list.txt)는 다음 `run_experiment` 실행에서 자동으로 읽힌다. 즉, 재현 가능한 비교 순서는 다음과 같다.

1. 제거 목록 없이 기준 실험을 실행한다.
2. `feature_drop`으로 중요도가 낮은 피처 목록을 만든다.
3. 같은 설정으로 다시 실행해 제거 전후 점수를 비교한다.

피처 제거 목록은 이전 실행 결과에 의존하므로, 기준 실험을 새로 만들 때는 기존 목록의 사용 여부를 명시적으로 확인해야 한다.

## 산출물

기본 출력 위치는 `experiments/exp02_catboost_feature/outputs/`이다.

- `val_results.txt`: `total_score`, `one_minus_nmae`, `ficr`
- `feature_importances_report.csv`: fold 평균 CatBoost 피처 중요도
- `submissions/submission_catboost_advanced.csv`: 전체 라벨로 재학습한 최종 예측
- `dropped_features_list.txt`: 과거 피처 선택 과정에서 생성된 참고 목록

별도 실행끼리 산출물이 덮어써지지 않도록 `--output-root`에 실행 ID를 포함하는 것을 권장한다.

## 디렉터리 구조

```text
exp02_catboost_feature/
├── configs/                 # CPU fallback 및 GPU 전용 설정, 피처 제거 목록
├── outputs/                 # 검증 지표, 중요도, submission
├── src/
│   ├── feature_blocks.py    # 기상·물리 파생 피처
│   ├── feature_drop.py      # 중요도 기반 피처 제거 목록 생성
│   ├── features.py          # 단조 제약 벡터 생성
│   └── run_experiment.py    # 검증, 학습, 추론 진입점
├── tests/
├── requirements.txt
└── readme.md
```
