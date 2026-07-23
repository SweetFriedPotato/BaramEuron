# Exp01: XGBoost with Monotonic Constraints

본 실험은 풍력 발전량 예측 모델에서 발생할 수 있는 비물리적 예측 오류를 방지하기 위해, 주요 기상 변수와 발전량 간의 물리적 인과 관계를 XGBoost의 단조성 제약 조건(Monotonic Constraints)으로 주입하고 그 성능을 검증하는 실험입니다.

---

## 1. 실험 개요 (Overview)

풍력 발전량은 풍속 등의 기상 조건이 개선됨에 따라 발전량이 단조 증가하는 물리적 특성을 가집니다. 일반적인 머신러닝 모델은 데이터의 노이즈로 인해 풍속이 증가함에도 예측 발전량이 감소하는 국소적 역전 현상(Non-monotonic behavior)을 보일 수 있습니다. 본 실험에서는 이러한 현상을 억제하기 위해 XGBoost 알고리즘에 단조 제약 조건을 인입하여 도메인 정렬을 꾀하고 일반화 성능을 극대화합니다.

* **대상 모델**: `XGBoost Regressor`
* **주요 기법**: 특정 피처 계열(예: 풍속 등)에 대한 `monotone_constraints` 주입
* **평가 메트릭**: 대회 공식 Custom Metric ($0.5 \times (1 - \text{NMAE}) + 0.5 \times \text{FICR}$)

---

## 2. 프로젝트 디렉토리 구조 (Directory Structure)

```text
experiments/exp01_xgboost_monotonic/
├── configs/
│   └── xgboost_mono.yaml      # 실험 하이퍼파라미터 및 데이터 경로 설정
└── src/
    ├── __init__.py
    ├── features.py            # 단조성 제약 조건 정의 및 매핑 유틸
    └── run_experiment.py      # 교차 검증, 전체 학습 및 추론 실행 메인 스크립트

```

---

## 3. 핵심 메커니즘 (Key Mechanisms)

### 3.1 단조성 제약 조건 주입 (Monotonic Constraints)

풍속 등 발전량과 강력한 양의 상관관계를 가져야 하는 독립 변수들을 정의하고, XGBoost 트리 분기 시 해당 변수의 분기 방향이 항상 목적 함수를 단조 증가($1$)시키는 방향으로만 제한되도록 규제합니다.

### 3.2 데이터 스케일 일관성 유지

본 파이프라인에서 모델은 원본 발전량 단위($\text{kWh}$)를 타깃으로 직접 학습 및 예측을 수행합니다. 검증 및 최종 제출 단계에서 불필요한 스케일 이중 곱셈(Double Scaling) 오차가 발생하지 않도록 역변환 로직을 제거하여 안정적인 잔차 계산을 보장합니다.

---

## 4. 환경 설정 및 실행 방법 (Usage)

### 4.1 의존성 및 환경 변수 설정

실행 전, 베이스라인 소스 폴더가 파이썬 경로에 포함되도록 환경 변수(`PYTHONPATH`)를 지정해야 합니다.


```powershell
$env:PYTHONPATH="baseline/src;."

```

### 4.2 실험 실행 (Run Experiment)

지정된 설정 파일(`yaml`)을 파라미터로 넘겨 학습, 5-Fold 교차 검증, 추론 및 최종 제출 파일 생성을 단일 파이프라인으로 수행합니다.

```bash
python -m experiments.exp01_xgboost_monotonic.src.run_experiment --config experiments/exp01_xgboost_monotonic/configs/xgboost_mono.yaml

```

**주요 CLI 옵션:**

* `--config` (Required): 실험 설정 `yaml` 파일 경로
* `--iterations` (Optional): XGBoost의 `n_estimators` 값을 임시로 오버라이드 (Smoke test 시 활용)
* `--output-root` (Optional): 결과물이 저장될 루트 디렉토리 지정
* `--no-finalize` (Optional): 최종 제출용 full-training 단계를 생략하고 검증(OOF) 점수만 빠르게 계산할 때 사용

---

## 5. 산출물 및 평가 결과 (Outputs)

실행이 완료되면 `--output-root` 경로(기본값: `outputs/`)에 아래와 같은 결과물이 저장됩니다.

* `val_results.txt`: 교차 검증 단계에서 도출된 `Total Score`, `1 - NMAE`, `FICR` 평가지표 요약본
* `submissions/submission_xgboost_monotonic.csv`: 데이콘 규격을 준수하여 물리적 모순이 해결된 최종 예측 제출 파일 (8,760 행, 결측치 없음)