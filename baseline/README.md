# Baram shared baseline

최고 점수보다 RandomForest, MLP, GRU가 동일한 시간당 feature, 시간 split, metric, submission 경로를 공유하도록 만든 팀 공용 실험 틀이다. SCADA와 target lag는 사용하지 않으며 원본 `open/`과 공식 notebook을 수정하지 않는다. 공식 notebook은 저장소 루트의 `[Baseline]_기상 예보 데이터 기반 RandomForest 풍력발전량 예측.ipynb`이고 감사 결과는 `docs/official_baseline_summary.md`다.

## 설치와 실행

저장소 루트에서 Python 3.11/3.12 환경을 권장한다.

```bash
python -m venv .venv
.venv/bin/pip install -r baseline/requirements.txt
.venv/bin/python baseline/scripts/build_features.py --config baseline/configs/random_forest.yaml
.venv/bin/python baseline/scripts/validate.py --config baseline/configs/random_forest.yaml
.venv/bin/python baseline/scripts/validate.py --config baseline/configs/mlp_smoke.yaml
.venv/bin/python baseline/scripts/validate.py --config baseline/configs/gru_smoke.yaml
.venv/bin/python baseline/scripts/train_full.py --config baseline/configs/random_forest.yaml
.venv/bin/python baseline/scripts/make_submission.py --config baseline/configs/random_forest.yaml
PYTHONPATH=baseline/src .venv/bin/pytest baseline/tests
```

데이터 기본 위치는 `Baram/open/`이다. feature cache는 원본 크기/mtime, feature config, schema version hash를 사용한다. 각 run은 `outputs/runs/`, 제출은 `outputs/submissions/`에 저장된다.

## 구조와 확장

`src/baram/features`는 시간·기상·공간·sequence를 분리하고, `models`는 registry와 공통 fit/predict/save 계약을 제공한다. 새 feature는 모듈에 구현한 뒤 config flag로 켜고 `common.py`에서 조합한다. 새 모델은 공통 계약을 구현하고 registry 및 `_model` factory에 등록한다. TCN/Transformer는 `make_sequences` 결과를 그대로 받을 수 있다.

공용 config를 직접 수정하지 말고 `configs/exp_*.yaml`로 복사한다. branch는 `model/gru`, `model/lightgbm`, `feat/grid-weighted`처럼 한 관심사만 담는다. `open/`, cache, outputs, model artifact, 가상환경은 commit하지 않는다. 권장 실험 순서는 LightGBM/CatBoost, hub-height 보간 풍속, 거리 가중 grid, power curve/air density, 48시간 sequence다.

