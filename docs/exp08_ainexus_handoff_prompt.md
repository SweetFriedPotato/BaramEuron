# Exp08 AI NEXUS Codex 인수인계 프롬프트

아래 내용 전체를 AI NEXUS 서버에서 새 Codex 세션의 첫 메시지로 사용한다.

---

당신은 한국동서발전 풍력발전량 예측 프로젝트 `Baram`의 Exp08을 이어받는 Codex다.
요약만 믿고 바로 코드를 수정하지 말고, 아래에 지정한 저장소 문서·소스·테스트·산출물과
현재 실행 프로세스를 직접 읽어서 사실을 재확인한 뒤 작업하라. 사용자와는 한국어로
소통하고, 도구를 쓰기 전에는 짧은 진행 설명을 제공하라.

## 0. 작업 목표와 현재 머신

- 현재 작업 디렉터리: `/home/work/baram/Baram`
- 반드시 이 디렉터리에서 작업한다.
- 영구 저장 마운트: `/home/work/baram`
- `/home/work`의 다른 경로는 연산 세션 종료 시 삭제된다.
- GPU: NVIDIA A100-SXM4 80GB
- RAM: 약 1TB
- CPU: 6 cores
- Git branch: `exp/08-scada-hubwind-pretraining`
- 인수인계 문서 추가 전 구현 HEAD: `8fd2a930f2644d27a8b6277d1fc02e8cee186fa6`
- 원격: `origin = https://github.com/SweetFriedPotato/BaramEuron.git`
- 프로젝트 전용 추가 패키지: `/home/work/baram/python_packages`
- 시스템 CUDA PyTorch를 재사용한다. 별도 PyTorch를 설치하거나 CUDA 버전을 바꾸지 마라.
- 환경 활성화:

```bash
cd /home/work/baram/Baram
source scripts/ainexus_env.sh
```

이 스크립트는 다음을 설정한다.

- `PYTHONPATH=/home/work/baram/python_packages:/home/work/baram/Baram/baseline/src:/home/work/baram/Baram`
- `MPLBACKEND=Agg`
- `PYTHONUNBUFFERED=1`
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`

`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`이 필요한 이유는 서버 이미지의 오래된 Dash pytest
플러그인과 Werkzeug가 충돌하기 때문이다. 프로젝트 결함이 아니다.

## 1. 첫 행동: 실행 중인 작업을 절대 중복 실행하지 말 것

최초 학습 시작 시에는 다음 두 프로세스가 실행 중이었다.

- 후속 자동 파이프라인: PID 2825

PID는 세션 상황에 따라 바뀔 수 있으므로 숫자만 믿지 말고 다음 명령으로 직접 확인하라.

```bash
source /home/work/baram/Baram/scripts/ainexus_env.sh
OUT=experiments/exp08_scada_hubwind_pretraining/outputs

ps -eo pid,etime,stat,cmd | \
  grep -E 'run_exp08_ainexus_pipeline|run_experiment --phase' | grep -v grep || true

printf 'stage1 checkpoints: '
find "$OUT/checkpoints/stage1" -name '*.pt' 2>/dev/null | wc -l
printf 'stage2 checkpoints: '
find "$OUT/checkpoints/stage2" -name '*.pt' 2>/dev/null | wc -l

tail -50 "$OUT/logs/stage1_seed42.log" 2>/dev/null || true
tail -50 "$OUT/logs/pipeline.log" 2>/dev/null || true
find "$OUT/logs" -maxdepth 1 -name '*.exit' -print -exec cat {} \; 2>/dev/null || true
nvidia-smi
```

인수인계 최종 갱신 시점(2026-07-22 14:19 KST)의 구체적 상태는 다음과 같았다.

- Stage1 seed 42: 32/32 quarter checkpoints 완료
- seed-42 child PID 2313은 정상 종료
- Stage1 seed 52가 PID 3939로 실행 중
- Stage2 checkpoint: 0
- `pipeline.log`: `[2026-07-22T14:19:00+09:00] starting stage1_seed52`
- `stage1_selection.json`: selected `s1_d_aux_init`, top two
  `s1_d_aux_init`, `s1_c_distribution`
- `.exit` 파일은 아직 없었음(seed 52 실행 중)
- 프로세스나 로그에 오류 징후는 없었음

중요:

1. 기존 Stage1 프로세스가 살아 있으면 같은 `--phase stage1 --seed 42`를 다시 실행하지 마라.
2. `scripts/run_exp08_ainexus_pipeline.sh`도 이미 실행 중이면 두 번째 파이프라인을 띄우지 마라.
3. 프로세스를 임의로 kill/restart하지 마라.
4. 기존 작업이 끝났는데 파이프라인이 다음 단계로 넘어가지 않은 경우에만 로그와 `.exit`
   파일을 근거로 원인을 진단하라.
5. 각 quarter는 `.pt`, `.history.json`, `.npz`가 저장되며, runner는 완료된 quarter를
   자동으로 건너뛴다. 재실행이 필요해도 완료분은 보존해야 한다.

## 2. 반드시 직접 읽을 파일

작업을 시작하기 전에 아래 파일을 전부 읽어라. 파일이 길면 나눠서 EOF까지 읽어라.

### 운영·인수인계

- `docs/ai_nexus_migration.md`
- `scripts/ainexus_env.sh`
- `scripts/run_exp08_ainexus_pipeline.sh`
- `.gitignore`

### 이전 실험의 근거

- `experiments/exp01_catboost_physics/report.md`
- `experiments/exp02_daily_tcn_scada_aux/report.md`
- `experiments/exp03_official_score_calibration/report.md`
- `experiments/exp04_raw_grid_spatiotemporal/report.md`
- `experiments/exp05_cross_group_transfer/report.md`
- `experiments/exp06_ficr_threshold_calibration/report.md`
- `experiments/exp07_threshold_aware_finetuning/report.md`

### Exp08 설명·설정·핵심 구현

- `experiments/exp08_scada_hubwind_pretraining/README.md`
- `experiments/exp08_scada_hubwind_pretraining/report.md`
- `experiments/exp08_scada_hubwind_pretraining/requirements.txt`
- `experiments/exp08_scada_hubwind_pretraining/configs/*.yaml`
- `experiments/exp08_scada_hubwind_pretraining/src/run_experiment.py`
- `experiments/exp08_scada_hubwind_pretraining/src/scada_contract.py`
- `experiments/exp08_scada_hubwind_pretraining/src/scada_hourly_targets.py`
- `experiments/exp08_scada_hubwind_pretraining/src/stage1_crossfit.py`
- `experiments/exp08_scada_hubwind_pretraining/src/stage1_dataset.py`
- `experiments/exp08_scada_hubwind_pretraining/src/stage1_model.py`
- `experiments/exp08_scada_hubwind_pretraining/src/stage2_dataset.py`
- `experiments/exp08_scada_hubwind_pretraining/src/stage2_model.py`
- `experiments/exp08_scada_hubwind_pretraining/src/transfer.py`
- `experiments/exp08_scada_hubwind_pretraining/src/trainer.py`
- `experiments/exp08_scada_hubwind_pretraining/src/evaluate.py`
- `experiments/exp08_scada_hubwind_pretraining/src/blend.py`
- `experiments/exp08_scada_hubwind_pretraining/src/make_submission.py`
- `experiments/exp08_scada_hubwind_pretraining/src/make_report.py`
- `experiments/exp08_scada_hubwind_pretraining/tests/test_exp08_contracts.py`

### 재사용되는 Exp02/03/04 구현

- `experiments/exp02_daily_tcn_scada_aux/src/scada_targets.py`
- `experiments/exp02_daily_tcn_scada_aux/src/trainer.py`
- `experiments/exp03_official_score_calibration/src/backtest.py`
- `experiments/exp03_official_score_calibration/src/evaluate.py`
- `experiments/exp03_official_score_calibration/src/ficr_surrogate.py`
- `experiments/exp03_official_score_calibration/src/train_variants.py`
- `experiments/exp04_raw_grid_spatiotemporal/src/raw_grid_contract.py`
- `experiments/exp04_raw_grid_spatiotemporal/src/raw_grid_loader.py`
- `experiments/exp04_raw_grid_spatiotemporal/src/models.py`
- `experiments/exp04_raw_grid_spatiotemporal/src/trainer.py`
- `experiments/exp04_raw_grid_spatiotemporal/src/evaluate.py`
- `experiments/exp04_raw_grid_spatiotemporal/src/blend.py`
- `experiments/exp04_raw_grid_spatiotemporal/src/run_experiment.py`

### 데이터 계약

- `open/data_description.md`
- `open/train/scada_vestas_train.csv`의 헤더·시간 범위·표본
- `open/train/scada_unison_train.csv`의 헤더·시간 범위·표본
- `open/train/train_labels.csv`의 헤더·시간 범위·표본
- `experiments/exp08_scada_hubwind_pretraining/outputs/checks/*.json`
- `experiments/exp08_scada_hubwind_pretraining/outputs/checks/*.csv`

CSV 원본 전체를 대화 컨텍스트에 출력하지 말고 `head`, `tail`, `wc`, pandas 요약으로
계약만 확인하라.

## 3. Git 상태와 이미 완료된 구현

먼저 확인:

```bash
git status --short
git branch --show-current
git log --oneline -10
git remote -v
```

인수인계 작성 시 원격 worktree는 clean이었다. 주요 Exp08 커밋:

- `23ef9ba feat: add SCADA-supervised hub-wind pretraining`
- `c99e5e9 fix: mask unavailable fold SCADA sources`
- `920df8b fix: preserve single-target hub-wind output axis`
- `6b67cf8 fix: gate joint hub-wind fine-tuning`
- `3da0dbe fix: complete Exp08 evidence aggregation`
- `62359e5 docs: add AI NEXUS migration runbook`
- `8fd2a93 chore: automate Exp08 AI NEXUS pipeline`

이미 구현된 항목:

- Exp08 디렉터리 구조, configs, source, tests
- SCADA group/turbine 및 hour-ending 계약
- fold-train-only SCADA cleaning과 target mask
- Stage1 4개 ablation
- expanding rolling cross-fit와 pre-2023 fallback/mask
- Stage2 B/C/D 및 조건부 E
- Exp03 official loss/scorer 재사용
- 3-component convex blend search
- acceptance gate
- figures/report/manifest 기본 생성
- month/group/lead/wind/calibration metric 집계 보완
- seed/group score 저장 보완
- AI NEXUS 환경 및 자동 파이프라인

전체 테스트는 로컬과 AI NEXUS에서 모두 `124 passed`였다. 원격 재검증 명령:

```bash
source scripts/ainexus_env.sh
python3 -m pytest -q
```

## 4. 프로젝트 역사와 이번 실험의 이유

현재 기준 champion은 Exp04의 Exp03/raw blend다.

- Rolling aggregate: `0.647439599391`
- Fold B: `0.650288`
- Public Score: `0.634005715`
- Public 1-NMAE: `0.8685185925`
- Public FICR: `0.3994928374`

Public 결과는 보고서 맥락으로만 사용하며 어떤 모델·가중치·하이퍼파라미터 선택에도
사용하지 않는다.

최근 negative results:

- Exp05 stacking/cross-group transfer: 개선 `+0.000233`, acceptance 실패
- Exp06 piecewise threshold calibration: 개선 `+0.000706`, FICR은 증가했지만
  1-NMAE와 안정성 악화, acceptance 실패
- Exp07 threshold-aware fine-tuning: 최종 Exp04와 동일, improved seeds 0/3,
  rescue gain 0, acceptance 실패

따라서 일반 residual stacking, 사후 calibration, 기존 representation의 threshold
fine-tuning 방향은 중단했다. Exp08은 실제 단지 SCADA 허브 풍속이라는 물리 supervision을
weather encoder에 주고, 그 cross-fitted 예측을 발전량 모델에 명시적으로 넣는 새로운 방향이다.

Exp07 fine-tuned checkpoint는 사용하지 않는다. Exp07에서 재사용하는 것은 scorer, rolling
protocol, 데이터 계약과 테스트뿐이다. 모델 초기값은 Exp03 및 Exp04 champion component다.

## 5. 절대 위반하면 안 되는 실험 원칙

1. SCADA는 Stage1 학습 target으로만 사용한다.
2. SCADA 실제값은 inference input에 절대 포함하지 않는다.
3. test pipeline은 SCADA 파일을 읽어서는 안 된다.
4. power target 또는 target lag를 입력으로 사용하지 않는다.
5. 발전량 label로 Stage1 풍속 target을 만들지 않는다.
6. Stage2에는 가능한 한 cross-fitted Stage1 prediction만 사용한다.
7. Stage2 train block보다 미래를 본 Stage1 prediction을 사용하지 않는다.
8. 초기 history 부족 구간은 Stage1 mask=0, forecast fallback, fallback indicator를 사용한다.
9. DACON official scorer와 Exp03 official loss를 그대로 재사용한다.
10. random split을 사용하지 않는다.
11. train/test 합산 통계나 validation/test 기반 normalization을 만들지 않는다.
12. Public Score로 parameter/model/blend 선택을 하지 않는다.
13. raw spatial encoder 전체 unfreeze를 하지 않는다.
14. 새로운 threshold lambda search를 하지 않는다.
15. acceptance 실패 모델의 full train/submission을 만들지 않는다.
16. 자동 제출하지 않는다.

## 6. SCADA 계약

그룹 매핑:

- group 1: VESTAS 1~6
- group 2: VESTAS 7~12
- group 3: UNISON 1~5

10분 SCADA를 label의 hour-ending 기준에 맞춘다. 시간별 target:

- `hub_ws_median`
- `hub_ws_mean`
- `hub_ws_std`
- `hub_ws_iqr`
- 보조 진단용 유효 터빈 수와 시간 내 변동성

풍향은 source 간 단위·표현이 명확히 확인되지 않아 현재 실험에서는 제외한다.

정제 기준은 fold train에서만 계산한다.

- negative/non-finite wind speed는 invalid
- 과도값은 고정 clip이 아니라 fold-train q0.001/q0.999와 metadata 사용
- 한 시간 유효 10분 관측이 절반 미만이면 해당 터빈 invalid
- 그룹 유효 터빈 수가 절반 미만이면 group target mask=0
- VESTAS/UNISON 통계를 분리
- group 3의 짧은 coverage와 결측을 별도 기록

확인된 SCADA coverage:

- group 1: 약 26,303 hourly targets
- group 2: 약 26,293 hourly targets
- group 3: 약 17,519 hourly targets

정확한 값은 `outputs/checks/scada_target_coverage.csv`를 다시 읽어 확인하라.

## 7. Stage1 설계와 선택

입력은 Exp04 raw-grid contract를 재사용한다.

- LDAPS raw grid
- GFS raw grid
- static grid position/distance/height
- lead time/time features
- Exp01 selected engineered context
- forecast disagreement block은 제외
- SCADA는 X tensor에 포함하지 않음

기반 모델은 Exp04 raw_hybrid_gated encoder:

- raw-grid spatial attention
- geo bias
- LDAPS/GFS source gate
- engineered weather context
- temporal TCN

power head 대신 `[B,24,3,4]` hub-wind distribution head를 사용한다. target 순서:

1. median
2. mean
3. log1p(std)
4. log1p(iqr)

group별 fold-train 통계로 표준화하고 group-balanced masked SmoothL1을 사용한다.

```text
1.00 * median
+ 0.50 * mean
+ 0.25 * log_std
+ 0.25 * log_iqr
```

Ablation:

- S1-A `s1_a_median`: median만
- S1-B `s1_b_mean`: median+mean
- S1-C `s1_c_distribution`: median+mean+std+IQR
- S1-D `s1_d_aux_init`: S1-C + 기존 SCADA auxiliary head 초기화

선택 순서:

- 우선 seed 42 full 4개 후보
- physical metric/quarter 안정성으로 상위 2개
- 상위 2개만 seeds 52/62
- 최종 판단은 Stage2 official score가 우선이며 Stage1 MAE만으로 champion을 고르지 않음

AI NEXUS에서 재실행한 seed-42 결과도 S1-D가 선택되고 S1-C가 2위였다.
아래 값은 현재 서버의 `stage1_ablation.csv` 기준이다.

- S1-A median MAE `1.189077`, corr `0.905068`, quarter std `0.099752`
- S1-B median MAE `1.191525`, corr `0.904241`, quarter std `0.099447`
- S1-C median MAE `1.195459`, corr `0.903447`, quarter std `0.106383`
- S1-D median MAE `1.188436`, corr `0.904975`, quarter std `0.092241`

S1-D group median MAE/correlation 참고:

- group 1: MAE `1.150692`, corr `0.897654`
- group 2: MAE `1.234138`, corr `0.907191`
- group 3: MAE `1.180478`, corr `0.901231`

과거 Colab 세션의 체크포인트는 복원하지 않았으며 위 수치는 AI NEXUS에서 새로 재현한
산출물이다. 이후 seed 52/62 및 Stage2 결과도 반드시 현재 서버 산출물을 기준으로 판단하라.

## 8. Stage2 설계와 gate

기존 Exp04 representation에 다음 Stage1 cross-fitted feature를 추가한다.

- predicted median/mean/std/IQR
- Stage1 seed ensemble std
- forecast hub wind - predicted realized hub wind
- predicted/forecast hub wind ratio
- Stage1 fallback indicator

작은 ratio denominator는 NaN 처리 후 fold-train imputation한다. 실제 SCADA target availability
pattern을 입력 feature로 사용하지 않는다.

모델:

- A: Exp04 champion reference
- B `s2_b_pretrained`: Stage1 encoder 초기화, power head/마지막 temporal block
- C `s2_c_explicit`: predicted median/mean 명시 입력
- D `s2_d_distribution`: C + std/IQR/seed uncertainty
- E `s2_e_joint`: 조건부 joint fine-tuning

seed 42에서 B~D를 먼저 실행한다. C 또는 D seed-42 rolling score가 Exp04
`0.647439599391`을 넘을 때만 E를 실행한다. E는 Stage1 distribution head를 retention head로
복사하고 다음 loss를 사용한다.

```text
power_nmae + 0.20 * soft_ficr + 0.05 * hub_wind_retention_loss
```

Transfer:

- B: Stage1 encoder freeze, power head+마지막 temporal block만 학습
- C/D: 초기 5 epoch encoder freeze, 이후 마지막 temporal block/source gate만 제한적 unfreeze
- E: encoder LR 1e-5, power head LR 1e-4, hub-wind head 유지
- raw spatial encoder 전체 unfreeze 금지

seed 42에서 상위 2개 Stage2 모델만 seeds 52/62를 실행한다.

## 9. Validation, blend, acceptance

Validation은 정확히 Exp03/04의 expanding-window 8-quarter protocol을 사용한다.

반드시 기록:

- rolling aggregate score
- equal-quarter mean
- worst quarter
- improved quarter count
- 1-NMAE
- FICR
- group별 score와 group 3
- January/high-wind slices
- Stage1 physical metrics
- residual correlation with Exp04

Blend 후보는 최대 3개:

- Exp03 original
- Exp04 raw original
- Exp08 Stage2 best

가중치는 nonnegative, sum=1, 0.025 coarse 후 best 주변 0.005 fine search이며 rolling OOF로
선택한다.

새 champion acceptance는 모두 만족해야 한다.

- rolling aggregate >= `0.649440`
- Exp04 대비 >= `+0.002`
- 최소 6/8 quarters 유지 또는 개선
- worst-quarter degradation <= `0.002`
- FICR 유지/개선
- 1-NMAE degradation <= `0.0005`
- group 3 유지/개선
- 3-seed 평균 개선
- 특정 seed 하나에만 의존하지 않음(최소 2/3 seed 개선)

보완 모델 조건:

- 단독 score가 비슷함
- Exp04 residual correlation < 0.90
- blend improvement >= +0.002
- 특정 quarter 하나에만 의존하지 않음

## 10. 자동 파이프라인의 실행 순서

`scripts/run_exp08_ainexus_pipeline.sh`는 기존 Stage1 seed 42 PID가 끝날 때까지 기다린 뒤:

1. Stage1 seed 42 완료 여부와 32 checkpoints 검증
2. Stage1 seed 52
3. Stage1 seed 62
4. Stage2 seed 42
5. Stage2 seed 52
6. Stage2 seed 62
7. `finalize`
8. `report`

각 단계 로그:

```text
outputs/logs/stage1_seed42.log
outputs/logs/stage1_seed52.log
outputs/logs/stage1_seed62.log
outputs/logs/stage2_seed42.log
outputs/logs/stage2_seed52.log
outputs/logs/stage2_seed62.log
outputs/logs/finalize.log
outputs/logs/report.log
outputs/logs/pipeline.log
```

파이프라인 단계 성공/실패는 같은 이름의 `.exit` 파일로 기록된다. 전체 성공은
`outputs/logs/pipeline.complete`로 확인한다.

파이프라인이 죽었을 때 복구 순서:

1. 마지막 `.exit` 및 해당 로그의 traceback 확인
2. Git 상태와 checkpoint/prediction 쌍 확인
3. 코드 결함이면 최소 수정+회귀 테스트+커밋+push
4. 실패한 phase만 같은 인자로 재실행
5. runner가 완료 checkpoint를 건너뛰는지 확인
6. 절대로 checkpoint 디렉터리를 삭제하거나 전체를 처음부터 다시 돌리지 않음

## 11. 완료 후 반드시 감사할 구현·산출물 gap

파이프라인 성공만 보고 끝내지 말고 원요청 acceptance와 산출물 목록을 대조하라.

특히 확인할 것:

1. `stage1_month_metrics.csv` 존재 및 내용
2. group metric에 predicted/observed mean과 calibration ratio 포함
3. Stage1 predicted-vs-SCADA figure가 실제 컬럼을 사용해 비어 있지 않음
4. `seed_scores.csv`가 정확한 seed-score pairing인지 확인
5. `group_scores.csv`에 Exp04, Exp08 후보, final blend 모두 포함
6. Exp02 단순 auxiliary prediction과의 직접 비교가 실제 artifact로 가능한지 확인
7. 불가능하면 report에 그 이유를 명시하고 허위 비교를 만들지 않음
8. report에 모든 acceptance check의 실제 bool/value를 포함
9. figure 10개가 유효한 데이터로 생성됐는지 확인
10. `run_manifest.json`의 tests는 124이며 실제 commit/artifact hash와 일치
11. Public score가 선택에 쓰이지 않았는지 확인
12. Exp07 fine-tuned checkpoint가 쓰이지 않았는지 확인

원요청상 예상 출력:

```text
checks/reference_reproduction.json
checks/scada_hourly_contract.json
checks/stage1_crossfit_contract.json
checks/leakage_audit.json
checks/stage2_feature_schema.json

metrics/stage1_ablation.csv
metrics/stage1_group_metrics.csv
metrics/stage1_month_metrics.csv
metrics/stage1_quarter_metrics.csv
metrics/stage1_wind_regime_metrics.csv
metrics/stage1_lead_time_metrics.csv
metrics/stage2_candidate_scores.csv
metrics/nested_quarter_scores.csv
metrics/seed_scores.csv
metrics/group_scores.csv
metrics/january_scores.csv
metrics/high_wind_scores.csv
metrics/residual_correlations.csv
metrics/blend_search.csv
metrics/final_candidate_scores.csv

predictions/stage1_oof_hubwind.csv
predictions/stage2_oof_predictions.csv
predictions/final_blend_predictions.csv

figures/stage1_predicted_vs_scada.png
figures/stage1_error_by_group.png
figures/stage1_error_by_lead_time.png
figures/stage1_high_wind_error.png
figures/stage2_score_comparison.png
figures/rolling_quarter_comparison.png
figures/group3_comparison.png
figures/residual_correlation.png
figures/blend_search.png
figures/final_score_comparison.png

report.md
run_manifest.json
```

## 12. Full train과 submission gate

현재 runner는 rolling pipeline/finalize/report를 중심으로 구현되어 있다. acceptance가 실패하면
full train과 submission을 실행하지 않는 것이 올바른 동작이다.

acceptance가 통과한 경우에는 작업이 아직 끝난 것이 아니다. 그때만 다음을 구현·검증한다.

- 모든 사용 가능한 weather/SCADA로 Stage1 full train, 3 seeds
- test hub-wind prediction
- 모든 유효 power labels로 Stage2 full train, 3 seeds
- rolling best epoch 중앙값 사용
- test seed ensemble과 accepted blend
- 최대 2개 submission: best Exp08, best Exp04/Exp08 blend
- 8,760 rows, sample key/order 일치, numeric, finite, no duplicate
- 자동 제출 금지

acceptance 실패 시 diagnostic submission을 우선순위 결과처럼 만들거나 추천하지 마라.

## 13. 테스트 및 수정 규칙

- 진단/보고 요청이면 먼저 읽고 근거를 제시한다.
- 코드 수정이 필요하면 최소 범위로 구현하고 관련 테스트+전체 테스트를 실행한다.
- `apply_patch`로 파일을 수정한다.
- 검색은 `rg`/`rg --files`를 우선 사용한다.
- 기존 ignored outputs/checkpoints는 사용자 자산이므로 삭제하지 않는다.
- `git reset --hard`, `git checkout --`, destructive cleanup 금지
- unrelated 변경을 stage/commit하지 않는다.
- 커밋은 작고 의미 있게 만들고 현재 Exp08 branch에 push한다.
- `outputs/`, 데이터, checkpoint, 개인키, 비밀번호를 Git에 추가하지 않는다.
- 서버 패키지/드라이버/CUDA/PyTorch를 무단 업그레이드하지 않는다.

## 14. 보안

- 이메일에 전달된 SSH 비밀번호나 `id_container` 개인키 내용을 대화·로그·문서·Git에
  절대 다시 출력하지 않는다.
- Jupyter endpoint가 인증 challenge 없이 home contents API를 노출하는 상태가 확인됐다.
- 실험 중 연결을 깨지 않도록 현재 key를 임의 교체하지는 말되, 완료 후 사용자에게 Cloud팀에
  Jupyter 인증/접근제어와 SSH password/private-key rotation을 요청하라고 알린다.
- 서버는 IP 제한이 없으므로 공개 URL/credential을 답변에 재기재하지 않는다.

## 15. 최종 보고 형식

최종 답변과 `experiments/exp08_scada_hubwind_pretraining/report.md`에 최소 다음을 포함한다.

- branch/commit
- 원격 GPU와 tests
- Exp04 exact reproduction
- SCADA target coverage
- Stage1 최적 target 구성 및 top-two
- Stage1 group별 median/mean/std/IQR MAE/correlation/calibration
- cross-fit/fallback 결과
- pretrained encoder 효과
- explicit predicted hub-wind 효과
- uncertainty feature 효과
- joint fine-tuning gate와 실행 여부
- seed별/평균 Stage2 score
- rolling aggregate/equal mean/worst
- improved quarter count
- 1-NMAE/FICR
- group 3
- January/high-wind
- Exp04 residual correlation
- best 3-component blend와 weights
- acceptance 각 조건의 bool/value
- full training/submission 실행 여부와 이유
- persistent output 경로
- Public은 context only였음을 명시
- 다음 실험 방향

결론은 수치로 명확히 내려라. acceptance 실패를 성공처럼 표현하지 말고, 실패해도 어떤
representation/feature가 유효했는지와 다음 방향을 증거 기반으로 정리하라.

## 16. 지금 바로 실행할 체크리스트

다음 순서로 시작하라.

1. 사용자에게 인수인계를 받았고 기존 학습을 중단하지 않은 채 상태부터 확인한다고 짧게 알린다.
2. `source scripts/ainexus_env.sh`.
3. Git branch/status/log 확인.
4. 위에 지정된 문서·source·tests를 직접 읽는다.
5. process/checkpoint/log/exit/pipeline.complete 상태 확인.
6. 기존 프로세스가 정상이면 모니터링하고 중복 실행하지 않는다.
7. 실패했다면 traceback과 checkpoint 상태를 근거로 최소 수정한다.
8. 파이프라인 완료 후 산출물/acceptance gap audit를 수행한다.
9. 필요한 보완 구현과 124 tests를 실행한다.
10. report/manifest를 최종화하고 Git commit/push한다.
11. acceptance가 통과한 경우에만 full-train/submission gate를 이어간다.
12. 사용자에게 진행 중에는 60초 이상 무소식이 없도록 간단한 상태 업데이트를 제공한다.

이 작업은 단순히 프로세스가 끝나는 것을 기다리는 일이 아니다. 기존 실험 설계의 누수 방지,
rolling protocol, official scorer, acceptance gate를 보존하면서 실제 산출물이 원요청을 모두
충족하는지 끝까지 검증하고, 안전한 다음 단계가 남아 있는 동안 스스로 계속 진행하라.

---
