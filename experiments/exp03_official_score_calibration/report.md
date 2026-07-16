# exp03 official score calibration report

## Official scorer

DACON codeshare 14035의 `metric.ipynb`를 byte-for-byte 보존했다. SHA-256은 `0a3ab5a57dba0705dbdbda73cd723be37ef39cce388fcb22b1a220ce523a70f9`이며 공식 S3 원본과 일치한다. 공식 규칙은 실제 발전량 10% capacity 이상만 평가하고, normalized error ≤6%/≤8%에 시간별 단가 4/3, 그 외 0을 적용한다.

## Existing models rescored

Fold B 공식 순위:

1. `tcn_aux_015` — 0.610151 (1-NMAE 0.868491, FICR 0.351810)
2. `tcn_plain` — 0.608655 (1-NMAE 0.866579, FICR 0.350732)
3. `tcn_aux_005` — 0.607590 (1-NMAE 0.867225, FICR 0.347955)
4. `cat025_tcn075` — 0.606004 (1-NMAE 0.868511, FICR 0.343497)
5. `mlp_pointwise` — 0.600098 (1-NMAE 0.864509, FICR 0.335688)
6. `catboost_selected` — 0.593782 (1-NMAE 0.864816, FICR 0.322749)
7. `rf_reference` — 0.585223 (1-NMAE 0.865191, FICR 0.305256)

기존 선택 blend의 unmasked macro nMAE는 0.091757이지만 공식 mask nMAE는 0.131489이다. 공식 Score로는 aux 0.15와 plain TCN이 기존 blend보다 높아 선택 순서가 바뀌었다.

## Calibration

기존 prediction-only affine calibration은 walk-forward 7개 평가 분기 모두 baseline보다 개선됐다. 평균 delta는 +0.023145, 최악 delta는 +0.007456이다. 최종 test용 calibration은 전체 OOF에서 TCN weight 1.000와 group별 affine을 fit했으며, 선택 과정에 Public 점수를 사용하지 않았다.

Fold A에서 선택한 group별 TCN weight는 group 1/2/3 각각 1.000/0.825/0.975이다. 최종 global base의 affine `(scale, offset_kWh)`는 group 1/2/3 각각 (1.07, 648.0)/(0.994, 648.0)/(1.086, 619.5)이다.

4계절 calibration은 2024의 독립 quarter 중 4/4개에서 개선되어 retained=True로 판정했다. 다만 2024 평균 Score가 global affine 0.623245, seasonal affine 0.622589여서 최종 방식은 `global_affine`이다. 12개 월별 개별 calibration은 수행하지 않았다.

## FICR-aware training

- Official-mask ensemble Fold B: Score 0.631638, 1-NMAE 0.875252, FICR 0.388025
- λ=0.20 ensemble Fold B: Score 0.647595, 1-NMAE 0.875861, FICR 0.419328
- FICR-aware delta: +0.015956
- Winter 1.15 seed42 delta vs same seed: -0.006445
- Recency seed42 delta vs same seed: -0.001599

Winter와 recency는 둘 다 baseline λ=0.20 seed42를 넘지 못해 제외했다.

## True expanding-window quarterly retraining

각 quarter마다 feature physics state와 neural preprocessing을 train cutoff까지만 fit하고, 01:00~다음날 00:00 issue block이 분기 경계를 넘지 않게 재학습했다.

- 2023Q1: mask 0.612479, FICR-aware 0.631106, delta +0.018627
- 2023Q2: mask 0.642161, FICR-aware 0.653076, delta +0.010916
- 2023Q3: mask 0.592126, FICR-aware 0.600376, delta +0.008249
- 2023Q4: mask 0.650558, FICR-aware 0.655456, delta +0.004897
- 2024Q1: mask 0.625617, FICR-aware 0.646651, delta +0.021034
- 2024Q2: mask 0.620527, FICR-aware 0.625517, delta +0.004989
- 2024Q3: mask 0.643681, FICR-aware 0.656796, delta +0.013116
- 2024Q4: mask 0.665549, FICR-aware 0.670668, delta +0.005118

FICR-aware가 개선한 분기는 8/8개이며, 평균 Score는 0.642456, 평균 delta는 +0.010868, worst-quarter Score는 0.600376이다.

## Final selection

3-model convex search의 최적 weight는 CatBoost 0.00, 기존 TCN 0.00, FICR-aware 1.00이다. 최적해가 pure FICR-aware이므로 중복 ensemble submission은 만들지 않았다.

생성 submission:

- `exp03_calibration_only_20260716_151351.csv`
- `exp03_ficr_aware_20260716_151351.csv`

제출 우선순위는 FICR-aware, calibration-only 순이다. Public 점수(Exp01 0.6128785636, Exp02 0.6152232779)는 결과 문맥으로만 기록했고 어떤 parameter search에도 넣지 않았다.

## Next experiment

FICR-aware loss가 공식 validation의 정산금 성분을 크게 개선했으나 시간 모델끼리의 convex blend는 추가 이득이 없었다. 다음 단계는 같은 시간 입력을 더 섞는 대신 raw spatial grid 모델로 오차 상관을 낮추는 것이 타당하다.
