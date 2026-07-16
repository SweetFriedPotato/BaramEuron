# exp01 CatBoost physics ablation report

## 실행 상태

- Branch: `exp/01-catboost-physics`
- Commit at run start: `a3d618f3bc74b7a9bcb3a73739d450aa8a7ae83f`
- Tests: `baseline 22 passed; baseline+experiment 25 passed`
- Submission: `/Users/jiheeandcats/Baram/experiments/exp01_catboost_physics/outputs/submissions/exp01_catboost_best_20260716_115249.csv`

## A-F 결과

| ablation_label   | experiment_id          | fold   |   macro_nmae |   feature_count_mean |   training_seconds |
|:-----------------|:-----------------------|:-------|-------------:|---------------------:|-------------------:|
| A                | rf_reference           | fold_a |     0.113616 |                  121 |                1.5 |
| A                | rf_reference           | fold_b |     0.103736 |                  121 |                3.9 |
| B                | catboost_basic         | fold_a |     0.109587 |                  121 |               15.6 |
| B                | catboost_basic         | fold_b |     0.097388 |                  121 |               24.9 |
| C                | catboost_spatial       | fold_a |     0.108073 |                  193 |               24.4 |
| C                | catboost_spatial       | fold_b |     0.095773 |                  193 |               38.8 |
| D                | catboost_wind_physics  | fold_a |     0.108151 |                  264 |               33.2 |
| D                | catboost_wind_physics  | fold_b |     0.095541 |                  264 |               64.2 |
| E                | catboost_thermodynamic | fold_a |     0.107381 |                  296 |               31.6 |
| E                | catboost_thermodynamic | fold_b |     0.095007 |                  296 |               76.1 |
| F                | catboost_full          | fold_a |     0.107532 |                  335 |               61.2 |
| F                | catboost_full          | fold_b |     0.095234 |                  335 |               76.5 |

## 결론

- 모델 교체 효과: Fold B macro nMAE가 RF `0.103736`에서 CatBoost basic `0.097388`로 `-0.006348` 변했다.
- RF 대비 selected model 개선 폭: `0.008729`.
- 공간 feature: 유지.
- 풍속 물리 feature: 유지.
- 열역학 feature: 유지.
- LDAPS/GFS 불일치 feature: 제외.
- selected raw/lower-clipped/capacity-clipped Fold B macro nMAE: `0.095128` / `0.095007` / `0.095007`.
- public 제출 가치: 있음 — RF reference보다 validation이 개선됐으며 계약 검증된 submission을 생성했다.
- TCN 진행 근거: 충분함 — 유지된 feature block과 제외된 block이 fold ablation으로 분리됐다.

## Feature block 판단

- `spatial`: 유지, Fold B macro Δ=-0.001615, 그룹 Δ=[g1=-0.002149, g2=-0.001173, g3=-0.001522], 유지/개선 그룹=3/3
- `wind_physics`: 유지, Fold B macro Δ=-0.000232, 그룹 Δ=[g1=-0.000084, g2=-0.000116, g3=-0.000496], 유지/개선 그룹=3/3 (Fold A/B 방향 불일치)
- `thermodynamic`: 유지, Fold B macro Δ=-0.000534, 그룹 Δ=[g1=-0.000853, g2=-0.000192, g3=-0.000557], 유지/개선 그룹=3/3
- `forecast_disagreement`: 제외, Fold B macro Δ=+0.000226, 그룹 Δ=[g1=+0.000297, g2=+0.000350, g3=+0.000033], 유지/개선 그룹=0/3

## Selected model 그룹별 Fold B

- group 1: nMAE `0.095511`
- group 2: nMAE `0.094562`
- group 3: nMAE `0.094948`

## Selected model high-wind 진단

- group 1: p90 `7.129 m/s`, nMAE `0.127472`, RF 대비 Δ `-0.002923`
- group 2: p90 `7.130 m/s`, nMAE `0.129377`, RF 대비 Δ `+0.001351`
- group 3: p90 `6.878 m/s`, nMAE `0.167700`, RF 대비 Δ `-0.023433`

## 월별 안정성

최저는 8월 `0.052102`, 최고는 1월 `0.169867`이며 월간 범위는 `0.117765`이다. 겨울, 특히 1월 오차가 커서 다음 TCN 실험에서도 계절별 안정성을 별도로 확인해야 한다.

## Final clipping

validation에서 capacity upper clipping의 추가 이득이 없으므로 상한은 적용하지 않았고, 음수만 0으로 clip했다.

- kpx_group_1: raw test range `-641.88–21348.83` kWh, final range `0.00–21348.83` kWh, upper clipping `미적용`
- kpx_group_2: raw test range `-579.08–21366.10` kWh, final range `0.00–21366.10` kWh, upper clipping `미적용`
- kpx_group_3: raw test range `-75.86–18790.31` kWh, final range `0.00–18790.31` kWh, upper clipping `미적용`

## 물리 feature 정의와 단위 검증

- 온도 원자료는 249.70–308.40 범위이고 압력은 약 87–104 kPa이므로 temperature는 K, pressure는 Pa로 확인했다. 상대습도는 %이며 LDAPS 최대가 110.38%라 수증기압 계산에서만 물리 범위 0–100으로 clip했다.
- 습공기 밀도는 포화수증기압에서 수증기 분압을 구한 뒤 `rho=(p-e)/(Rd*T)+e/(Rv*T)` 한 식만 사용했다.
- GFS 80/100 m 및 LDAPS 10/50 m shear alpha의 1%/99% clipping 한계는 매 fold train에서만 계산했다.
- group 3은 2022 label이 없고 Fold A에서도 평가되지 않아 group 1/2와 학습 이력 및 안정성이 다를 수 있다.

## TCN 전달 feature

- 공용 baseline raw/time/grid-summary/nearest feature.
- spatial
- wind_physics
- thermodynamic
- target lag와 SCADA는 전달하지 않는다. 세부 컬럼은 `outputs/feature_list_by_experiment.json`의 `catboost_selected` 항목을 사용한다.

공식 scorer는 저장소에 없었으므로 정산금 지표는 구현하거나 추측하지 않았다.
