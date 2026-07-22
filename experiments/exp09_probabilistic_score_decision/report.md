# Exp09 — probabilistic score-optimal decision

- Branch/commit: `exp/09-probabilistic-score-decision` / `332344a1d9c587d5f02b5e5ee8eb4fec43e26510`
- Tests: 138 passed
- GPU: `NVIDIA A100-SXM4-80GB`
- Exp04 exact reproduction: 0.6474395993905896 (target 0.647439599391)
- Public Score used for selection: no.

## Quantile models and score decision

| model_id                     |   seed |   total_score |   one_minus_nmae |     ficr |   groups_available | is_official_three_group_score   |   evaluated_samples |   total_samples |   evaluated_rate |   equal_quarter_mean |   worst_quarter |
|:-----------------------------|-------:|--------------:|-----------------:|---------:|-------------------:|:--------------------------------|--------------------:|----------------:|-----------------:|---------------------:|----------------:|
| q_c_calibrated_nested_shrink |     42 |      0.641514 |         0.875485 | 0.407542 |                  3 | True                            |               25515 |           43849 |         0.581883 |             0.64207  |        0.60239  |
| q_a_exp04_nested_shrink      |     42 |      0.640876 |         0.875302 | 0.40645  |                  3 | True                            |               25515 |           43849 |         0.581883 |             0.642692 |        0.603075 |
| q_b_hubwind_nested_shrink    |     42 |      0.640394 |         0.875032 | 0.405757 |                  3 | True                            |               25515 |           43849 |         0.581883 |             0.641699 |        0.60373  |
| q_a_exp04_decision           |     42 |      0.608706 |         0.867583 | 0.349829 |                  3 | True                            |               25515 |           43849 |         0.581883 |             0.613426 |        0.575543 |
| q_a_exp04_q50                |     42 |      0.605481 |         0.866531 | 0.344432 |                  3 | True                            |               25515 |           43849 |         0.581883 |             0.611111 |        0.567475 |
| q_b_hubwind_q50              |     42 |      0.600274 |         0.864279 | 0.336269 |                  3 | True                            |               25515 |           43849 |         0.581883 |             0.605619 |        0.573325 |
| q_a_exp04_mean               |     42 |      0.59944  |         0.864986 | 0.333893 |                  3 | True                            |               25515 |           43849 |         0.581883 |             0.606049 |        0.556388 |
| q_c_calibrated_decision      |     42 |      0.597981 |         0.86563  | 0.330333 |                  3 | True                            |               25515 |           43849 |         0.581883 |             0.601086 |        0.585138 |
| q_b_hubwind_decision         |     42 |      0.596793 |         0.863554 | 0.330033 |                  3 | True                            |               25515 |           43849 |         0.581883 |             0.6021   |        0.57461  |
| q_c_calibrated_q50           |     42 |      0.595663 |         0.86547  | 0.325857 |                  3 | True                            |               25515 |           43849 |         0.581883 |             0.599548 |        0.580487 |
| q_c_calibrated_mean          |     42 |      0.591852 |         0.865013 | 0.318691 |                  3 | True                            |               25515 |           43849 |         0.581883 |             0.595593 |        0.578643 |
| q_b_hubwind_mean             |     42 |      0.590664 |         0.863047 | 0.318281 |                  3 | True                            |               25515 |           43849 |         0.581883 |             0.597036 |        0.569368 |

Q-B adds cross-fitted Stage1 median/mean to Q-A; Q-C adds std/IQR/seed uncertainty. The Exp04 encoder architecture was reused but trained from scratch inside each nested fold; no outer-quarter-selected checkpoint was loaded.

## Calibration

|   seed | quarter   |   history_quarters |   pinball |   approximate_crps |   interval_90_coverage |   mean_absolute_coverage_error |   coverage_q05 |   coverage_q10 |   coverage_q20 |   coverage_q30 |   coverage_q40 |   coverage_q50 |   coverage_q60 |   coverage_q70 |   coverage_q80 |   coverage_q90 |   coverage_q95 |
|-------:|:----------|-------------------:|----------:|-------------------:|-----------------------:|-------------------------------:|---------------:|---------------:|---------------:|---------------:|---------------:|---------------:|---------------:|---------------:|---------------:|---------------:|---------------:|
|     42 | 2023Q1    |                  0 | 0.0429137 |          0.0858274 |              0.82037   |                      0.0339226 |      0.0898148 |       0.158102 |       0.198611 |       0.290972 |       0.384722 |       0.511343 |       0.566435 |       0.67037  |       0.729167 |       0.835648 |       0.910185 |
|     42 | 2023Q2    |                  1 | 0.0547831 |          0.109566  |              0.0984432 |                      0.251636  |      0.5538    |       0.57326  |       0.578297 |       0.587912 |       0.59707  |       0.610577 |       0.619048 |       0.62706  |       0.631181 |       0.641484 |       0.652244 |
|     42 | 2023Q3    |                  2 | 0.046884  |          0.0937679 |              0.133757  |                      0.281853  |      0.627749  |       0.662888 |       0.677624 |       0.694627 |       0.705282 |       0.720245 |       0.724552 |       0.729313 |       0.733167 |       0.747223 |       0.761505 |
|     42 | 2023Q4    |                  3 | 0.0552325 |          0.110465  |              0.205616  |                      0.215073  |      0.392663  |       0.406703 |       0.422101 |       0.425951 |       0.436594 |       0.468297 |       0.487998 |       0.500906 |       0.512002 |       0.550725 |       0.598279 |
|     42 | 2024Q1    |                  4 | 0.0526032 |          0.105206  |              0.457112  |                      0.158763  |      0.379121  |       0.406899 |       0.451923 |       0.506868 |       0.555861 |       0.636294 |       0.665293 |       0.695818 |       0.735501 |       0.788309 |       0.836233 |
|     42 | 2024Q2    |                  5 | 0.0459387 |          0.0918774 |              0.12851   |                      0.237407  |      0.527167  |       0.530983 |       0.559982 |       0.56685  |       0.572497 |       0.59417  |       0.603785 |       0.613553 |       0.627442 |       0.647283 |       0.655678 |
|     42 | 2024Q3    |                  6 | 0.0440218 |          0.0880435 |              0.133212  |                      0.264468  |      0.616863  |       0.622313 |       0.652589 |       0.661974 |       0.669694 |       0.691795 |       0.699213 |       0.708295 |       0.723282 |       0.740236 |       0.750076 |
|     42 | 2024Q4    |                  7 | 0.048578  |          0.097156  |              0.285326  |                      0.243196  |      0.547101  |       0.569746 |       0.624698 |       0.654891 |       0.676932 |       0.721769 |       0.74381  |       0.760266 |       0.787742 |       0.803895 |       0.832428 |

Seed expansion gate: `{'q_c_improves_q_b': True, 'q_c_minus_q_b': 0.0011189979702258546, 'calibration_stable': False, 'stable_quarters': 1, 'required_stable_quarters': 6, 'seeds_52_62_executed': False}`. The stability rule was fixed as at least 6/8 quarters having absolute 90% interval-coverage error <=0.10; therefore seeds 52/62 were not run.

## Final nested candidate

- Selected: `q_c_calibrated_nested_shrink`
- Rolling / delta: 0.64151349725854 / -0.005926102
- Equal-quarter mean / worst: 0.6420695043172239 / 0.6023899665467716
- Improved quarters / worst degradation: 0/8 / 0.01032838697091465
- 1-NMAE / FICR / group 3: 0.8754851486766227 / 0.4075418458404574 / 0.6164265114987935
- Decision shift (CF): `{'mean_cf': 0.018677442423678744, 'p50_cf': 0.016556898328993038, 'p95_cf': 0.041819714807581, 'maximum_cf': 0.10613082320601852}`

| model_id                     | slice     |   total_score |   one_minus_nmae |     ficr |   groups_available | is_official_three_group_score   |   evaluated_samples |   total_samples |   evaluated_rate |
|:-----------------------------|:----------|--------------:|-----------------:|---------:|-------------------:|:--------------------------------|--------------------:|----------------:|-----------------:|
| q_c_calibrated_nested_shrink | january   |      0.624668 |         0.860587 | 0.38875  |                  3 | True                            |                2772 |            3720 |         0.745161 |
| q_c_calibrated_nested_shrink | high_wind |      0.678491 |         0.880697 | 0.476285 |                  3 | True                            |                4147 |            4226 |         0.981306 |
| exp04                        | january   |      0.631677 |         0.857887 | 0.405468 |                  3 | True                            |                2772 |            3720 |         0.745161 |
| exp04                        | high_wind |      0.688622 |         0.879056 | 0.498189 |                  3 | True                            |                4147 |            4226 |         0.981306 |

## Acceptance

| check                                   | passed   |
|:----------------------------------------|:---------|
| rolling_at_least_0_649440               | False    |
| improvement_at_least_0_002              | False    |
| improved_quarters_at_least_6            | False    |
| worst_quarter_degradation_at_most_0_002 | False    |
| ficr_maintained                         | False    |
| one_minus_nmae_within_0_0005            | True     |
| group_3_maintained                      | False    |
| three_seed_mean_improves                | False    |
| decision_shift_p95_at_most_0_03         | False    |
| not_single_seed_dependent               | False    |

- Acceptance: **FAIL**
- Full training/submission: not executed.
- Persistent output: `experiments/exp09_probabilistic_score_decision/outputs`

## 다음 방향

보정 후 구간 폭이 급격히 축소되는 원인을 먼저 해결해야 한다. 다음 실험은 previous-only coverage calibration을 직접 목적함수로 검증하거나, Exp04 point champion을 유지한 채 분포 head의 scale parameterization을 재설계하는 것이 타당하다.
