# Exp08 — SCADA-supervised hub-wind pretraining

- Branch/commit: `exp/08-scada-hubwind-pretraining` / `c316842763e682adf0103a43a0429a9f8fbdee1c`
- Tests: 124
- GPU: `NVIDIA A100-SXM4-80GB`
- Exp04 exact reproduction: 0.6474395993905896 (exact=True)
- Public usage: context only; never used for model/weight selection.
- Exp07 fine-tuned checkpoints: not used.

## SCADA target contract

|   group_id | source   |   available_hours |   missing_hours |   coverage | first_available     | last_available      |
|-----------:|:---------|------------------:|----------------:|-----------:|:--------------------|:--------------------|
|          1 | vestas   |             26303 |               1 |   0.999962 | 2022-01-01 02:00:00 | 2025-01-01 00:00:00 |
|          2 | vestas   |             26293 |              11 |   0.999582 | 2022-01-01 02:00:00 | 2025-01-01 00:00:00 |
|          3 | unison   |             17519 |            8785 |   0.66602  | 2023-01-01 01:00:00 | 2025-01-01 00:00:00 |

## Stage 1

| model_id          |   seed |   target_count |   group_balanced_median_mae |   median_pearson |   quarter_mae_std |   stage2_score |
|:------------------|-------:|---------------:|----------------------------:|-----------------:|------------------:|---------------:|
| s1_a_median       |     42 |              1 |                     1.18908 |         0.905068 |         0.0997519 |            nan |
| s1_b_mean         |     42 |              2 |                     1.19152 |         0.904241 |         0.0994466 |            nan |
| s1_c_distribution |     42 |              4 |                     1.19546 |         0.903447 |         0.106383  |            nan |
| s1_d_aux_init     |     42 |              4 |                     1.18844 |         0.904975 |         0.0922408 |            nan |
| s1_d_aux_init     |     52 |              4 |                     1.19281 |         0.903876 |         0.0918336 |            nan |
| s1_c_distribution |     52 |              4 |                     1.19713 |         0.902831 |         0.101345  |            nan |
| s1_d_aux_init     |     62 |              4 |                     1.19086 |         0.904907 |         0.0937202 |            nan |
| s1_c_distribution |     62 |              4 |                     1.18927 |         0.904247 |         0.0930021 |            nan |
Selected/top-two: `s1_d_aux_init` / `['s1_d_aux_init', 's1_c_distribution']`.

| model_id      |   seed |   group_id | target        |   samples |      mae |     rmse |   pearson |   spearman |   predicted_mean |   observed_mean |   calibration_ratio |
|:--------------|-------:|-----------:|:--------------|----------:|---------:|---------:|----------:|-----------:|-----------------:|----------------:|--------------------:|
| s1_d_aux_init |     42 |          1 | hub_ws_median |     17543 | 1.15069  | 1.5099   |  0.897654 |   0.895425 |         6.88858  |        6.91054  |            0.996823 |
| s1_d_aux_init |     42 |          1 | hub_ws_mean   |     17543 | 1.14654  | 1.5078   |  0.895786 |   0.893975 |         6.63884  |        6.8128   |            0.974465 |
| s1_d_aux_init |     42 |          1 | hub_ws_std    |     17543 | 0.343025 | 0.49593  |  0.706147 |   0.700055 |         0.916202 |        0.82039  |            1.11679  |
| s1_d_aux_init |     42 |          1 | hub_ws_iqr    |     17543 | 0.487972 | 0.715087 |  0.71931  |   0.663385 |         0.98849  |        1.07639  |            0.918338 |
| s1_d_aux_init |     42 |          2 | hub_ws_median |     17529 | 1.23414  | 1.62513  |  0.907191 |   0.902628 |         7.19763  |        7.23249  |            0.995181 |
| s1_d_aux_init |     42 |          2 | hub_ws_mean   |     17529 | 1.20335  | 1.58421  |  0.908182 |   0.902852 |         7.04926  |        7.12695  |            0.989099 |
| s1_d_aux_init |     42 |          2 | hub_ws_std    |     17529 | 0.304677 | 0.424852 |  0.779358 |   0.791497 |         0.924449 |        0.982517 |            0.940898 |
| s1_d_aux_init |     42 |          2 | hub_ws_iqr    |     17529 | 0.513528 | 0.725534 |  0.701941 |   0.716895 |         1.16575  |        1.30119  |            0.895914 |
| s1_d_aux_init |     42 |          3 | hub_ws_median |     15354 | 1.18048  | 1.59245  |  0.901231 |   0.890838 |         5.84798  |        5.80778  |            1.00692  |
| s1_d_aux_init |     42 |          3 | hub_ws_mean   |     15354 | 1.13459  | 1.52798  |  0.90439  |   0.893888 |         5.76823  |        5.74513  |            1.00402  |
| s1_d_aux_init |     42 |          3 | hub_ws_std    |     15354 | 0.272129 | 0.365492 |  0.753707 |   0.76922  |         0.784334 |        0.784792 |            0.999418 |
| s1_d_aux_init |     42 |          3 | hub_ws_iqr    |     15354 | 0.439326 | 0.622241 |  0.607523 |   0.646431 |         0.848204 |        0.889793 |            0.953259 |
| s1_d_aux_init |     52 |          1 | hub_ws_median |     17543 | 1.15344  | 1.50792  |  0.897521 |   0.897666 |         6.9227   |        6.91054  |            1.00176  |
| s1_d_aux_init |     52 |          1 | hub_ws_mean   |     17543 | 1.13396  | 1.48867  |  0.897516 |   0.897801 |         6.71408  |        6.8128   |            0.98551  |
| s1_d_aux_init |     52 |          1 | hub_ws_std    |     17543 | 0.358048 | 0.519264 |  0.695207 |   0.699298 |         0.949907 |        0.82039  |            1.15787  |
| s1_d_aux_init |     52 |          1 | hub_ws_iqr    |     17543 | 0.485583 | 0.709447 |  0.724787 |   0.667731 |         1.00084  |        1.07639  |            0.929813 |
| s1_d_aux_init |     52 |          2 | hub_ws_median |     17529 | 1.23958  | 1.63581  |  0.905918 |   0.903565 |         7.31484  |        7.23249  |            1.01139  |
| s1_d_aux_init |     52 |          2 | hub_ws_mean   |     17529 | 1.20636  | 1.58943  |  0.907235 |   0.903783 |         7.22186  |        7.12695  |            1.01332  |
| s1_d_aux_init |     52 |          2 | hub_ws_std    |     17529 | 0.310348 | 0.429819 |  0.771876 |   0.789848 |         0.936463 |        0.982517 |            0.953126 |
| s1_d_aux_init |     52 |          2 | hub_ws_iqr    |     17529 | 0.520942 | 0.734325 |  0.692527 |   0.716982 |         1.16924  |        1.30119  |            0.898597 |
| s1_d_aux_init |     52 |          3 | hub_ws_median |     15354 | 1.18542  | 1.6059   |  0.89927  |   0.891038 |         5.84679  |        5.80778  |            1.00672  |
| s1_d_aux_init |     52 |          3 | hub_ws_mean   |     15354 | 1.1414   | 1.54026  |  0.902564 |   0.894385 |         5.76921  |        5.74513  |            1.00419  |
| s1_d_aux_init |     52 |          3 | hub_ws_std    |     15354 | 0.273161 | 0.368832 |  0.747882 |   0.767595 |         0.776591 |        0.784792 |            0.989551 |
| s1_d_aux_init |     52 |          3 | hub_ws_iqr    |     15354 | 0.438419 | 0.623176 |  0.608359 |   0.647504 |         0.82418  |        0.889793 |            0.92626  |
| s1_d_aux_init |     62 |          1 | hub_ws_median |     17543 | 1.15438  | 1.50807  |  0.898292 |   0.897193 |         6.88144  |        6.91054  |            0.99579  |
| s1_d_aux_init |     62 |          1 | hub_ws_mean   |     17543 | 1.13569  | 1.49106  |  0.898216 |   0.8978   |         6.63848  |        6.8128   |            0.974413 |
| s1_d_aux_init |     62 |          1 | hub_ws_std    |     17543 | 0.342576 | 0.493034 |  0.707317 |   0.70218  |         0.921026 |        0.82039  |            1.12267  |
| s1_d_aux_init |     62 |          1 | hub_ws_iqr    |     17543 | 0.481287 | 0.707449 |  0.732484 |   0.668695 |         0.968103 |        1.07639  |            0.899398 |
| s1_d_aux_init |     62 |          2 | hub_ws_median |     17529 | 1.2347   | 1.62825  |  0.907105 |   0.904622 |         7.22679  |        7.23249  |            0.999212 |
| s1_d_aux_init |     62 |          2 | hub_ws_mean   |     17529 | 1.20052  | 1.58073  |  0.908541 |   0.90483  |         7.08639  |        7.12695  |            0.994309 |
| s1_d_aux_init |     62 |          2 | hub_ws_std    |     17529 | 0.301336 | 0.42116  |  0.786001 |   0.800119 |         0.912355 |        0.982517 |            0.928589 |
| s1_d_aux_init |     62 |          2 | hub_ws_iqr    |     17529 | 0.510609 | 0.724109 |  0.711903 |   0.726711 |         1.13897  |        1.30119  |            0.875334 |
| s1_d_aux_init |     62 |          3 | hub_ws_median |     15354 | 1.1835   | 1.597    |  0.900565 |   0.892089 |         5.84426  |        5.80778  |            1.00628  |
| s1_d_aux_init |     62 |          3 | hub_ws_mean   |     15354 | 1.13333  | 1.53005  |  0.904057 |   0.895284 |         5.7527   |        5.74513  |            1.00132  |
| s1_d_aux_init |     62 |          3 | hub_ws_std    |     15354 | 0.267301 | 0.359978 |  0.761744 |   0.772243 |         0.773763 |        0.784792 |            0.985947 |
| s1_d_aux_init |     62 |          3 | hub_ws_iqr    |     15354 | 0.432836 | 0.613925 |  0.623194 |   0.651545 |         0.827408 |        0.889793 |            0.929887 |

Cross-fitted features use only earlier-quarter Stage-1 outer predictions; 2022 early history uses a target-free GFS ws100 fallback with mask 0 and an explicit fallback indicator.
Exp02's simple SCADA auxiliary model has no hub-wind OOF prediction artifact under the same eight-quarter rolling keys, so a direct physical-metric comparison cannot be made honestly; no synthetic comparison was created.

## Stage 2 and transfer

| model_id          |   seed |   total_score |   one_minus_nmae |     ficr |   groups_available | is_official_three_group_score   |   evaluated_samples |   total_samples |   evaluated_rate |
|:------------------|-------:|--------------:|-----------------:|---------:|-------------------:|:--------------------------------|--------------------:|----------------:|-----------------:|
| s2_c_explicit     |     52 |      0.639808 |         0.871482 | 0.408135 |                  3 | True                            |               25515 |           43849 |         0.581883 |
| s2_c_explicit     |     62 |      0.638918 |         0.871251 | 0.406585 |                  3 | True                            |               25515 |           43849 |         0.581883 |
| s2_d_distribution |     42 |      0.638457 |         0.871442 | 0.405472 |                  3 | True                            |               25515 |           43849 |         0.581883 |
| s2_d_distribution |     52 |      0.638401 |         0.871922 | 0.40488  |                  3 | True                            |               25515 |           43849 |         0.581883 |
| s2_d_distribution |     62 |      0.637186 |         0.871679 | 0.402692 |                  3 | True                            |               25515 |           43849 |         0.581883 |
| s2_c_explicit     |     42 |      0.636559 |         0.871323 | 0.401795 |                  3 | True                            |               25515 |           43849 |         0.581883 |
| s2_b_pretrained   |     42 |      0.633695 |         0.871534 | 0.395857 |                  3 | True                            |               25515 |           43849 |         0.581883 |

Pretrained-only seed 42 scored 0.633695. Explicit median/mean scored 0.636559 at seed 42, and distribution/uncertainty scored 0.638457; thus explicit features added +0.002864 over pretrained-only and uncertainty added +0.001898 over explicit for that seed.
Joint fine-tuning was not executed because neither C nor D seed-42 rolling score exceeded Exp04 0.647440. Raw spatial attention remained frozen.
Best Exp08 seed scores: [{'model_id': 's2_d_distribution', 'seed': 42, 'total_score': 0.6384568490443155}, {'model_id': 's2_d_distribution', 'seed': 52, 'total_score': 0.6384013323340622}, {'model_id': 's2_d_distribution', 'seed': 62, 'total_score': 0.6371857259062735}]; mean=0.6380146357615505, improved seeds=0/3.

## Final decision

- Acceptance: **FAIL**
- Rolling aggregate: 0.6489012445865796 (Exp04 0.6474395993905896, delta=+0.001462)
- Equal-quarter mean / worst quarter: 0.6474887487032782 / 0.6011806185371973
- Maintained/improved quarters: 4/8; worst-quarter degradation: 0.0042822006598015605
- 1-NMAE / FICR: 0.8748174419034879 / 0.4229850472696714 (Exp04 0.8731518084215217 / 0.4217273903596575)
- Group 3: 0.6313110544649307 (Exp04 0.6288882169170712)
- January: 0.6308400046468222 (Exp04 0.6316773661490157); high-wind: 0.6885552835633567 (Exp04 0.6886224798253164)
- Exp04 residual correlation: 0.9474036886163416
- Best blend: `{'weight_exp03': 0.58, 'weight_exp04_raw': 0.175, 'weight_exp08': 0.245}`
| check                                   | actual                                   | passed   |
|:----------------------------------------|:-----------------------------------------|:---------|
| rolling_at_least_0_649440               | 0.6489012445865796                       | False    |
| improvement_at_least_0_002              | 0.0014616451959900134                    | False    |
| improved_quarters_at_least_6            | 4/8                                      | False    |
| worst_quarter_degradation_at_most_0_002 | 0.0042822006598015605                    | False    |
| ficr_maintained                         | 0.4229850472696714 vs 0.4217273903596575 | True     |
| one_minus_nmae_within_0_0005            | 0.8748174419034879 vs 0.8731518084215217 | True     |
| group_3_maintained                      | 0.6313110544649307 vs 0.6288882169170712 | True     |
| three_seed_mean_improves                | 0.6380146357615505                       | False    |
| not_single_seed_dependent               | 0/3                                      | False    |
- Submission: `[]`
- Full training allowed/executed: `False` / `False`; acceptance failed, so no full train or diagnostic submission was created.
- Persistent output: `/home/work/baram/Baram/experiments/exp08_scada_hubwind_pretraining/outputs`
- Drive: `None`
- Public submission priority: only accepted Exp08 model and accepted Exp04/Exp08 blend; no automatic submission.

## Next direction

Retain Exp04. The hub-wind representation is physically meaningful and helped seed-42 Stage2 ablations, but its residual correlation with Exp04 is too high and three-seed power scores regress. A next experiment should target lower-correlation site/regime information or improve temporal cross-fitted hub-wind calibration, while preserving the same leakage and rolling contracts.
