# exp06 FICR threshold calibration report

## Contract

Exp04 global blend was reproduced at `0.647439599391` with error below `1e-8`. The official scorer hash and exact 6%/8% reward aggregation matched. Public results were not used.

## Threshold audit

- `exp05_ridge`: 4→3 749, 4→0 218, 3→4 886, 0→4 229, energy-weighted reward delta -5387136
- `exp05_catboost`: 4→3 706, 4→0 167, 3→4 808, 0→4 147, energy-weighted reward delta -5307154
- `exp05_final`: 4→3 165, 4→0 0, 3→4 225, 0→4 0, energy-weighted reward delta +593814

Exp04 boundary samples: 6% ±0.5pp `1270`, 8% ±0.5pp `1144`.

## Oracle and regimes

- `exp04_global`: Score 0.647440, headroom +0.000000
- `sample_nmae_oracle`: Score 0.714499, headroom +0.067059
- `sample_ficr_oracle`: Score 0.714499, headroom +0.067059
- `deployable_regime`: Score 0.638408, headroom -0.009032

Stable model-win regimes meeting the diagnostic support/stability rule: `0`.

## Piecewise affine

Selected band scheme: `quantile_three`; penalty `{'identity': 0.0005, 'smoothness': 0.0005, 'instability': 0.0005}`.
Final predicted-CF boundaries: `{'kpx_group_1': [0.19739819938753855, 0.48675521436149694], 'kpx_group_2': [0.1934404025607639, 0.5419135380497685], 'kpx_group_3': [0.16554901413690476, 0.38213604290674597]}`.
- kpx_group_1 bin 0: scale 0.9750, offset +0.0050 capacity
- kpx_group_1 bin 1: scale 1.0275, offset -0.0200 capacity
- kpx_group_1 bin 2: scale 0.9925, offset +0.0200 capacity
- kpx_group_2 bin 0: scale 0.9775, offset +0.0075 capacity
- kpx_group_2 bin 1: scale 0.9900, offset -0.0200 capacity
- kpx_group_2 bin 2: scale 0.9750, offset +0.0150 capacity
- kpx_group_3 bin 0: scale 0.9825, offset +0.0200 capacity
- kpx_group_3 bin 1: scale 0.9725, offset -0.0150 capacity
- kpx_group_3 bin 2: scale 1.0250, offset +0.0187 capacity

Score 0.648145 (+0.000706), 1-NMAE 0.870997, FICR 0.425294, equal-quarter 0.647468, worst 0.605191, improved 5/8.
Group Scores: g1 0.653237, g2 0.660703, g3 0.630496.
Matched group-3 delta vs Exp04: +0.001607.
FICR delta +0.003566; 1-NMAE delta -0.002155. Change p95 `0.030682` capacity.
Nested parameter std mean/max: scale 0.0126/0.0182, offset 0.0098/0.0171.
January Score 0.630129; high-wind Score 0.689214.
Piecewise acceptance conditions: `{'aggregate': False, 'improved_quarters': False, 'worst_quarter': True, 'group3': True, 'ficr': True, 'nmae': False, 'change_p95': False}`.

## Gate and final decision

Gate executed: `False`. Gate acceptance: `{'accepted': False, 'reason': 'oracle headroom below 0.003'}`.
Selected deployable rule: `exp04_global`; accepted new rule: `False`.

Submissions:

- `exp06_piecewise_20260716_194218.csv`

Diagnostic only: `True`. No submission was sent automatically.

## Next direction

If threshold calibration does not clear acceptance, retain Exp04 and focus on training-time FICR threshold robustness rather than another residual stacker or larger cross-group model.
