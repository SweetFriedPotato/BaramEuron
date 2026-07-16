# exp05 cross-group transfer v2 report

## Contract

Exp04 0.4 reference reproduced at `0.647439599391` (absolute error `0`). Stacker training used rolling OOF rows only; Public metrics were not used for selection.

## Cheap stages

- `exp04_global`: Score 0.647440, 1-NMAE 0.873152, FICR 0.421727, equal-quarter 0.646748, worst 0.605463, improved 8/8
- `constrained`: Score 0.645573, 1-NMAE 0.872831, FICR 0.418315, equal-quarter 0.645506, worst 0.605261, improved 3/8
- `ridge`: Score 0.644599, 1-NMAE 0.874470, FICR 0.414729, equal-quarter 0.645137, worst 0.603093, improved 5/8
- `catboost`: Score 0.644473, 1-NMAE 0.874197, FICR 0.414749, equal-quarter 0.644899, worst 0.601543, improved 5/8

## Constrained weights

Final all-OOF raw weights: g1 `0.40`, g2 `0.41`, g3 `0.29`. Selected penalties: `{'lambda_global': 0.002, 'lambda_spread': 0.001, 'lambda_instability': 0.001}`.
Nested mean/std: g1 0.3313/0.0745, g2 0.4287/0.0694, g3 0.4625/0.1639.

Quarter weights:

- 2023Q1: 0.40/0.40/0.40 (fallback_no_history)
- 2023Q2: 0.26/0.51/0.40 (default_penalty_insufficient_inner_quarters)
- 2023Q3: 0.25/0.34/0.39 (nested_selected)
- 2023Q4: 0.25/0.34/0.39 (nested_selected)
- 2024Q1: 0.29/0.53/0.40 (nested_selected)
- 2024Q2: 0.40/0.43/0.78 (nested_selected)
- 2024Q3: 0.40/0.44/0.65 (nested_selected)
- 2024Q4: 0.40/0.44/0.29 (nested_selected)

## Stage D

- smoke seed 42: Fold B Score 0.628811, delta vs raw seed42 -0.015324
- full seed 42: Fold B Score 0.641451, delta vs raw seed42 -0.002684
Full seed 42 did not beat raw_hybrid_gated; seeds 52/62 and conditional cross-group regularization were therefore skipped.

## Final

Best new rolling candidate: `final_ensemble` with Score 0.647673, equal-quarter mean 0.647083, worst 0.604469, and 4/8 improved quarters.
Group Scores: g1 0.653124, g2 0.661942, g3 0.627952.
Matched rolling group deltas vs Exp04: g1 +0.000166, g2 +0.001469, g3 -0.000936.
The supplied report-context group-3 reference 0.617185 is exceeded by +0.010767, but the matched rolling Exp04 group-3 score is not preserved.
January Score 0.632996; high-wind Score 0.689706.
Final convex weights: global_blend_prediction=0.80, constrained_prediction=0.00, ridge_prediction=0.20.
Acceptance: `failed`. Exp04 remains champion; Exp05 submissions are diagnostic only.

Generated submissions:

- `exp05_constrained_group_blend_20260716_185929.csv`
- `exp05_ridge_stacker_20260716_185929.csv`
- `exp05_final_ensemble_20260716_185929.csv`

No submission was sent automatically. Public priority remains the existing Exp04 blend. If one diagnostic slot is intentionally used, the Exp05 order is final ensemble, constrained blend, then Ridge.

## Public context and next step

Exp04 Public Score 0.634005715 and Exp03 Public Score 0.6315350794 are report context only and were not used for fitting or selection.
The next experiment should target FICR threshold calibration within each group using the same nested OOF contract; a larger cross-group neural model is not supported by this result.
