# exp05 cross-group transfer v2 report

## Contract

Exp04 0.4 reference reproduced at `0.647439599391` (absolute error `0`). Stacker training used rolling OOF rows only; Public metrics were not used for selection.

## Cheap stages

- `exp04_global`: Score 0.647440, 1-NMAE 0.873152, FICR 0.421727
- `constrained`: Score 0.645573, 1-NMAE 0.872831, FICR 0.418315
- `ridge`: Score 0.644599, 1-NMAE 0.874470, FICR 0.414729
- `catboost`: Score 0.644473, 1-NMAE 0.874197, FICR 0.414749

Final constrained all-OOF raw weights are recorded in `constrained_group_summary.json`. Nested quarter weight standard deviations: g1 0.0745, g2 0.0694, g3 0.1639.

## Stage D

Stage D required: `True`; result not yet present.

## Final

Best rolling candidate: `final_ensemble` with Score 0.647673, equal-quarter mean 0.647083, worst 0.604469, and 4/8 improved quarters.

Generated submissions:

- `exp05_constrained_group_blend_20260716_180604.csv`
- `exp05_ridge_stacker_20260716_180604.csv`
- `exp05_final_ensemble_20260716_180604.csv`

No submission was sent automatically.
