# exp04 raw-grid spatiotemporal report

## Contract

LDAPS dynamic tensor is `[1096, 24, 16, 16]` and GFS is `[1096, 24, 9, 26]` before variant channel selection. Static group tensors are `[3, 16, 11]` and `[3, 9, 11]`. Dynamic imputation, clipping, and scaling were fit on each fold's training blocks only. SCADA remained an auxiliary target and was never included in model input.

## B-F ablation

- `raw_hybrid_gated`: Score 0.644135, 1-NMAE 0.873162, FICR 0.415109
- `raw_hybrid`: Score 0.642093, 1-NMAE 0.874100, FICR 0.410087
- `raw_wind_geo`: Score 0.638939, 1-NMAE 0.873133, FICR 0.404746
- `raw_wind_thermo`: Score 0.638143, 1-NMAE 0.871819, FICR 0.404466
- `raw_wind`: Score 0.633678, 1-NMAE 0.873380, FICR 0.393976

Selected architecture: `raw_hybrid_gated`.

## Official validation

- Fold A raw: Score 0.633518, 1-NMAE 0.864631, FICR 0.402406
- Fold A Exp03/raw blend: Score 0.636287, 1-NMAE 0.868775, FICR 0.403800
- Fold B raw: Score 0.646584, 1-NMAE 0.875274, FICR 0.417893
- Fold B Exp03/raw blend: Score 0.650288, 1-NMAE 0.877061, FICR 0.423515
- Fold B Exp03: Score 0.647595, 1-NMAE 0.875861, FICR 0.419328

The selected raw seed Score mean/std is 0.643268/0.001431. Exp03's Public Score 0.631535, 1-NMAE 0.865998, and FICR 0.397072 were report context only and were not used for selection.

## Rolling and ensemble

Raw rolling mean Score is 0.640285; worst quarter is 0.612639; it improves 3/8 quarters. Residual Pearson versus Exp03 is 0.921821. The best global blend uses raw weight 0.400 and reaches rolling aggregate Score 0.647440, a +0.004836 gain over Exp03-only. Its equal-quarter mean/worst are 0.646748/0.605463, and it improves 7/8 individual quarters.

## Slice results

- Group 1: blend 0.668600, raw 0.665964, Exp03 0.662744
- Group 2: blend 0.665079, raw 0.660987, Exp03 0.661661
- Group 3: blend 0.617185, raw 0.612800, Exp03 0.618378
- January: blend 0.651009, raw 0.655607, Exp03 0.645052
- High-wind: blend 0.682865, raw 0.680193, Exp03 0.682919

## Attention

- LDAPS top grids: G1→11 (0.115), G2→5 (0.122), G3→8 (0.091)
- GFS top grids: all groups→5 (0.161/0.164/0.169)
- LDAPS grid 13: G1 rank 14, G2 rank 14, G3 rank 3
- Mean LDAPS gate: 0.477; lead 12-19h 0.430, 20-27h 0.558, 28-35h 0.443

Attention is interpretation-only and was not used as a selection metric.

## Full train and submissions

Full training used 15 epochs and seeds 42/52/62 on A100. Generated submissions:

- `exp04_exp03_raw_blend_20260716_081002.csv`
- `exp04_raw_grid_20260716_081002.csv`

Submission priority is the Exp03/raw blend, followed by raw-only. No submission was sent automatically.
