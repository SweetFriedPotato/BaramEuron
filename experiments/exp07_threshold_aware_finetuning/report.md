# Exp07 threshold-aware fine-tuning report

## Outcome

Exp07 did **not** replace Exp04. The selected nested OOF blend is the unchanged
Exp03/raw 0.6/0.4 champion with Score `0.647439599391`
and delta `0.000000000000`. Full fine-tuning and submission
generation were therefore skipped by contract.

- Branch: `exp/07-threshold-aware-finetune`
- Evidence commit: `a06110b`
- Tests: `109 passed`
- A100 Drive run: `/content/drive/MyDrive/Baram/runs/exp07_threshold_aware_finetuning/20260721_121800`
- Public Score used for selection: no
- Public submission priority: none; Exp04 remains champion

## Reference contract

- Exp04 reproduced Score: `0.647439599391`
- Absolute reproduction error: `0.000000000000` (tolerance `1e-8`)
- Official scorer/checkpoint/preprocessing contracts: passed
- Random split: no; outer target used for selection: no

## Fine-tuning selection

- Exp03 head-only: `fixed_006_lambda_005`
- Raw head-only: `annealed_015_004_lambda_005`
- Temperature: Exp03 fixed `tau=0.006`; raw cosine `0.015→0.004`
- Boundary: symmetric, detached, `sigma=0.006`; selected `lambda=0.05`
- Exp03 last-block: `last_block_fixed_006_lambda_005`;
  inner Score `0.679442` →
  `0.682674`
- Raw last-block: rejected; inner Score
  `0.678809` →
  `0.676625`

| Component | Original Score | Fine-tuned Score | Delta |
|---|---:|---:|---:|
| Exp03 | 0.642604 | 0.626783 | -0.015821 |
| raw | 0.641179 | 0.633767 | -0.007412 |

## Final nested evaluation

- Best combination: `A_original_exp03_original_raw`
- Raw weight: `0.400`
- Rolling aggregate: `0.647439599391`
- Equal-quarter mean: `0.646748`
- Worst quarter: `0.605463`
- Maintained/improved quarters: `8/8`
- Quarter delta range: `0.000000` to
  `0.000000`
- 1-NMAE: `0.873152`
- FICR: `0.421727`
- Group 3: `0.628888`
- January Score: `0.631677`
- High-wind Score: `0.688622`
- 3-seed mean/std: `0.647440` / `0.000000`
- Improved seeds: `0/3`

## Threshold diagnostics

- Rescue transitions: `0`
- Loss transitions: `0`
- Rescue gain: `+0`
- near_6pct: tier-4 0.517323 → 0.517323, rewarded 1.000000 → 1.000000
- near_8pct: tier-4 0.000000 → 0.000000, rewarded 0.532343 → 0.532343

The zero transition counts are expected because final blend selection returned the
unchanged Exp04 prediction, even though its component candidates were fully evaluated.

## Clipping and acceptance

- Best clipping diagnostic: `eda_observed_upper`
  (`0.647651`)
- Acceptance: `FAIL`
- minimum_score: FAIL
- minimum_delta: FAIL
- improved_quarters: PASS
- worst_quarter: PASS
- ficr: FAIL
- one_minus_nmae: PASS
- group3: PASS
- rescue_gain: FAIL
- seed_mean: FAIL

No accepted submission was created and nothing was submitted automatically.

## Interpretation and next direction

The loss improved inner validation for both heads and justified Exp03 last-block
testing, but the gains did not survive outer-quarter/seed evaluation. The deployable
global blend search consequently returned the exact incumbent. The next experiment
should target regime information that is available at inference time—especially
forecasted wind-distribution and ramp features—rather than increasing threshold-loss
strength or model unfreezing.

## Figures

- `figures/training_curves.png`
- `figures/threshold_transition_heatmap.png`
- `figures/boundary_rescue_vs_loss.png`
- `figures/quarter_score_comparison.png`
- `figures/component_comparison.png`
- `figures/blend_search.png`
- `figures/nmae_ficr_tradeoff.png`
- `figures/final_score_comparison.png`
