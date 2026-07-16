# exp04 raw-grid spatiotemporal

This experiment consumes the 16 LDAPS and 9 GFS grids without spatial averaging. A group-query cross-attention encoder combines dynamic wind/thermodynamic fields with deterministic geographic priors, then applies the same non-causal daily TCN and official-score-aware loss used by exp03.

The experiment never uses SCADA, power targets, or target lags as model inputs. SCADA wind is retained only as a masked auxiliary target. Every dynamic preprocessing state is fit on the training portion of a fold.

## Variants

- `raw_wind`: wind vectors and content attention.
- `raw_wind_geo`: adds group-relative position, distance, height, and a learnable positive attention penalty.
- `raw_wind_thermo`: adds the prescribed dense thermodynamic channels.
- `raw_hybrid`: adds exp03 engineered context without forecast-disagreement features.
- `raw_hybrid_gated`: adds lead-time-aware LDAPS/GFS fusion.

Run local tests before launching the A100 workflow:

```bash
PYTHONPATH=baseline/src:. python -m pytest experiments/exp04_raw_grid_spatiotemporal/tests -q
PYTHONPATH=baseline/src:. python -m experiments.exp04_raw_grid_spatiotemporal.src.run_experiment --config-dir experiments/exp04_raw_grid_spatiotemporal/configs
```

Generated tensors, checkpoints, predictions, attention dumps, submissions, and other outputs are ignored by Git.
