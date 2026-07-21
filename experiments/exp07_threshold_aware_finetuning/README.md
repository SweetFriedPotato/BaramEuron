# Exp07 threshold-aware fine-tuning

Conservative nested rolling fine-tuning of the fixed Exp03 FICR-aware TCN and
Exp04 `raw_hybrid_gated` checkpoints.  Public leaderboard scores are report
context only and are never selection inputs.

The first rolling quarter falls back to the incumbent prediction because it has
no prior inner-validation quarter.  Every later outer quarter trains only on
earlier issue blocks, selects epochs and hyperparameters on the immediately
preceding quarter, and evaluates the outer quarter exactly once.

Neural stages require CUDA and are intended for the official Colab A100 session.
The contract and unit-test stages run locally:

```bash
python -m experiments.exp07_threshold_aware_finetuning.src.nested_finetune --phase contract
pytest -q
```

