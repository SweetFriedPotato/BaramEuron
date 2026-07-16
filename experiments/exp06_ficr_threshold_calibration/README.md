# exp06 FICR threshold calibration

Exp06 audits the 6%/8% official-settlement boundaries and tests one deployable post-hoc rule at a time. It reuses Exp05's exact rolling OOF contract, never fits on full/test targets, and never uses Public scores for selection.

```bash
PYTHONPATH=baseline/src:. python -m experiments.exp06_ficr_threshold_calibration.src.evaluate
PYTHONPATH=baseline/src:. pytest -q
```

Exp04 remains champion unless the documented rolling, group-3, worst-quarter, and stability gates all pass.
