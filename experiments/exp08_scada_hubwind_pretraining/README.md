# Exp08 — SCADA-supervised hub-wind pretraining

This experiment learns actual site hub-wind distributions from the Exp04 LDAPS/GFS raw-grid inputs, then feeds strictly cross-fitted predictions into an Exp04 power model. SCADA is a training target only. Test inference never reads a SCADA file.

## Fixed contracts

- Groups: VESTAS 1–6 → group 1, VESTAS 7–12 → group 2, UNISON 1–5 → group 3.
- SCADA alignment: right-closed hour ending via `timestamp.ceil("h")`.
- Cleaning: non-finite/negative invalid; source-specific q0.001/q0.999 learned on fold training only.
- Stage 1 output: median, mean, `log1p(std)`, `log1p(IQR)` with group-balanced masked SmoothL1.
- Stage 2 power loss: the unchanged Exp03 official-mask normalized MAE plus `0.20 ×` soft FICR; joint retention is `0.05 ×`.
- Validation: the exact Exp04 expanding 2023Q1–2024Q4 protocol. Public scores are report context only.
- Full training and submissions are gated by the acceptance checks in the request.

## Commands

```bash
PYTHONPATH=baseline/src python -m experiments.exp08_scada_hubwind_pretraining.src.run_experiment --phase contracts
PYTHONPATH=baseline/src python -m experiments.exp08_scada_hubwind_pretraining.src.run_experiment --phase smoke
PYTHONPATH=baseline/src python -m experiments.exp08_scada_hubwind_pretraining.src.run_experiment --phase stage1 --seed 42
PYTHONPATH=baseline/src python -m experiments.exp08_scada_hubwind_pretraining.src.run_experiment --phase stage2 --seed 42
PYTHONPATH=baseline/src python -m experiments.exp08_scada_hubwind_pretraining.src.run_experiment --phase finalize
```

On Colab, pass `--drive-run /content/drive/MyDrive/Baram/runs/exp08_scada_hubwind_pretraining/<run_id>`. Every completed quarter is copied there and can be restored by rerunning the same command.

Generated `outputs/` artifacts are intentionally ignored by Git.
