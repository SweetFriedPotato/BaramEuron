# AI NEXUS migration and operation

## Current workspace

- SSH alias: `bai-vscode`
- Persistent repository: `/home/work/baram/Baram`
- Persistent extra Python packages: `/home/work/baram/python_packages`
- GPU: NVIDIA A100-SXM4 80GB
- Branch: `exp/08-scada-hubwind-pretraining`

`/home/work` is deleted with the compute session. Keep every repository, dataset,
checkpoint, log, and package under the mounted `/home/work/baram` directory.

## Connect

Terminal:

```bash
ssh bai-vscode
```

VS Code:

1. Install the `Remote - SSH` extension.
2. Run `Remote-SSH: Connect to Host...` from the command palette.
3. Select `bai-vscode`.
4. Open `/home/work/baram/Baram`.

The local SSH config points at `~/.ssh/ainexus_euram_container`; do not copy that
private key into the repository.

## Activate the project environment

```bash
cd /home/work/baram/Baram
source scripts/ainexus_env.sh
python3 -c 'import torch; print(torch.cuda.get_device_name(0))'
```

The server-provided CUDA PyTorch is reused. Project-only packages are installed
under `/home/work/baram/python_packages`, because the image does not include the
`python3-venv` package.

## Verify

```bash
source /home/work/baram/Baram/scripts/ainexus_env.sh
python3 -m pytest -q
python3 -m experiments.exp08_scada_hubwind_pretraining.src.run_experiment \
  --phase contracts
```

## Run Exp08 safely after disconnecting SSH

Use `nohup`; this image does not provide tmux or screen. All outputs and logs stay
on the persistent mount.

```bash
source /home/work/baram/Baram/scripts/ainexus_env.sh
mkdir -p experiments/exp08_scada_hubwind_pretraining/outputs/logs
nohup python3 -u -m experiments.exp08_scada_hubwind_pretraining.src.run_experiment \
  --phase stage1 --seed 42 \
  > experiments/exp08_scada_hubwind_pretraining/outputs/logs/stage1_seed42.log 2>&1 &
echo $! > experiments/exp08_scada_hubwind_pretraining/outputs/logs/stage1_seed42.pid
```

Monitor:

```bash
tail -f experiments/exp08_scada_hubwind_pretraining/outputs/logs/stage1_seed42.log
nvidia-smi
```

Each rolling quarter writes its own checkpoint and prediction before the next
quarter starts, so rerunning the same command resumes completed work.

To run every remaining seed and phase sequentially:

```bash
source /home/work/baram/Baram/scripts/ainexus_env.sh
nohup scripts/run_exp08_ainexus_pipeline.sh \
  > experiments/exp08_scada_hubwind_pretraining/outputs/logs/pipeline.log 2>&1 < /dev/null &
```

The pipeline stops on the first failed phase and writes one `.exit` file per
phase. A successful run ends with `outputs/logs/pipeline.complete`.

## Sync changes

Code changes should move through Git. Large ignored artifacts can be resumed with
rsync:

```bash
rsync -a --partial \
  -e 'ssh -i ~/.ssh/ainexus_euram_container -p 10560' \
  LOCAL_PATH/ bai-vscode:/home/work/baram/Baram/REMOTE_PATH/
```

Never commit or send passwords, private keys, SCADA source data, checkpoints, or
competition data to GitHub.

## Security follow-up

The delivered Jupyter endpoint currently exposes the home-directory contents
without an authentication challenge. Ask the Cloud team to add authentication or
access control and rotate the SSH password/private key after confirming access.
