from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from baram.constants import CAPACITY_KWH, TARGETS, TIME_COL
from experiments.exp02_daily_tcn_scada_aux.src.trainer import seed_everything
from experiments.exp03_official_score_calibration.src.backtest import ROLLING_QUARTERS
from experiments.exp03_official_score_calibration.src.evaluate import score_available_groups
from experiments.exp04_raw_grid_spatiotemporal.src.evaluate import prediction_frame
from experiments.exp08_scada_hubwind_pretraining.src.run_experiment import DataContext, crossfit_stage2_features
from experiments.exp08_scada_hubwind_pretraining.src.stage2_dataset import Stage2Dataset
from .conditional_distribution import deterministic_samples
from .expected_official_score import score_optimal_decision, shrink_decision
from .quantile_calibration import GroupConformalOffsets
from .quantile_loss import quantile_training_loss
from .quantile_power_model import VARIANT_INDICES, build_quantile_power_model

PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_DIR = PROJECT_ROOT / "experiments/exp09_probabilistic_score_decision"
DEFAULT_OUTPUT = EXPERIMENT_DIR / "outputs"
EXP08_OUTPUT = PROJECT_ROOT / "experiments/exp08_scada_hubwind_pretraining/outputs"
EXP04_OUTPUT = PROJECT_ROOT / "experiments/exp04_raw_grid_spatiotemporal/outputs"


def _move(batch, device): return [value.to(device, non_blocking=True) for value in batch]


def predict(model, inputs, hub, batch_size, device):
    loader=DataLoader(Stage2Dataset(inputs, hub),batch_size=batch_size,shuffle=False); out=[]; model.eval()
    with torch.no_grad():
        for batch in loader:
            ldaps,gfs,common,group,hubs=_move(batch[:5],device); values,_,_=model(ldaps,gfs,common,group,hubs); out.append(values.cpu().numpy())
    return np.concatenate(out)


def _inner_indices(timestamps):
    periods=pd.PeriodIndex(pd.DatetimeIndex(timestamps[:,0])-pd.Timedelta(hours=1),freq="Q")
    inner=periods.max(); valid=np.flatnonzero(periods==inner); train=np.flatnonzero(periods<inner)
    if not len(train) or not len(valid): raise ValueError("nested split needs earlier train and inner quarter")
    return train,valid,str(inner)


def train_outer(context, quarter, variant, seed, output):
    data=context.quarter(quarter); stage1=json.loads((EXP08_OUTPUT/"stage1_selection.json").read_text())["selected_model"]
    train_hub,valid_hub=crossfit_stage2_features(context,data,quarter,stage1,seed,EXP08_OUTPUT)
    ti,ii,inner=_inner_indices(data.raw.train_timestamps); device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(seed); model=build_quantile_power_model({"use_geo":True,"use_thermo":True,"use_engineered":True,"gated_fusion":True,
        "model":{"token_dim":64,"attention_heads":4,"attention_dropout":.1,"hidden_channels":128,"kernel_size":3,"dilations":[1,2,4,8],"temporal_dropout":.15,"non_causal":True}},data.raw,variant).to(device)
    dataset=Stage2Dataset(data.raw.train_inputs,train_hub,data.raw.train_y,data.raw.train_mask)
    loader=DataLoader(Subset(dataset,ti.tolist()),batch_size=16,shuffle=True,generator=torch.Generator().manual_seed(seed),pin_memory=device.type=="cuda")
    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4); scaler=torch.amp.GradScaler(device.type,enabled=device.type=="cuda")
    best=np.inf; state=None; best_epoch=0; stale=0
    inner_inputs=data.raw.train_inputs.subset(ii); inner_hub=train_hub[ii]; inner_y=data.raw.train_y[ii]; inner_mask=data.raw.train_mask[ii]
    for epoch in range(1,61):
        model.train()
        for batch in loader:
            ldaps,gfs,common,group,hubs,target,mask=_move([*batch[:5],batch[5],batch[6]],device); optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type,enabled=device.type=="cuda"):
                qp,_,_=model(ldaps,gfs,common,group,hubs); loss,_=quantile_training_loss(qp,target,mask)
            scaler.scale(loss).backward(); scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); scaler.step(optimizer); scaler.update()
        inner_pred=predict(model,inner_inputs,inner_hub,32,device)
        inner_loss=float(quantile_training_loss(torch.from_numpy(inner_pred),torch.from_numpy(inner_y),torch.from_numpy(inner_mask))[0])
        print(f"{quarter} {variant} seed={seed} epoch={epoch} inner={inner_loss:.6f}",flush=True)
        if inner_loss<best-1e-7: best,state,best_epoch,stale=inner_loss,copy.deepcopy(model.state_dict()),epoch,0
        else: stale+=1
        if stale>=8: break
    model.load_state_dict(state); checkpoint=output/f"checkpoints/{variant}/{seed}/{quarter}.pt"; checkpoint.parent.mkdir(parents=True,exist_ok=True)
    torch.save({"state_dict":state,"outer":quarter,"inner":inner,"best_epoch":best_epoch,"seed":seed},checkpoint)
    outer=predict(model,data.raw.valid_inputs,valid_hub,32,device)
    npz=output/f"predictions/{variant}/{seed}/{quarter}.npz"; npz.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(npz,quantiles=outer,target=data.raw.valid_y,mask=data.raw.valid_mask,timestamps=data.raw.valid_timestamps,
                        validation_wind=data.raw.validation_wind,high_wind_threshold=data.raw.high_wind_threshold,inner=inner)
    return npz


def phase_nested(output,seed):
    context=DataContext(output); variants=list(VARIANT_INDICES); paths=[]
    for variant in variants:
        for quarter in ROLLING_QUARTERS:
            path=output/f"predictions/{variant}/{seed}/{quarter}.npz"
            if not path.exists(): path=train_outer(context,quarter,variant,seed,output)
            paths.append(path)
    return {"seed":seed,"variants":variants,"predictions":[str(p) for p in paths]}


def main():
    p=argparse.ArgumentParser();p.add_argument("--seed",type=int,default=42,choices=[42,52,62]);p.add_argument("--output-root",type=Path,default=DEFAULT_OUTPUT);a=p.parse_args();a.output_root.mkdir(parents=True,exist_ok=True)
    result=phase_nested(a.output_root,a.seed);(a.output_root/f"nested_seed{a.seed}.json").write_text(json.dumps(result,indent=2))

if __name__=="__main__":main()
