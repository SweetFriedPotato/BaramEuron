from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from baram.constants import CAPACITY_KWH, TARGETS, TIME_COL
from experiments.exp03_official_score_calibration.src.evaluate import score_available_groups
from .conditional_distribution import deterministic_samples
from .expected_official_score import score_optimal_decision
from .quantile_calibration import GroupConformalOffsets

PROJECT_ROOT=Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT=PROJECT_ROOT/"experiments/exp09_probabilistic_score_decision/outputs"
EXP04=PROJECT_ROOT/"experiments/exp04_raw_grid_spatiotemporal/outputs/predictions/best_blend_predictions.csv"
QUARTERS=[f"{y}Q{q}" for y in (2023,2024) for q in (1,2,3,4)]
VARIANTS=("q_a_exp04","q_b_hubwind","q_c_calibrated")


def _reference_lookup():
    frame=pd.read_csv(EXP04,parse_dates=[TIME_COL]); capacity=frame["target"].map(CAPACITY_KWH)
    return {(r.fold,pd.Timestamp(getattr(r,TIME_COL)),int(r.group_id)):float(r.y_pred_kwh/CAPACITY_KWH[r.target]) for r in frame.itertuples()}


def _frame(times,target,mask,prediction,model,quarter,seed):
    parts=[]
    for g,name in enumerate(TARGETS):
        valid=mask[...,g].reshape(-1); cap=CAPACITY_KWH[name]
        parts.append(pd.DataFrame({"fold":quarter,"quarter":quarter,TIME_COL:times.reshape(-1)[valid],"target":name,"group_id":g+1,
            "y_true_kwh":target[...,g].reshape(-1)[valid]*cap,"y_pred_kwh":np.maximum(prediction[...,g].reshape(-1)[valid],0)*cap,
            "model_id":model,"seed":seed}))
    return pd.concat(parts,ignore_index=True)


def build_variant(output:Path,variant:str,seed:int,reference:dict):
    history_q=[];history_y=[];history_m=[];frames=[];calibration=[]; alpha_history=[]
    for quarter in QUARTERS:
        data=np.load(output/f"predictions/{variant}/{seed}/{quarter}.npz"); q=data["quantiles"]; y=data["target"]; mask=data["mask"].astype(bool); times=data["timestamps"]
        if history_q:
            calibrator=GroupConformalOffsets().fit(np.concatenate(history_q),np.concatenate(history_y),np.concatenate(history_m))
            calibrated=calibrator.transform(q); offsets=calibrator.offsets.tolist()
        else: calibrated=q; offsets=np.zeros((3,11)).tolist()
        q50=calibrated[...,5]; mean=deterministic_samples(calibrated).mean(axis=-1); decision=np.empty_like(q50)
        refs=np.empty_like(q50)
        for i in range(q50.shape[0]):
            for h in range(24):
                for g in range(3):
                    ref=reference.get((quarter,pd.Timestamp(times[i,h]),g+1),float(q50[i,h,g]));refs[i,h,g]=ref
                    samples=deterministic_samples(calibrated[i,h,g])
                    try: decision[i,h,g]=score_optimal_decision(samples,calibrated[i,h,g],ref)["prediction"]
                    except ValueError: decision[i,h,g]=q50[i,h,g]
        candidates={a:a*decision+(1-a)*refs for a in (.25,.5,.75,1.0)}
        if alpha_history:
            scored={a:score_available_groups(pd.concat([x[a] for x in alpha_history],ignore_index=True))[0]["total_score"] for a in candidates}
            alpha=max(scored,key=scored.get)
        else: alpha=.25
        alpha_frames={a:_frame(times,y,mask,p,f"{variant}_shrink_{a}",quarter,seed) for a,p in candidates.items()}
        alpha_history.append(alpha_frames)
        frames.extend([_frame(times,y,mask,q50,f"{variant}_q50",quarter,seed),
                       _frame(times,y,mask,mean,f"{variant}_mean",quarter,seed),
                       _frame(times,y,mask,decision,f"{variant}_decision",quarter,seed),
                       alpha_frames[alpha].assign(model_id=f"{variant}_nested_shrink",selected_alpha=alpha)])
        calibration.append({"variant":variant,"seed":seed,"quarter":quarter,"history_quarters":len(history_q),"offsets":json.dumps(offsets),
                            "interval_90_coverage":float(((y>=calibrated[...,0])&(y<=calibrated[...,-1])&mask).sum()/max(mask.sum(),1)),
                            "selected_alpha":alpha})
        history_q.append(q);history_y.append(y);history_m.append(mask)
    return pd.concat(frames,ignore_index=True),pd.DataFrame(calibration)


def finalize(output:Path,seed:int):
    ref=_reference_lookup(); frames=[]; calibration=[]
    for variant in VARIANTS:
        frame,cal=build_variant(output,variant,seed,ref);frames.append(frame);calibration.append(cal)
    predictions=pd.concat(frames,ignore_index=True); metrics=[]
    for model,part in predictions.groupby("model_id"):
        summary,_=score_available_groups(part); quarters=[]
        for quarter,qpart in part.groupby("quarter"): quarters.append(score_available_groups(qpart)[0]["total_score"])
        metrics.append({"model_id":model,"seed":seed,**summary,"equal_quarter_mean":np.mean(quarters),"worst_quarter":np.min(quarters)})
    (output/"predictions").mkdir(exist_ok=True);(output/"metrics").mkdir(exist_ok=True)
    predictions.to_csv(output/f"predictions/decision_predictions_seed{seed}.csv",index=False)
    pd.DataFrame(metrics).sort_values("total_score",ascending=False).to_csv(output/f"metrics/candidate_scores_seed{seed}.csv",index=False)
    pd.concat(calibration,ignore_index=True).to_csv(output/f"metrics/quantile_calibration_seed{seed}.csv",index=False)
    return metrics


def main():
    p=argparse.ArgumentParser();p.add_argument("--seed",type=int,default=42);p.add_argument("--output-root",type=Path,default=DEFAULT_OUTPUT);a=p.parse_args();print(json.dumps(finalize(a.output_root,a.seed),indent=2,default=str))
if __name__=="__main__":main()
