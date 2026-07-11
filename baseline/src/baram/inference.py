"""Shared experiment orchestration for validation, full training, and inference."""
from pathlib import Path
import numpy as np, pandas as pd
from .constants import TARGETS,TIME_COL
from .data import load_labels,load_sample_submission
from .features.common import build_feature_tables
from .features.sequence import make_sequences
from .preprocessing import Preprocessor
from .models import SklearnModel,MLPNet,GRUNet
from .trainer import TorchRegressor,save_history
from .validation import time_split
from .metrics import detailed_metrics
from .submission import postprocess,create_submission
from .utils import run_dir,dump_json,dump_config

def _xy(features,labels,target):
    d=features.merge(labels.rename(columns={"kst_dtm":TIME_COL})[[TIME_COL,target]],on=TIME_COL,how="left",validate="one_to_one")
    return d,d.drop(columns=[TIME_COL,target]),d[target]
def _model(cfg,input_size):
    name=cfg["model"]["name"]; p=cfg["model"].get("params",{})
    if name in ("random_forest","extra_trees"): return SklearnModel(name,**p)
    net=MLPNet(input_size,p.get("hidden_dims",[128,64]),p.get("dropout",.1)) if name=="mlp" else GRUNet(input_size,p.get("hidden_size",64),p.get("num_layers",1),p.get("dropout",0),p.get("bidirectional",False))
    return TorchRegressor(net,{**p,"seed":cfg.get("seed",42)})

def validate(cfg):
    features,_=build_feature_tables(cfg); labels=load_labels(cfg); root=run_dir(cfg,cfg["model"]["name"]+"_validation"); dump_config(cfg,root/"config.yaml"); results={}; maes=[]; nmaes=[]
    for target in TARGETS:
        d,x,y=_xy(features,labels,target); tr,va=time_split(d[TIME_COL],target); tr&=y.notna().to_numpy(); va&=y.notna().to_numpy()
        prep=Preprocessor(scale=cfg["model"]["name"] in ("mlp","gru")); all_z=prep.fit_transform(x.loc[tr]); valid_z=prep.transform(x.loc[va]); train_y=y.loc[tr].to_numpy(); valid_y=y.loc[va].to_numpy(); valid_times=d.loc[va,TIME_COL]
        if cfg["model"]["name"]=="gru":
            seq=cfg["model"]["params"].get("sequence_length",24); zframe=pd.DataFrame(prep.transform(x),columns=x.columns); zframe.insert(0,TIME_COL,d[TIME_COL]); all_seq,all_times=make_sequences(zframe,d.loc[tr|va,TIME_COL],seq); lookup={t:i for i,t in enumerate(all_times)}; ti=[lookup[t] for t in d.loc[tr,TIME_COL] if t in lookup]; vi=[lookup[t] for t in d.loc[va,TIME_COL] if t in lookup]; ymap=pd.Series(y.to_numpy(),index=d[TIME_COL]); all_z=all_seq[ti]; valid_z=all_seq[vi]; train_y=ymap.loc[all_times[ti]].to_numpy(); valid_y=ymap.loc[all_times[vi]].to_numpy(); valid_times=all_times[vi]
        model=_model(cfg,all_z.shape[-1]); model.fit(all_z,train_y,valid_z,valid_y); raw=model.predict(valid_z); processed=postprocess(raw,target,cfg["postprocess"])
        results[target]={"train_samples":len(train_y),"valid_samples":len(valid_y),"raw":detailed_metrics(pd.Series(valid_times),valid_y,raw,target),"postprocessed":detailed_metrics(pd.Series(valid_times),valid_y,processed,target)}; maes.append(results[target]["postprocessed"]["mae"]); nmaes.append(results[target]["postprocessed"]["nmae"])
        target_dir=root/target; target_dir.mkdir(); prep.save(target_dir/"preprocessing.joblib"); model.save(target_dir/("model.joblib" if cfg["model"]["name"] in ("random_forest","extra_trees") else "model.pt")); dump_json(list(x.columns),target_dir/"feature_columns.json")
        if hasattr(model,"history"): save_history(model.history,target_dir/"history.csv")
    results["macro_mae"]=float(np.mean(maes)); results["macro_nmae"]=float(np.mean(nmaes)); dump_json(results,root/"metrics.json"); return root,results

def train_and_submit(cfg):
    train,test=build_feature_tables(cfg); labels=load_labels(cfg); sample=load_sample_submission(cfg); root=run_dir(cfg,cfg["model"]["name"]+"_full"); dump_config(cfg,root/"config.yaml"); predictions={}
    for target in TARGETS:
        d,x,y=_xy(train,labels,target); mask=y.notna(); prep=Preprocessor(scale=cfg["model"]["name"] in ("mlp","gru")); tr=prep.fit_transform(x.loc[mask]); te=prep.transform(test.drop(columns=TIME_COL)); train_y=y.loc[mask].to_numpy()
        if cfg["model"]["name"]=="gru":
            seq=cfg["model"]["params"].get("sequence_length",24); combined=pd.concat([train,test],ignore_index=True).sort_values(TIME_COL); z=prep.transform(combined.drop(columns=TIME_COL)); zf=pd.DataFrame(z,columns=x.columns); zf.insert(0,TIME_COL,combined[TIME_COL].to_numpy()); trseq,trtimes=make_sequences(zf,d.loc[mask,TIME_COL],seq); teseq,tetimes=make_sequences(zf,test[TIME_COL],seq); ymap=pd.Series(y.to_numpy(),index=d[TIME_COL]); tr,te,train_y=trseq,teseq,ymap.loc[trtimes].to_numpy()
            if not tetimes.equals(pd.DatetimeIndex(test[TIME_COL])): raise ValueError("test sequence context incomplete")
        model=_model(cfg,tr.shape[-1]); model.fit(tr,train_y); predictions[target]=postprocess(model.predict(te),target,cfg["postprocess"]); td=root/target; td.mkdir(); prep.save(td/"preprocessing.joblib"); model.save(td/("model.joblib" if cfg["model"]["name"] in ("random_forest","extra_trees") else "model.pt"))
    out=Path(cfg["output_root"])/"submissions"; out.mkdir(parents=True,exist_ok=True); path=out/f"{cfg['model']['name']}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"; create_submission(sample,predictions,path); return root,path

