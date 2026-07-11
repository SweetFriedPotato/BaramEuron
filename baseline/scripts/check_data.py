#!/usr/bin/env python3
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from baram.config import load_config
from baram.data import load_ldaps,load_gfs,load_labels,load_sample_submission,validate_periods
from baram.constants import TIME_COL
p=argparse.ArgumentParser(); p.add_argument("--config",required=True); a=p.parse_args(); cfg=load_config(a.config)
validate_periods(cfg); result={"label_semantics":"kst_dtm is the end of the one-hour generation interval and matches forecast_kst_dtm","official_baseline_alignment":"direct equality merge between label kst_dtm and forecast_kst_dtm"}
for split in ("train","test"):
    result[split]={}
    for kind,loader in (("ldaps",load_ldaps),("gfs",load_gfs)):
        d=loader(split,cfg); lead=(d[TIME_COL]-d["data_available_kst_dtm"]).dt.total_seconds()/3600
        result[split][kind]={"forecast_times":int(d[TIME_COL].nunique()),"lead_time_h":{"min":float(lead.min()),"max":float(lead.max()),"unique":[float(x) for x in sorted(lead.unique())]}}
result["labels"]={"rows":len(load_labels(cfg))}; result["sample_submission"]={"rows":len(load_sample_submission(cfg))}
out=Path(cfg["output_root"])/"checks"/"time_semantics.json"; out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(result,ensure_ascii=False,indent=2)); print(out); print(json.dumps(result,ensure_ascii=False,indent=2))

