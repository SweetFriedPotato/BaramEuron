from pathlib import Path
from datetime import datetime
import json, yaml
def run_dir(cfg,suffix=""):
    rid=datetime.now().strftime("%Y%m%d_%H%M%S")+(f"_{suffix}" if suffix else ""); p=Path(cfg["output_root"])/"runs"/rid; p.mkdir(parents=True,exist_ok=True); return p
def dump_json(data,path): Path(path).write_text(json.dumps(data,ensure_ascii=False,indent=2,default=str))
def dump_config(cfg,path):
    clean={k:v for k,v in cfg.items() if not k.startswith("_")}; Path(path).write_text(yaml.safe_dump(clean,sort_keys=False,allow_unicode=True))

