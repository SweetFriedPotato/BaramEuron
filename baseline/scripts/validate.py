#!/usr/bin/env python3
import argparse,sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from baram.config import load_config
from baram.inference import validate
p=argparse.ArgumentParser(); p.add_argument("--config",required=True); a=p.parse_args(); root,result=validate(load_config(a.config)); print(root); print(json.dumps(result,ensure_ascii=False,indent=2))

