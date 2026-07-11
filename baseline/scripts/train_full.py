#!/usr/bin/env python3
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from baram.config import load_config
from baram.inference import train_and_submit
p=argparse.ArgumentParser(); p.add_argument("--config",required=True); a=p.parse_args(); root,submission=train_and_submit(load_config(a.config)); print(f"run={root}\nsubmission={submission}")

