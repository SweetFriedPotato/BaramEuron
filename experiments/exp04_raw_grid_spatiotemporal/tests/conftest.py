from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]
BASELINE_SRC = ROOT / "baseline/src"
for value in (ROOT, BASELINE_SRC):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))
