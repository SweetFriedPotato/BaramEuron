import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
BASELINE_SRC = PROJECT_ROOT / "baseline" / "src"
for path in (PROJECT_ROOT, BASELINE_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
