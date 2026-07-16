import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
for path in (ROOT, ROOT / "baseline" / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


@pytest.fixture(scope="session")
def project_root():
    return ROOT
