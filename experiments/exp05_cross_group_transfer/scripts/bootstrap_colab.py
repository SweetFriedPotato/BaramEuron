"""Restore an Exp05 Colab VM after Drive has been mounted."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


REPOSITORY_URL = "https://github.com/SweetFriedPotato/BaramEuron.git"
BRANCH = "exp/05-cross-group-transfer"


def run(command: list[str], cwd: Path | None = None) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path("/content/Baram"))
    parser.add_argument("--drive-root", type=Path, default=Path("/content/drive/MyDrive/Baram"))
    args = parser.parse_args()
    if not (args.drive_root / "cache").is_dir():
        raise RuntimeError("mount Google Drive before running this bootstrap")
    if not args.repo.exists():
        run(["git", "clone", REPOSITORY_URL, str(args.repo)])
    run(["git", "fetch", "origin"], args.repo)
    run(["git", "switch", BRANCH], args.repo)
    run(["git", "pull", "--ff-only", "origin", BRANCH], args.repo)
    raw_archive = args.drive_root / "cache/baram_open.tar.gz"
    if not (args.repo / "open").is_dir():
        run(["tar", "-xzf", str(raw_archive), "-C", str(args.repo)])
    feature_archive = args.drive_root / "cache/baseline_features_3ea4bc4.tar.gz"
    if not (args.repo / "baseline/cache/features").is_dir():
        run(["tar", "-xzf", str(feature_archive), "-C", str(args.repo)])
    print(f"restored {BRANCH} at {args.repo}")
    print(f"raw cache: {raw_archive} ({os.path.getsize(raw_archive)} bytes)")


if __name__ == "__main__":
    main()
