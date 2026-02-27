# pipeline/config.py

import os
from pathlib import Path

# Root is assumed to be project_root where this file lives under pipeline/
ROOT_DIR = Path(__file__).resolve().parents[1]

ARTIFACTS_DIR = ROOT_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

ENTRY_LOG_PATH = ARTIFACTS_DIR / "entry_log.parquet"

# Forward return horizons in trading days
FORWARD_HORIZONS = [5, 10, 15, 20, 30, 60]