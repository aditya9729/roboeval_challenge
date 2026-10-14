"""Backward-compatible re-export; implementation lives in the package."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from roboeval_entry.data.seed_guard import (  # noqa: E402,F401
    OFFICIAL,
    POOLS,
    validate_lineage,
    validate_seeds,
)
