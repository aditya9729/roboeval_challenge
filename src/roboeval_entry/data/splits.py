"""Dataset/demo split assignment.

Decision: demonstrations are split 90/10 train/dev by episode uuid with salt
"roboeval-entry-v1"; there are no holdout demos — holdout is seed-based only.
Seed pools live in seed_guard (train 100000-199999, dev 200000-200099,
holdout 300000-300099; official 1000-1049 reserved).
"""
from __future__ import annotations

import hashlib

from roboeval_entry.data.seed_guard import (  # noqa: F401
    OFFICIAL,
    POOLS,
    validate_lineage,
    validate_seeds,
)

DEMO_SPLIT_SALT = "roboeval-entry-v1"
DEV_FRACTION = 0.1


def assign_demo_split(uuid: str, salt: str = DEMO_SPLIT_SALT,
                      dev_fraction: float = DEV_FRACTION) -> str:
    """Deterministically map a demo uuid to 'train' or 'dev'."""
    digest = hashlib.sha256(f"{salt}:{uuid}".encode()).digest()
    value = int.from_bytes(digest[:8], "big") / 2**64
    return "dev" if value < dev_fraction else "train"
