"""Conservative split checks for local RoboEval experiments.

These checks protect seed identifiers, not all forms of data leakage. The full
implementation must also group demonstration descendants and inspect configs.
"""
from __future__ import annotations

from collections.abc import Iterable

POOLS: dict[str, tuple[int, int]] = {
    "train": (100000, 199999),
    "dev": (200000, 200099),
    "holdout": (300000, 300099),
}
OFFICIAL = (1000, 1049)


def validate_seeds(seeds: Iterable[int], split: str) -> list[int]:
    """Return a validated nonempty unique list; reject reserved/out-of-pool IDs."""
    if split not in POOLS:
        raise ValueError(f"Unknown local split: {split!r}")
    values = list(seeds)
    if not values:
        raise ValueError("Seed list cannot be empty")
    if any(type(s) is not int for s in values):
        raise ValueError("Seeds must be integers, not bool/float/string values")
    if len(set(values)) != len(values):
        raise ValueError("Duplicate seeds are not independent configuration coverage")
    lo, hi = POOLS[split]
    for seed in values:
        if OFFICIAL[0] <= seed <= OFFICIAL[1]:
            raise ValueError(f"Official evaluation seed {seed} is reserved")
        if not lo <= seed <= hi:
            raise ValueError(f"Seed {seed} is outside {split} pool [{lo}, {hi}]")
    return values


def validate_lineage(records: Iterable[dict[str, str | None]]) -> None:
    """Require unique IDs, known parents, acyclic ancestry and inherited splits.

Records use episode_id, parent_id (None for roots), and split. Call this on
complete ancestry, not an isolated batch of descendants.
"""
    entries = list(records)
    by_id: dict[str, dict[str, str | None]] = {}
    for row in entries:
        eid = row.get("episode_id")
        if not isinstance(eid, str) or not eid:
            raise ValueError("Missing/non-string episode_id")
        if eid in by_id:
            raise ValueError(f"Duplicate episode ID: {eid}")
        if row.get("split") not in POOLS:
            raise ValueError(f"Unknown split on episode {eid}")
        by_id[eid] = row
    done: set[str] = set()
    for eid in by_id:
        visiting: set[str] = set()
        current = eid
        while current not in done:
            if current in visiting:
                raise ValueError(f"Cyclic lineage at {current}")
            visiting.add(current)
            row = by_id[current]
            parent = row.get("parent_id")
            if parent is None:
                break
            if not isinstance(parent, str) or parent not in by_id:
                raise ValueError(f"Unknown parent {parent!r} for {current}")
            if row["split"] != by_id[parent]["split"]:
                raise ValueError(f"Split leakage: {current} and parent {parent}")
            current = parent
        done.update(visiting)
