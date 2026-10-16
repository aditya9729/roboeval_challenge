"""Dataset registry and action statistics over the LeRobot parquet dump.

Requires pyarrow + safetensors at call time (sim/train envs), imported lazily
so the package stays importable on a bare CPU interpreter.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from roboeval_entry.data.io import sha256_file, write_json_atomic
from roboeval_entry.data.splits import assign_demo_split

DATASET_NAME = "roboeval_ee_delta_20hz_scaled"
VARIANT_SUFFIX = "PositionAndOrientation"
VARIANT_SUFFIXES = ("PositionAndOrientation", "Position", "Orientation")
PERCENTILES = (0.1, 1.0, 99.0, 99.9)


def family_of(variant: str) -> str:
    for s in VARIANT_SUFFIXES:
        if variant.endswith(s):
            return variant[: -len(s)]
    return variant


@dataclass
class EpisodeRecord:
    episode_id: str
    parent_id: Optional[str]
    family: str
    variant: str
    split: str
    source: str
    source_revision: str
    length_frames: int
    dataset_episode_index: int
    reset_seed: Optional[int]


def available_variants(dataset_root: str | Path,
                       all_variants: bool = False) -> list[Path]:
    root = Path(dataset_root)
    if not root.exists():
        return []
    return sorted(p for p in root.iterdir()
                  if p.is_dir() and (all_variants
                                     or p.name.endswith(VARIANT_SUFFIX))
                  and list((p / "data").rglob("*.parquet")))


def dataset_revision(dataset_root: str | Path) -> str:
    """Dataset snapshot commit from HF download metadata (first line of each
    `.cache/huggingface/download/**/*.metadata` file). Raises if the files
    disagree; 'unknown' only when no metadata files exist."""
    meta_files = sorted(
        Path(dataset_root).glob(".cache/huggingface/download/**/*.metadata"))
    if not meta_files:
        return "unknown"
    commits = {p.read_text().splitlines()[0].strip() for p in meta_files}
    if len(commits) != 1 or not all(
            len(c) == 40 and all(ch in "0123456789abcdef" for ch in c)
            for c in commits):
        raise ValueError(f"inconsistent dataset commits in metadata: {sorted(commits)}")
    return commits.pop()


def _episode_columns(variant_dir: Path):
    import pyarrow.parquet as pq

    cols = ["episode_index", "frame_index", "uuid"]
    tables = [pq.read_table(p, columns=cols)
              for p in sorted((variant_dir / "data").rglob("*.parquet"))]
    return {c: np.concatenate([np.asarray(t.column(c).to_pylist(), dtype=object)
                               for t in tables]) for c in cols}


def _demo_seeds(demo_cache_root: Path, family_variant: str) -> dict[str, int]:
    """uuid -> reset seed from raw safetensors demo metadata."""
    from safetensors import safe_open

    seeds: dict[str, int] = {}
    for p in demo_cache_root.rglob(f"{family_variant}/*/lightweight/*.safetensors"):
        with safe_open(str(p), "np") as f:
            meta = f.metadata()
        seeds[json.loads(meta["uuid"])] = int(meta["seed"])
    return seeds


def build_demo_registry(dataset_root: str | Path, demo_cache_root: str | Path,
                        out_path: str | Path,
                        codebase_version: str = "unknown",
                        all_variants: bool = False) -> dict:
    """Scan variant dirs and write the registry.

    With all_variants=False (default) only *PositionAndOrientation dirs are
    scanned (original behaviour). With True, every variant dir with parquet
    data is included; family is the variant minus its condition suffix.
    """
    dataset_root, demo_cache_root = Path(dataset_root), Path(demo_cache_root)
    revision = dataset_revision(dataset_root)
    records: list[EpisodeRecord] = []
    families_done: list[str] = []
    variant_counts: dict[str, int] = {}
    for vdir in available_variants(dataset_root, all_variants):
        variant = vdir.name
        family = family_of(variant)
        seeds = _demo_seeds(demo_cache_root, variant)
        rows = _episode_columns(vdir)
        ep_idx = rows["episode_index"].astype(int)
        for ep in sorted(set(ep_idx.tolist())):
            idx = np.flatnonzero(ep_idx == ep)
            uuid = str(rows["uuid"][idx[0]])
            records.append(EpisodeRecord(
                episode_id=uuid,
                parent_id=None,
                family=family,
                variant=variant,
                split=assign_demo_split(uuid),
                source=DATASET_NAME,
                source_revision=revision if revision != "unknown" else codebase_version,
                length_frames=int(len(idx)),
                dataset_episode_index=int(ep),
                reset_seed=seeds.get(uuid),
            ))
        families_done.append(variant)
        variant_counts[variant] = int(len(set(ep_idx.tolist())))
    payload = {
        "dataset": DATASET_NAME,
        "source_revision": revision,
        "variants": families_done,
        "variant_counts": variant_counts,
        "episodes": [asdict(r) for r in records],
    }
    out = write_json_atomic(out_path, payload)
    write_json_atomic(str(out) + ".sha256", {"sha256": sha256_file(out)})
    payload["out_path"] = str(out)
    payload["sha256"] = sha256_file(out)
    return payload


def _action_columns(variant_dir: Path) -> np.ndarray:
    import pyarrow.parquet as pq

    tables = [pq.read_table(p, columns=["actions", "episode_index"])
              for p in sorted((variant_dir / "data").rglob("*.parquet"))]
    actions = np.concatenate(
        [np.stack([np.asarray(a, dtype=np.float64) for a in t.column("actions").to_pylist()])
         for t in tables])
    eps = np.concatenate([np.asarray(t.column("episode_index").to_pylist(), dtype=np.int64)
                          for t in tables])
    order = np.lexsort((np.arange(len(eps)), eps))
    return actions[order]


def action_statistics(dataset_root: str | Path, out_path: str | Path) -> dict:
    """Per-dim action stats per family and pooled over all available PR variants."""
    per_family: dict[str, np.ndarray] = {}
    for vdir in available_variants(dataset_root):
        family = vdir.name[: -len(VARIANT_SUFFIX)]
        per_family[family] = _action_columns(vdir)
    pooled = np.concatenate(list(per_family.values())) if per_family else np.zeros((0, 14))

    def stats(a: np.ndarray) -> dict:
        pct = np.percentile(a, PERCENTILES, axis=0)
        return {
            "count": int(a.shape[0]),
            "min": a.min(axis=0).tolist(),
            "max": a.max(axis=0).tolist(),
            "mean": a.mean(axis=0).tolist(),
            "std": a.std(axis=0).tolist(),
            "p0.1": pct[0].tolist(),
            "p1": pct[1].tolist(),
            "p99": pct[2].tolist(),
            "p99.9": pct[3].tolist(),
        }

    result = {
        "dataset": DATASET_NAME,
        "percentiles": list(PERCENTILES),
        "families": {f: stats(a) for f, a in per_family.items()},
        "pooled": stats(pooled),
    }
    out = write_json_atomic(out_path, result)
    result["out_path"] = str(out)
    result["sha256"] = sha256_file(out)
    return result
