"""Extend demo_registry.json to all dataset variants (sim/train env).

Rebuilds with all_variants=True, asserts every pre-existing episode record
is byte-identical, cross-checks uuids against the vendor successful_demos
list, writes registry + .sha256 sidecar atomically.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

DATASET_ROOT = Path("/workspace/roboeval/data/roboeval_ee_delta_20hz_scaled")
DEMO_CACHE = Path("/workspace/roboeval/data/roboeval_cache/roboeval_demos/1.0.0/BimanualPanda")
REGISTRY = ROOT / "data_manifests" / "demo_registry.json"
SUCCESS_FILE = Path("/workspace/roboeval/vendor/RoboEval/successful_demos_ee_delta_20hz.txt")


def main() -> int:
    from roboeval_entry.data.registry import build_demo_registry

    old = json.loads(REGISTRY.read_text())
    old_eps = {e["episode_id"]: e for e in old["episodes"]}

    new = build_demo_registry(DATASET_ROOT, DEMO_CACHE, REGISTRY,
                              codebase_version="extend_all_variants",
                              all_variants=True)
    new_eps = {e["episode_id"]: e for e in new["episodes"]}

    diffs = [u for u, e in old_eps.items() if new_eps.get(u) != e]
    assert not diffs, f"{len(diffs)} existing episodes changed: {diffs[:3]}"
    assert len(new_eps) == len(new["episodes"]), "duplicate episode_ids"

    succ = {ln.strip() for ln in SUCCESS_FILE.read_text().splitlines()
            if ln.strip() and not ln.startswith("#")}
    missing_succ = [u for u in new_eps if u not in succ]
    have_seed = sum(1 for e in new["episodes"]
                    if e["reset_seed"] is not None)

    by_variant = Counter(e["variant"] for e in new["episodes"])
    by_family = Counter(e["family"] for e in new["episodes"])
    splits = Counter(e["split"] for e in new["episodes"])
    print(json.dumps({
        "out": str(REGISTRY), "sha256": new["sha256"],
        "revision": new["source_revision"],
        "n_variants": len(new["variants"]),
        "episodes_total": len(new["episodes"]),
        "splits": dict(splits),
        "per_variant": dict(sorted(by_variant.items())),
        "per_family": dict(sorted(by_family.items())),
        "reset_seed_found": have_seed,
        "uuids_not_in_successful_demos": missing_succ[:20],
        "n_not_in_successful_demos": len(missing_succ),
    }, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
