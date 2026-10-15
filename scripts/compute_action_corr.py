"""Estimate action-chunk covariance and build the correlated-noise matrix."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def main() -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="pi05_roboeval_pr8")
    p.add_argument("--beta", type=float, default=0.5)
    p.add_argument("--max-chunks", type=int, default=50_000)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    from roboeval_entry.training.corr_noise import (
        CorrelatedNoise, estimate_action_correlation)
    from roboeval_entry.training.openpi_bridge import (
        PR8_VARIANTS, REGISTRY_PATH)

    assets_root = (Path("/workspace/roboeval/models/openpi_assets")
                   / args.config / "roboeval" / "pr8_qpos18_h16")
    norm_stats_path = assets_root / "norm_stats.json"
    if not norm_stats_path.exists():
        raise FileNotFoundError(f"norm stats missing: {norm_stats_path}")

    est = estimate_action_correlation(
        PR8_VARIANTS, "train", REGISTRY_PATH, norm_stats_path,
        max_chunks=args.max_chunks)
    cn = CorrelatedNoise(est["corr"], beta=args.beta,
                         norm_stats_sha256=est["norm_stats_sha256"])
    cn.save(args.out)

    eig = np.linalg.eigvalsh(cn.M)
    off = cn.corr.copy()
    np.fill_diagonal(off, 0)
    corr14 = est["corr"].reshape(16, 32, 16, 32)
    lag1 = float(corr14[0, 0, 1, 0])
    print(json.dumps({
        "out": str(args.out),
        "n_chunks": est["n"],
        "beta": args.beta,
        "M_eig_min": float(eig.min()),
        "M_eig_max": float(eig.max()),
        "corr_offdiag_absmean": float(np.abs(off).mean()),
        "lag1_corr_dim0": lag1,
        "norm_stats_sha256": est["norm_stats_sha256"],
    }, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
