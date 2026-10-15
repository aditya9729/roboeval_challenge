"""E011 inpainting validation (openpi env, in-process, no server)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def main() -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="pi05_roboeval_overfit")
    p.add_argument("--checkpoint-dir", required=True)
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--prefix-rows", type=int, default=4)
    p.add_argument("--execute", type=int, default=4)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from roboeval_entry.training.openpi_bridge import install
    install()
    import openpi.policies.policy_config as policy_config
    import openpi.training.config as _config
    import openpi.training.data_loader as dl
    from roboeval_entry.training.serving import SeededPolicy

    config = _config.get_config(args.config)
    data_config = config.data.create(config.assets_dirs, config.model)
    dataset = dl.create_torch_dataset(data_config, config.model.action_horizon,
                                      config.model)
    policy = policy_config.create_trained_policy(config, args.checkpoint_dir)
    seeded = SeededPolicy(policy, action_horizon=config.model.action_horizon,
                          action_dim=config.model.action_dim)

    idxs = np.linspace(0, len(dataset) - 1, args.n, dtype=int)
    samples = [dataset[int(i)] for i in idxs]
    r0 = args.execute
    r1 = args.execute + args.prefix_rows

    # determinism check on sample 0
    s0 = samples[0]
    req0 = {k: v for k, v in s0.items()
            if k.startswith("observation/") or k == "prompt"}
    A0 = seeded.infer({**req0, "noise_seed": 0})["actions"]
    d1 = seeded.infer({**req0, "noise_seed": 1,
                       "prefix_actions": A0[r0:r1]})["actions"]
    d2 = seeded.infer({**req0, "noise_seed": 1,
                       "prefix_actions": A0[r0:r1]})["actions"]
    det_inpaint = bool(np.array_equal(d1, d2))

    keys = ("B0k_vs_prefix", "Bk_end_vs_A_shifted", "rows0_e", "rows_e1_e2",
            "rows_e2_end")
    diffs = {"inpaint": {k: [] for k in keys}, "plain": {k: [] for k in keys}}
    for i, s in enumerate(samples):
        req = {k: v for k, v in s.items()
               if k.startswith("observation/") or k == "prompt"}
        A = seeded.infer({**req, "noise_seed": i * 10})["actions"]
        prefix = A[r0:r1].astype(np.float32)
        for name, extra in (("plain", {}),
                            ("inpaint", {"prefix_actions": prefix})):
            B = seeded.infer({**req, "noise_seed": i * 10 + 1,
                              **extra})["actions"]
            d = np.abs(B - A)
            k = r1 - r0
            # time-aligned: B starts where the prefix starts (A row r0)
            diffs[name]["B0k_vs_prefix"].append(np.abs(B[:k] - prefix).mean())
            n_rest = len(A) - r1
            diffs[name]["Bk_end_vs_A_shifted"].append(
                np.abs(B[k:k + n_rest] - A[r1:]).mean())
            diffs[name]["rows0_e"].append(d[:r0].mean())
            diffs[name]["rows_e1_e2"].append(d[r0:r1].mean())
            diffs[name]["rows_e2_end"].append(d[r1:].mean())

    summary = {name: {k: float(np.mean(v)) for k, v in dd.items()}
               for name, dd in diffs.items()}
    report = {
        "config": args.config,
        "checkpoint_dir": args.checkpoint_dir,
        "n_samples": args.n,
        "execute_rows": [0, r0],
        "prefix_rows": [r0, r1],
        "determinism_same_seed_same_prefix": det_inpaint,
        "mean_abs_diff_vs_chunk_A": summary,
    }
    (out / "report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
