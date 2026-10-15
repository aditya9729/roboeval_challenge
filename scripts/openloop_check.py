"""E005 open-loop fit check (openpi env, in-process, no server)."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

CONT_DIMS = list(range(12))
GRIP_DIMS = (12, 13)


def main() -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint-dir", required=True)
    p.add_argument("--n", type=int, default=40)
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

    # Determinism: same obs+seed identical, different seed differs.
    req = {k: v for k, v in samples[0].items()
           if k.startswith("observation/") or k == "prompt"}
    a1 = seeded.infer({**req, "noise_seed": 7})["actions"]
    a2 = seeded.infer({**req, "noise_seed": 7})["actions"]
    a3 = seeded.infer({**req, "noise_seed": 8})["actions"]
    det_same = bool(np.array_equal(a1, a2))
    det_diff = not bool(np.array_equal(a1, a3))

    # Mean-baseline: per-dim mean action chunk over the dataset's real actions.
    all_actions = np.concatenate([ep["actions"] for ep in dataset._episodes])
    mean_action = all_actions.mean(axis=0)

    preds, targets = [], []
    lat = []
    for i, s in enumerate(samples):
        request = {k: v for k, v in s.items()
                   if k.startswith("observation/") or k == "prompt"}
        request["noise_seed"] = i
        t0 = time.perf_counter()
        pred = seeded.infer(request)["actions"]
        lat.append(time.perf_counter() - t0)
        preds.append(pred)
        targets.append(s["actions"])
    preds = np.stack(preds)     # (N,H,14)
    targets = np.stack(targets)

    mse_pred = ((preds - targets) ** 2).mean(axis=(0, 1))
    base = np.broadcast_to(mean_action, targets.shape)
    mse_base = ((base - targets) ** 2).mean(axis=(0, 1))
    grip_pred = (preds[..., list(GRIP_DIMS)] > 0.5).astype(np.float32)
    grip_true = (targets[..., list(GRIP_DIMS)] > 0.5).astype(np.float32)
    grip_agree = float((grip_pred == grip_true).mean())

    cont_ratio = float(mse_pred[CONT_DIMS].mean() / mse_base[CONT_DIMS].mean())
    report = {
        "config": args.config,
        "checkpoint_dir": args.checkpoint_dir,
        "n_samples": args.n,
        "determinism_same_seed_identical": det_same,
        "determinism_diff_seed_differs": det_diff,
        "per_dim_mse_pred": mse_pred.tolist(),
        "per_dim_mse_mean_baseline": mse_base.tolist(),
        "cont_mse_pred": float(mse_pred[CONT_DIMS].mean()),
        "cont_mse_baseline": float(mse_base[CONT_DIMS].mean()),
        "cont_mse_ratio": cont_ratio,
        "gripper_agreement": grip_agree,
        "infer_ms_mean": float(np.mean(lat) * 1000),
    }
    (out / "report.json").write_text(json.dumps(report, indent=1))
    md = [f"# E005 open-loop check — {args.config}",
          f"samples: {args.n}, ckpt: {args.checkpoint_dir}",
          f"determinism: same-seed identical={det_same}, diff-seed differs={det_diff}",
          f"cont-dim MSE pred={report['cont_mse_pred']:.5f} vs baseline={report['cont_mse_baseline']:.5f} (ratio {cont_ratio:.3f})",
          f"gripper agreement={grip_agree:.3f}", "",
          "| dim | mse_pred | mse_baseline |", "|---|---|---|"]
    md += [f"| {d} | {mse_pred[d]:.5f} | {mse_base[d]:.5f} |" for d in range(14)]
    (out / "report.md").write_text("\n".join(md) + "\n")
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
