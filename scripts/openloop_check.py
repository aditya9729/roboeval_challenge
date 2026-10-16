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
    p.add_argument("--k", type=int, default=4,
                   help="chunks sampled per sample for the displacement ratio")
    p.add_argument("--split", choices=["train", "dev"], default="train")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from roboeval_entry.training.openpi_bridge import (
        install, maybe_install_corr_noise)
    install()
    corr_noise = maybe_install_corr_noise(args.config)
    import openpi.policies.policy_config as policy_config
    import openpi.training.config as _config
    import openpi.training.data_loader as dl
    from roboeval_entry.training.serving import SeededPolicy
    from roboeval_entry.training.train_loop import displacement_mse_ratio

    config = _config.get_config(args.config)
    data_config = config.data.create(config.assets_dirs, config.model)
    if args.split == "train":
        dataset = dl.create_torch_dataset(
            data_config, config.model.action_horizon, config.model)
        mean_source = dataset
    else:
        # dev eval set; baseline still computed on the TRAIN split.
        from roboeval_entry.training.parquet_dataset import (
            RoboEvalParquetDataset)
        import dataclasses as _dc
        fac = _dc.replace(config.data, split="dev")
        dataset = RoboEvalParquetDataset(
            dataset_root=fac.dataset_root, variants=fac.variants,
            split="dev", registry_path=fac.registry_path,
            action_horizon=config.model.action_horizon,
            state_mode=fac.state_mode)
        mean_source = dl.create_torch_dataset(
            data_config, config.model.action_horizon, config.model)
    policy = policy_config.create_trained_policy(config, args.checkpoint_dir)
    seeded = SeededPolicy(policy, action_horizon=config.model.action_horizon,
                          action_dim=config.model.action_dim,
                          corr_noise=corr_noise)

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
    all_actions = np.concatenate(
        [ep["actions"] for ep in mean_source._episodes])
    mean_action = all_actions.mean(axis=0)

    preds_k, targets = [], []
    lat = []
    for i, s in enumerate(samples):
        request = {k: v for k, v in s.items()
                   if k.startswith("observation/") or k == "prompt"}
        chunks = []
        for kk in range(args.k):
            # k=0 keeps the historical per-sample seed i.
            request["noise_seed"] = i if kk == 0 else i * 1_000_003 + kk
            t0 = time.perf_counter()
            chunks.append(seeded.infer(dict(request))["actions"])
            lat.append(time.perf_counter() - t0)
        preds_k.append(np.stack(chunks))
        targets.append(s["actions"])
    preds_k = np.stack(preds_k)          # (N,K,H,14)
    targets = np.stack(targets)
    preds = preds_k[:, 0]                # single-sample (k=0) view

    mse_pred = ((preds - targets) ** 2).mean(axis=(0, 1))
    base = np.broadcast_to(mean_action, targets.shape)
    mse_base = ((base - targets) ** 2).mean(axis=(0, 1))
    grip_pred = (preds[..., list(GRIP_DIMS)] > 0.5).astype(np.float32)
    grip_true = (targets[..., list(GRIP_DIMS)] > 0.5).astype(np.float32)
    grip_agree = float((grip_pred == grip_true).mean())

    cont_ratio = float(mse_pred[CONT_DIMS].mean() / mse_base[CONT_DIMS].mean())
    disp_k1 = displacement_mse_ratio(preds, targets, mean_action)
    disp_kk = displacement_mse_ratio(preds_k.mean(axis=1), targets,
                                     mean_action)
    report = {
        "config": args.config,
        "checkpoint_dir": args.checkpoint_dir,
        "n_samples": args.n,
        "split": args.split,
        "determinism_same_seed_identical": det_same,
        "determinism_diff_seed_differs": det_diff,
        "per_dim_mse_pred": mse_pred.tolist(),
        "per_dim_mse_mean_baseline": mse_base.tolist(),
        "cont_mse_pred": float(mse_pred[CONT_DIMS].mean()),
        "cont_mse_baseline": float(mse_base[CONT_DIMS].mean()),
        "cont_mse_ratio": cont_ratio,
        "disp_mse_ratio_k1": disp_k1,
        f"disp_mse_ratio_k{args.k}": disp_kk,
        "k": args.k,
        "gripper_agreement": grip_agree,
        "infer_ms_mean": float(np.mean(lat) * 1000),
    }
    (out / "report.json").write_text(json.dumps(report, indent=1))
    md = [f"# E005 open-loop check — {args.config}",
          f"samples: {args.n}, ckpt: {args.checkpoint_dir}",
          f"determinism: same-seed identical={det_same}, diff-seed differs={det_diff}",
          f"cont-dim MSE pred={report['cont_mse_pred']:.5f} vs baseline={report['cont_mse_baseline']:.5f} (ratio {cont_ratio:.3f})",
          f"disp_mse_ratio_k1={disp_k1:.3f} disp_mse_ratio_k{args.k}={disp_kk:.3f}",
          f"gripper agreement={grip_agree:.3f}", "",
          "| dim | mse_pred | mse_baseline |", "|---|---|---|"]
    md += [f"| {d} | {mse_pred[d]:.5f} | {mse_base[d]:.5f} |" for d in range(14)]
    (out / "report.md").write_text("\n".join(md) + "\n")
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
