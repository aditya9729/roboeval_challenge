"""Open-loop conditioning diagnostic (openpi env, in-process).

Extends the E005 check with controls that discriminate "model ignores the
observation" from "model is undertrained": a shuffled-observation control,
per-dim pred/target correlation, and the across-seed spread of samples.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

CONT_DIMS = list(range(12))


def _obs(sample: dict) -> dict:
    return {k: v for k, v in sample.items()
            if k.startswith("observation/") or k == "prompt"}


def main() -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint-dir", required=True)
    p.add_argument("--n", type=int, default=40)
    p.add_argument("--seeds", type=int, default=4)
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
    targets = np.stack([s["actions"] for s in samples])  # (N,H,14)
    all_actions = np.concatenate([ep["actions"] for ep in dataset._episodes])
    mean_action = all_actions.mean(axis=0)

    def run(obs_list: list[dict], seed_offset: int) -> np.ndarray:
        return np.stack([seeded.infer({**o, "noise_seed": seed_offset + i})["actions"]
                         for i, o in enumerate(obs_list)])

    obs = [_obs(s) for s in samples]
    preds_seeds = np.stack([run(obs, 1000 * k) for k in range(args.seeds)])  # (S,N,H,14)
    pred_mean = preds_seeds.mean(0)
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(obs))
    while np.any(perm == np.arange(len(obs))):
        perm = rng.permutation(len(obs))
    preds_shuf = run([obs[j] for j in perm], 0)

    def mse(a, b):
        return ((a - b) ** 2).mean(axis=(0, 1))

    base = mse(np.broadcast_to(mean_action, targets.shape), targets)
    m_single = mse(preds_seeds[0], targets)
    m_avg = mse(pred_mean, targets)
    m_shuf = mse(preds_shuf, targets)
    seed_spread = preds_seeds.std(0).mean(axis=(0, 1))
    # per-dim correlation of first-step prediction vs target across samples
    corr = []
    for d in range(14):
        x, y = pred_mean[:, :, d].ravel(), targets[:, :, d].ravel()
        corr.append(float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 and y.std() > 0 else 0.0)

    def ratio(m):
        return float(m[CONT_DIMS].mean() / base[CONT_DIMS].mean())

    report = {
        "config": args.config,
        "checkpoint_dir": args.checkpoint_dir,
        "n_samples": args.n,
        "seeds": args.seeds,
        "cont_ratio_single_seed": ratio(m_single),
        "cont_ratio_seed_avg": ratio(m_avg),
        "cont_ratio_shuffled_obs": ratio(m_shuf),
        "per_dim_corr_seed_avg": corr,
        "per_dim_seed_spread_std": seed_spread.tolist(),
        "per_dim_target_std": targets.std(axis=(0, 1)).tolist(),
        "per_dim_mse_single": m_single.tolist(),
        "per_dim_mse_shuffled": m_shuf.tolist(),
        "per_dim_mse_baseline": base.tolist(),
    }
    np.savez_compressed(out / "preds.npz", preds_seeds=preds_seeds,
                        preds_shuf=preds_shuf, targets=targets, idxs=idxs, perm=perm)
    (out / "report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items()
                      if not k.startswith("per_dim_mse")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
