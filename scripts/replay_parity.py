"""Replay-parity check (SIM env: /workspace/roboeval/envs/sim/bin/python).

Mirrors make_human_reference.run_family exactly for the converted route, then
compares success/steps/metrics and per-step proprioception vs dataset `state`.
Also replays the same uuids' ORIGINAL 500 Hz absolute-joint demos in the
original env ("direct" route) as a sanity baseline.

Run with the pinned math env, e.g.:
  OPENBLAS_CORETYPE=Haswell OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  MKL_CBWR=COMPATIBLE MUJOCO_GL=egl python scripts/replay_parity.py \
      --episodes-per-family 5 --out /workspace/roboeval/runs/m0_replay_parity
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from roboeval_entry.data.io import write_json_atomic  # noqa: E402
from roboeval_entry.data.registry import available_variants  # noqa: E402

VARIATION = "PositionAndOrientation"
DATASET_ROOT = Path("/workspace/roboeval/data/roboeval_ee_delta_20hz_scaled")
REFERENCE_EPISODES = Path(os.environ.get(
    "ROBOEVAL_VENDOR", "/workspace/roboeval/vendor/RoboEval")) \
    / "roboeval/challenge/human_reference_episodes.json"
RTOL, ATOL = 1e-4, 1e-6


def load_episode_rows(variant_dir: Path, k: int):
    """First k episodes: (uuid, ep_index, actions [T,14], states [T,36])."""
    import pyarrow.parquet as pq

    cols = ["episode_index", "frame_index", "uuid", "actions", "state"]
    tables = [pq.read_table(p, columns=cols)
              for p in sorted((variant_dir / "data").rglob("*.parquet"))]
    rows = {c: np.concatenate([np.asarray(t.column(c).to_pylist(), dtype=object)
                               for t in tables]) for c in cols}
    episodes = []
    for ep in sorted(set(rows["episode_index"].tolist()))[:k]:
        idx = np.flatnonzero(rows["episode_index"] == ep)
        idx = idx[np.argsort(rows["frame_index"][idx].astype(int))]
        episodes.append((
            str(rows["uuid"][idx[0]]),
            int(ep),
            np.stack([np.asarray(a, dtype=np.float32) for a in rows["actions"][idx]]),
            np.stack([np.asarray(s, dtype=np.float32) for s in rows["state"][idx]]),
        ))
    return episodes


def demo_seed_map(family_variant: str) -> dict:
    from safetensors import safe_open

    from roboeval.demonstrations.demo_store import DemoStore

    seeds = {}
    for p in DemoStore()._cache_path.rglob(f"{family_variant}/*/lightweight/*.safetensors"):
        with safe_open(str(p), "np") as f:
            meta = f.metadata()
        seeds[json.loads(meta["uuid"])] = int(meta["seed"])
    return seeds


def replay_converted(env, seed, actions, states, max_steps):
    """Returns per-step state errors plus the episode metric record."""
    from roboeval.challenge import results as res
    from roboeval.challenge import spec

    obs, info = env.reset(seed=seed)
    start_err = float(np.abs(obs["proprioception"] - states[0]).max())
    errors_full, errors_qpos, errors_qvel, errors_finger = [], [], [], []
    align_same, align_shift = [], []
    success, steps = False, 0
    for t, a in enumerate(actions[:max_steps]):
        obs_before = obs
        obs, _, terminated, truncated, info = env.step(
            np.clip(a, env.action_space.low, env.action_space.high))
        steps += 1
        if t < len(states):
            diff = np.abs(obs["proprioception"] - states[t])
            errors_full.append(diff)
            errors_qpos.append(diff[:18])
            errors_qvel.append(diff[18:])
            errors_finger.append(diff[[7, 8, 16, 17]])
            # Alignment: dataset frame t's state is the observation AFTER
            # actions[t]; compare both candidate pairings on qpos dims.
            align_same.append(np.abs(obs_before["proprioception"][:18] - states[t][:18]).max())
            align_shift.append(np.abs(obs["proprioception"][:18] - states[t][:18]).max())
        if env.success:
            success = True
            break
        if terminated or truncated:
            break
    metrics = res.episode_metrics(res.to_jsonable(info), steps, spec.CONTROL_FREQUENCY)
    return {
        "success": success, "steps": steps, "dataset_steps": int(len(actions)),
        "start_state_error": start_err, "metrics": metrics,
        "state_err_max": float(np.max(errors_full)) if errors_full else None,
        "state_err_qpos_max": float(np.max(errors_qpos)) if errors_qpos else None,
        "state_err_qvel_max": float(np.max(errors_qvel)) if errors_qvel else None,
        "state_err_finger_max": float(np.max(errors_finger)) if errors_finger else None,
        "align_same_index_max": float(np.max(align_same)) if align_same else None,
        "align_shifted_max": float(np.max(align_shift)) if align_shift else None,
        "align_same_index_first10_mean": float(np.mean(align_same[:10])) if align_same else None,
        "align_shifted_first10_mean": float(np.mean(align_shift[:10])) if align_shift else None,
    }


def replay_direct(uuid: str, family_variant: str):
    """Replay the raw 500 Hz absolute-joint demo for this uuid in its own env."""
    from roboeval.demonstrations.demo import Demo
    from roboeval.demonstrations.demo_store import DemoStore

    matches = list(DemoStore()._cache_path.rglob(
        f"{family_variant}/*/lightweight/{uuid}.safetensors"))
    if not matches:
        return {"error": "demo file not found"}
    demo = Demo.from_safetensors(matches[0])
    # The cache directory layout is authoritative: the file lives under
    # <family_variant>/, while some legacy demos record a stale env_name
    # ("...Env" suffix, "PositionAndRotation"). Trust the path.
    demo.metadata.environment_data.env_name = family_variant
    env = demo.metadata.get_env(control_frequency=500)
    try:
        env.reset(seed=demo.seed)
        success, steps = False, 0
        for step in demo.timesteps:
            env.step(np.asarray(step.executed_action, dtype=np.float64))
            steps += 1
            if env.success:
                success = True
                break
        return {"success": success, "steps": steps, "seed": int(demo.seed),
                "demo_steps": len(demo.timesteps)}
    finally:
        env.close()


def compare_with_reference(ref: dict | None, ours: dict) -> dict:
    if ref is None:
        return {"reference_found": False}
    metrics_match = all(
        np.isclose(ours["metrics"][k], ref["metrics"][k], rtol=RTOL, atol=ATOL)
        for k in ref["metrics"])
    return {
        "reference_found": True,
        "success_equal": ours["success"] == ref["success"],
        "steps_equal": ours["steps"] == ref["steps"],
        "metrics_within_tol": bool(metrics_match),
        "ref_start_state_error": ref["start_state_error"],
        "ref_success": ref["success"],
        "ref_steps": ref["steps"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes-per-family", type=int, default=5)
    parser.add_argument("--families", nargs="*", default=None)
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--skip-direct", action="store_true")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    reference = json.loads(REFERENCE_EPISODES.read_text()) if REFERENCE_EPISODES.exists() else {}

    from roboeval.challenge import spec
    from roboeval.challenge.submission import Interface, make_env

    report = {"env": {"OPENBLAS_CORETYPE": os.environ.get("OPENBLAS_CORETYPE"),
                      "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
                      "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
                      "MKL_CBWR": os.environ.get("MKL_CBWR"),
                      "MUJOCO_GL": os.environ.get("MUJOCO_GL")},
              "families": {}}
    lines = ["# Replay parity", "",
             "| family | n | conv. success | ref match | max state err "
             "| align same max | align shifted max | align same f10 | align shifted f10 "
             "| direct success |",
             "|---|---|---|---|---|---|---|---|---|"]

    for vdir in available_variants(args.dataset_root):
        family = vdir.name[: -len(VARIATION)]
        if args.families and family not in args.families:
            continue
        episodes = load_episode_rows(vdir, args.episodes_per_family)
        if not episodes:
            continue
        seeds = demo_seed_map(vdir.name)
        refs = {e["uuid"]: e for e in reference.get(family, [])}
        env = make_env(family, VARIATION, Interface(absolute=False, ee=True))
        max_steps = spec.max_steps(family, spec.CONTROL_FREQUENCY)
        entries = []
        t0 = time.time()
        try:
            for uuid, ep_idx, actions, states in episodes:
                if uuid not in seeds:
                    entries.append({"uuid": uuid, "error": "no seed in demo metadata"})
                    continue
                ours = replay_converted(env, seeds[uuid], actions, states, max_steps)
                rec = {"uuid": uuid, "episode_index": ep_idx, "seed": seeds[uuid],
                       "converted": ours,
                       "comparison": compare_with_reference(refs.get(uuid), ours)}
                if not args.skip_direct:
                    rec["direct"] = replay_direct(uuid, vdir.name)
                entries.append(rec)
                print(f"{family} ep{ep_idx} uuid={uuid[:8]} conv_success={ours['success']} "
                      f"err={ours['state_err_max']}", flush=True)
        finally:
            env.close()
        n_ok = sum(1 for e in entries if e.get("converted", {}).get("success"))
        n_match = sum(1 for e in entries
                      if e.get("comparison", {}).get("metrics_within_tol")
                      and e["comparison"].get("success_equal")
                      and e["comparison"].get("steps_equal"))
        errs = [e["converted"]["state_err_max"] for e in entries
                if e.get("converted", {}).get("state_err_max") is not None]

        def _m(key):
            vals = [e["converted"][key] for e in entries
                    if e.get("converted", {}).get(key) is not None]
            return float(np.mean(vals)) if vals else float("nan")

        align = {k: _m(k) for k in
                 ("align_same_index_max", "align_shifted_max",
                  "align_same_index_first10_mean", "align_shifted_first10_mean")}
        d_ok = sum(1 for e in entries if e.get("direct", {}).get("success"))
        n_direct = sum(1 for e in entries if "direct" in e)
        report["families"][family] = {
            "episodes": entries,
            "summary": {
                "n": len(entries),
                "converted_success": n_ok,
                "reference_matched": n_match,
                "max_state_err": max(errs) if errs else None,
                "alignment": align,
                "direct_success": d_ok,
                "direct_n": n_direct,
                "wall_time_s": time.time() - t0,
            },
        }
        lines.append(
            f"| {family} | {len(entries)} | {n_ok} | {n_match} "
            f"| {max(errs) if errs else float('nan'):.3e} "
            f"| {align['align_same_index_max']:.3e} "
            f"| {align['align_shifted_max']:.3e} "
            f"| {align['align_same_index_first10_mean']:.3e} "
            f"| {align['align_shifted_first10_mean']:.3e} "
            f"| {d_ok}/{n_direct} |")

    write_json_atomic(args.out / "parity.json", report)
    (args.out / "parity.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
