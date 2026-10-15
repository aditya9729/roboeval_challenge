"""E006 diagnostic rollout harness (sim env; NOT the official evaluator)."""
from __future__ import annotations

import argparse
import dataclasses
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

ENTRY_ROOT = Path(__file__).resolve().parents[1]


def _episode_spec(family: str, variation: str, env_name: str, seed: int,
                  policy_seed: int, control_frequency: int, max_steps: int,
                  action_space):
    from roboeval.challenge.submission import EpisodeSpec, task_instruction

    return EpisodeSpec(
        family=family, variation=variation, env_name=env_name,
        instruction=task_instruction(family), episode_index=0, seed=seed,
        policy_seed=policy_seed, control_frequency=control_frequency,
        max_steps=max_steps, action_space=action_space)


def run_episode(env, policy, spec, record_actions: bool = True):
    from roboeval.challenge import results as res

    actions, lat = [], []
    info, termination, error = {}, "budget", None
    import traceback
    start = time.time()
    try:
        obs, info = env.reset(seed=spec.seed)
        random.seed((spec.policy_seed * 1_000_003 + spec.seed) % (2 ** 32))
        policy.reset(spec)
        for _ in range(spec.max_steps):
            t0 = time.perf_counter()
            action = np.asarray(policy.act(obs), dtype=np.float32)
            lat.append(time.perf_counter() - t0)
            obs, _, terminated, truncated, info = env.step(action)
            actions.append(action)
            if env.success:
                termination = "success"
                break
            if terminated:
                termination = "fail"
                break
            if truncated:
                termination = "unhealthy"
                break
    except Exception:
        termination, error = "error", traceback.format_exc(limit=8)
    raw = res.to_jsonable(info)
    if termination == "error":
        raw["success"] = 0.0
    return {
        "seed": spec.seed, "termination": termination, "error": error,
        "steps": len(actions), "wall_time_s": time.time() - start,
        "infer_ms_mean": float(np.mean(lat) * 1000) if lat else None,
        "raw": raw,
        "metrics": res.episode_metrics(raw, len(actions),
                                       spec.control_frequency),
        "actions": np.stack(actions) if (actions and record_actions) else None,
    }


def main() -> int:
    sys.path.insert(0, str(ENTRY_ROOT / "src"))
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--family", required=True)
    p.add_argument("--seeds", type=int, nargs="+", required=True)
    p.add_argument("--policy", choices=["hold", "pi05_remote"], required=True)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--execute-steps", type=int, default=4)
    p.add_argument("--cameras", default="head,left_wrist,right_wrist")
    p.add_argument("--policy-seed", type=int, default=0)
    p.add_argument("--max-steps-override", type=int, default=None)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cameras = tuple(args.cameras.split(","))

    from roboeval.challenge.submission import (CameraSpec, Interface, make_env)
    from roboeval.challenge import spec as spec_mod
    from roboeval_entry.contracts.action_spec import load_action_spec
    from roboeval_entry.control.guard import ActionGuard
    from roboeval_entry.policies.base import ObservationSchema
    from roboeval_entry.policies.official_bridge import OfficialPolicyBridge

    interface = Interface(
        absolute=False, ee=True,
        cameras=tuple(CameraSpec(name=c, resolution=(256, 256))
                      for c in cameras))
    env = make_env(args.family, "PositionAndOrientation", interface)

    if args.policy == "hold":
        from roboeval_entry.policies.hold import HoldPolicy as _H
        adapter = _H()
    else:
        from roboeval_entry.policies.pi05_client import Pi05RemoteAdapter
        adapter = Pi05RemoteAdapter(host=args.host, port=args.port,
                                    execute_steps=args.execute_steps,
                                    cameras=cameras)
    guard = ActionGuard(load_action_spec(ENTRY_ROOT / "contracts" /
                                         "action_spec.json"))
    policy = OfficialPolicyBridge(adapter, ObservationSchema(cameras), guard)

    max_steps = args.max_steps_override or spec_mod.max_steps(
        args.family, interface.control_frequency)
    records = []
    for seed in args.seeds:
        spec = _episode_spec(args.family, "PositionAndOrientation",
                             getattr(env, "name", args.family), seed,
                             args.policy_seed, interface.control_frequency,
                             max_steps, env.action_space)
        rec = run_episode(env, policy, spec)
        np.savez_compressed(out / f"actions_seed{seed}.npz",
                            actions=rec.pop("actions"))
        records.append(rec)
        print(json.dumps({k: v for k, v in rec.items() if k != "raw"}),
              flush=True)

    (out / "episodes.json").write_text(json.dumps(records, indent=1))
    n_succ = sum(r["termination"] == "success" for r in records)
    md = ["# rollout summary", f"policy={args.policy} family={args.family} "
          f"execute_steps={args.execute_steps} seeds={args.seeds}",
          f"successes: {n_succ}/{len(records)}", "",
          "| seed | termination | steps | tp | wall_s | infer_ms |",
          "|---|---|---|---|---|---|"]
    for r in records:
        tp = r["metrics"].get("tp", r["raw"].get("subtask_progress", "?"))
        md.append(f"| {r['seed']} | {r['termination']} | {r['steps']} | {tp} "
                  f"| {r['wall_time_s']:.0f} | {r['infer_ms_mean'] or 0:.1f} |")
    (out / "summary.md").write_text("\n".join(md) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
