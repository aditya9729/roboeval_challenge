"""Measure the action contract (SIM env: /workspace/roboeval/envs/sim/bin/python).

Writes /workspace/roboeval/runs/m0_contract_probe/probe.json, fills the
measured fields of contracts/action_spec.json, and emits
contracts/probe_summary.md. Read-only wrt the vendor tree.
"""
from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from roboeval_entry.data.io import write_json_atomic  # noqa: E402

PROBE_SEED = 100000
OUT_DIR = Path("/workspace/roboeval/runs/m0_contract_probe")
SPEC_PATH = REPO / "contracts" / "action_spec.json"


def make_state_env(family: str = "LiftPot", cameras=()):
    from roboeval.challenge.submission import CameraSpec, Interface, make_env

    interface = Interface(
        absolute=False, ee=True,
        cameras=tuple(CameraSpec(name=c, resolution=(256, 256)) for c in cameras),
    )
    return make_env(family, "PositionAndOrientation", interface)


def fk_poses(env):
    """EE poses of the CURRENT actuator ctrl targets (matches delta reference)."""
    jp = env.robot.get_initial_joint_positions()
    return np.asarray(env.robot.forward_kinematics(jp), dtype=np.float64)


def run_steps(env, action, n, collect=None):
    obs = None
    for _ in range(n):
        obs, _, term, trunc, info = env.step(np.asarray(action, np.float32))
        if collect is not None:
            collect(obs)
        if term or trunc:
            break
    return obs, info


def gripper_probe(env) -> dict:
    obs, _ = env.reset(seed=PROBE_SEED)
    phases = {}
    for label, cmd, steps in (
        ("cmd_1.0", 1.0, 40), ("cmd_0.0", 0.0, 40),
        ("cmd_0.49", 0.49, 20), ("cmd_0.51", 0.51, 20),
    ):
        traj_q, traj_g = [], []
        action = np.zeros(14)
        action[12] = action[13] = cmd
        for _ in range(steps):
            obs, _, term, trunc, _ = env.step(action.astype(np.float32))
            s = obs["proprioception"]
            traj_q.append([float(s[7]), float(s[8]), float(s[16]), float(s[17])])
            traj_g.append([float(x) for x in obs["proprioception_grippers"]])
            if term or trunc:
                break
        phases[label] = {
            "finger_qpos_start": traj_q[0], "finger_qpos_end": traj_q[-1],
            "gripper_obs_start": traj_g[0], "gripper_obs_end": traj_g[-1],
            "steps": len(traj_q),
        }
    open_cmd, closed_cmd = None, None
    q10 = np.asarray(phases["cmd_1.0"]["finger_qpos_end"])[::2]
    q00 = np.asarray(phases["cmd_0.0"]["finger_qpos_end"])[::2]
    if np.all(q10 > q00):
        open_cmd, closed_cmd = 1, 0
    elif np.all(q00 > q10):
        open_cmd, closed_cmd = 0, 1
    rounding = {
        "0.49_acts_like": _which(phases["cmd_0.49"], q10, q00),
        "0.51_acts_like": _which(phases["cmd_0.51"], q10, q00),
    }
    return {
        "phases": phases,
        "open_command": open_cmd,
        "closed_command": closed_cmd,
        "rounding": rounding,
        "note": "larger finger joint qpos = more open",
    }


def _which(phase, q_open, q_closed) -> str:
    end = np.asarray(phase["finger_qpos_end"])[::2]
    if np.allclose(end, q_open, atol=5e-3):
        return "cmd_1.0"
    if np.allclose(end, q_closed, atol=5e-3):
        return "cmd_0.0"
    return "ambiguous"


def frame_probe(env) -> dict:
    out = {}
    for side, sl in (("left", slice(0, 3)), ("right", slice(6, 9))):
        for axis, name in enumerate("xyz"):
            obs, _ = env.reset(seed=PROBE_SEED)
            before = fk_poses(env)
            action = np.zeros(14)
            action[sl.start + axis] = 0.01
            run_steps(env, action, 20)
            after = fk_poses(env)
            ee = 0 if side == "left" else 1
            out[f"{side}_d{name}"] = {
                "delta_world": (after[ee * 6: ee * 6 + 3] - before[ee * 6: ee * 6 + 3]).tolist(),
            }
    obs, _ = env.reset(seed=PROBE_SEED)
    before = fk_poses(env)
    action = np.zeros(14)
    action[5] = 0.05
    run_steps(env, action, 20)
    after = fk_poses(env)
    out["left_rotvec_z"] = {
        "delta_rot": (after[3:6] - before[3:6]).tolist(),
        "note": "rotvec change of left EE in pelvis frame after +z rotvec delta",
    }
    return out


def speed_clip_probe(env) -> dict:
    env.reset(seed=PROBE_SEED)
    before = np.asarray(env.robot.get_initial_joint_positions(), dtype=np.float64).copy()
    action = np.zeros(14)
    action[0] = 0.1
    env.step(action.astype(np.float32))
    after = np.asarray(env.robot.get_initial_joint_positions(), dtype=np.float64)
    delta = np.abs(after - before)
    return {
        "max_ctrl_delta_per_step": float(delta.max()),
        "limit": 0.10875,
        "within_limit": bool(delta.max() <= 0.10875 + 1e-9),
        "per_joint_delta": delta.tolist(),
    }


def determinism_probe(env) -> dict:
    rng = np.random.default_rng(0)
    actions = (rng.uniform(-0.01, 0.01, size=(30, 14))).astype(np.float32)
    actions[:, 12:] = 0.5
    trajs = []
    for rep in range(2):
        obs, info0 = env.reset(seed=PROBE_SEED)
        traj = [obs["proprioception"].copy()]
        for a in actions:
            obs, *_ = env.step(a)
            traj.append(obs["proprioception"].copy())
        trajs.append(np.stack(traj))
    same_seed_identical = bool(np.array_equal(trajs[0], trajs[1]))
    obs_a, info_a = env.reset(seed=PROBE_SEED)
    obs_b, info_b = env.reset(seed=PROBE_SEED + 1)
    return {
        "same_seed_bit_identical": same_seed_identical,
        "different_seed_proprioception_identical": bool(
            np.array_equal(obs_a["proprioception"], obs_b["proprioception"])),
        "different_seed_target_distance": [
            str(info_a.get("target_distance"))[:200],
            str(info_b.get("target_distance"))[:200]],
    }


def throughput_probe() -> dict:
    out = {}
    for label, cameras in (("state_only", ()), ("three_cameras", ("head", "left_wrist", "right_wrist"))):
        env = make_state_env(cameras=cameras)
        env.reset(seed=PROBE_SEED)
        action = np.zeros(14, dtype=np.float32)
        env.step(action)  # warmup
        t0 = time.perf_counter()
        for _ in range(50):
            env.step(action)
        wall = time.perf_counter() - t0
        out[label] = {
            "steps": 50,
            "wall_time_s": wall,
            "ms_per_step": wall / 50 * 1000,
            "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        }
        env.close()
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", default="LiftPot")
    parser.add_argument("--skip-throughput", action="store_true")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    probe: dict = {"family": args.family, "seed": PROBE_SEED,
                   "mujoco_gl": __import__("os").environ.get("MUJOCO_GL")}

    env = make_state_env(args.family)
    obs, info = env.reset(seed=PROBE_SEED)
    probe["action_space"] = {
        "low": np.asarray(env.action_space.low).tolist(),
        "high": np.asarray(env.action_space.high).tolist(),
        "dtype": str(env.action_space.dtype),
    }
    probe["obs_keys"] = list(obs.keys())

    probe["gripper"] = gripper_probe(env)
    probe["frame"] = frame_probe(env)
    probe["speed_clip"] = speed_clip_probe(env)
    probe["determinism"] = determinism_probe(env)
    env.close()
    if not args.skip_throughput:
        probe["throughput"] = throughput_probe()

    write_json_atomic(OUT_DIR / "probe.json", probe)

    spec = json.loads(SPEC_PATH.read_text())
    spec["env_bounds"] = {"low": probe["action_space"]["low"],
                          "high": probe["action_space"]["high"]}
    spec["gripper"]["open_command"] = probe["gripper"]["open_command"]
    spec["gripper"]["closed_command"] = probe["gripper"]["closed_command"]
    spec["gripper"]["verified"] = probe["gripper"]["open_command"] is not None
    write_json_atomic(SPEC_PATH, spec)

    lines = [
        "# Contract probe summary", "",
        f"- family: {args.family}, reset seed {PROBE_SEED}, MUJOCO_GL={probe['mujoco_gl']}",
        f"- gripper: open={probe['gripper']['open_command']} "
        f"closed={probe['gripper']['closed_command']} "
        f"rounding={probe['gripper']['rounding']}",
        f"- speed clip: max per-step ctrl delta "
        f"{probe['speed_clip']['max_ctrl_delta_per_step']:.5f} rad "
        f"(limit {probe['speed_clip']['limit']}, "
        f"within={probe['speed_clip']['within_limit']})",
        f"- determinism: same-seed bit-identical = "
        f"{probe['determinism']['same_seed_bit_identical']}",
    ]
    if "throughput" in probe:
        for k, v in probe["throughput"].items():
            lines.append(f"- {k}: {v['ms_per_step']:.1f} ms/step, peak RSS {v['peak_rss_mb']:.0f} MB")
    (REPO / "contracts" / "probe_summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
