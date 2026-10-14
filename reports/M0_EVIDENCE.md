# M0 evidence — environment, contracts, and audit

Date: 2026-10-14. Host: 1x NVIDIA A100-SXM4-80GB (driver 580.126.16), 2x AMD EPYC 7742 (128 CPUs, AVX2), 2.1 TB RAM, Ubuntu 24.04 kernel 6.8 (`reports/hardware.json`).
Layout: code in `/root/roboeval-entry` (this repo); vendor checkout, envs, data, runs under `/workspace/roboeval/`.

Nothing in this report is a trained or evaluated robot policy. The only policy that ran is the organizers' hold-still template, used for plumbing.

## 1. Source and dependency locks (facts)

| Item | Value |
|---|---|
| Official checkout | `/workspace/roboeval/vendor/RoboEval` @ `c8b79598c9fb9021652128020517d1847abe4c95` (= `configs/challenge.json.expected_commit`), tag `corl2026-challenge`, harness 1.1.0 |
| Submodule | `thirdparty/mujoco_menagerie` @ `375dbb94ef35645f7a02813f6dac44fbd95f0064` |
| Tree state | `git status --porcelain --untracked-files=normal` empty after every step (editable install only creates gitignored `*.egg-info`) |
| Sim env | `/workspace/roboeval/envs/sim` (uv venv, Python 3.10.18). The literal official `pip install -e ".[examples]"` resolved: numpy 1.26.4, mujoco 3.3.3, dm_control 1.0.31, lerobot 0.3.3, torch 2.7.1, roboeval-metrics @ `4136b1be`. Lock: `/workspace/roboeval/envs/sim_requirements.lock.txt` sha256 `ea6e217e…ee0b3` |
| Train/data env | `/workspace/roboeval/envs/train` (Python 3.11): lerobot 0.4.4, huggingface_hub 0.35.3, numpy 2.2.6, torch 2.10.0. Lock sha256 `1ceadf4f…ee74e`. Not used for any training yet. |
| Dataset | `helen9975/roboeval_ee_delta_20hz_scaled`, 8 `*PositionAndOrientation` variants (29 GB) at `/workspace/roboeval/data/roboeval_ee_delta_20hz_scaled`, HF commit `918b4072cd1d15f28b1f96a440ac7aa84591ce92` |
| Raw demos | `RoboEval_data` 1.0.0 (4155 safetensors, 1.6 GB) at `/workspace/roboeval/data/roboeval_cache` (symlinked from `~/.roboeval`, which the vendor hardcodes) |
| EGL | Host lacked GL libs; installed `libegl1 libopengl0 libglvnd0 libglx0` (apt). NVIDIA EGL ICD present; `MUJOCO_GL=egl` works. No osmesa fallback. |

Full command log with exit codes: `/workspace/roboeval/runs/m0_env_setup_log.md`.

## 2. Official Getting Started checks (commands actually run)

1. `python -m roboeval.challenge.evaluate --submission examples/challenge_submission --tasks LiftPot --episodes 2 --output-dir /workspace/roboeval/runs/official_install_check --overwrite` → exit 0, ~22 s, `LiftPot 2 eps success 0.000 tp 0.000 errors 0`, RDS 0.000 (hold-still template). This is the organizers' install sanity check; it uses official seeds 1000/1001 only as such. results.json sha256 `eb2c661f…6b74e`.
2. Getting Started "explore" snippet + extended read-only probe → `runs/official_install_check/interface_probe.json`, `explore_snippet.txt`.
3. LeRobot load snippet (train env) on CubeHandoverPositionAndOrientation → 101 episodes, 15 092 frames, fps 20, keys `cam_*` (5×3×256×256), `state`(36), `state_gripper`(2), `actions`(14), `task`.
4. Non-official plumbing run (all 8 families × dev seeds 200000/200001, 3 cameras @256², template policy): `bash scripts/run_local_eval.sh ./submission configs/dev_smoke_seeds.json dev /workspace/roboeval/runs/m0_plumbing_dev_smoke` → exit 0, wall 171 s, 0 errors, RDS 0.000 (expected). results.json sha256 `622f7edf…7c9a9`. The wrapper's vendor/seed-file guards passed unchanged.

## 3. Verified interface contract (`contracts/action_spec.json`, sha256 `bd01278c…19d8`)

Source facts (pinned code) and measurements (`scripts/probe_contract.py`, `runs/m0_contract_probe/probe.json`):

- Action (14,) float32: `[left pos xyz (m), left rotvec (rad), right pos xyz, right rotvec, left gripper, right gripper]`. Position deltas are added in the pelvis (`invisible_base`) frame; rotation deltas pre-multiply the current rotation (`R(delta) * R(current)`). The reference pose is the FK of the current actuator *ctrl targets*, not measured qpos, so deltas integrate on the command. Each joint target then moves at most `2.175 rad/s * 0.05 s = 0.10875 rad` per step (measured: exactly 0.10875). Env action-space bounds are ±25 m / ±π / [0,1] — the ±25 is `action_scale`, not a physical limit.
- Gripper: discrete (rounded), reverse-controlled. Measured: command 0 = OPEN (finger qpos 0.04), command 1 = CLOSED; 0.49→open, 0.51→closed. `proprioception_grippers` reports openness (1 = open). Consequence: the template's `action[-2:] = obs["proprioception_grippers"]` commands *close* on an open gripper.
- State (36,) = 18 qpos [left joint1-7, left finger1-2, right joint1-7, right finger1-2] + 18 qvel in the same order; `proprioception_grippers` (2,) [LEFT, RIGHT]; two zero-length floating-base keys are present. RGB is CHW uint8.
- Dynamics: 25 physics substeps of 2 ms = 20 Hz. Same-seed replays are bit-identical. Reward is sparse `float(success)`; info carries `subtask_progress` and the metric fields.
- Throughput (LiftPot, single process): 7.8 ms/step state-only, 17.5 ms/step with 3×256² cameras; peak RSS 1.8/2.2 GB.
- Guard bounds (data-derived, dims 0–11): pooled p0.1/p99.9 over 121 k dataset actions, expanded by 25 % of span → |pos| ≲ 0.045–0.075 m, |rot| ≲ 0.12–0.22 rad per step; grippers [0,1]. Provenance: `data_manifests/action_statistics.json` sha256 `290a60fe…6afbe6b`.
- Dataset alignment (source: `examples/8_replay_to_lerobot.py` L483–519; measured in `runs/m0_replay_parity`): frame *t*'s `state`/images are the observation **after** `actions[t]`. Correct BC pairing is `obs = frame[t-1]` (reset observation for t = 0) → `actions[t]`. Over the first 10 steps the shifted pairing error is 3e-4–6e-3 rad vs 6e-3–6e-2 for same-index pairing across all 8 families. Dataset observations were recorded in the original absolute-joint 500 Hz replay while actions are the ee-delta conversion, so challenge-env rollouts drift up to ~0.1–0.8 rad qpos from the recorded `state` late in an episode even when the outcome is identical.

## 4. Demonstration replay evidence (`runs/m0_replay_parity/parity.{json,md}`, sha256 `1b4e4adb…658af`)

Pinned math env (`OPENBLAS_CORETYPE=Haswell`, 1 thread). First 5 dataset episodes of every PR family (40 episodes), seeds from raw-demo metadata:

| route | result |
|---|---|
| Converted (dataset ee-delta actions → official challenge env, same loop as `make_human_reference.py`) | 38/40 succeed; **40/40 match the organizers' `human_reference_episodes.json` entry exactly** (success, step count, all 14 metrics within validator tolerance), including the 2 LiftTray + 1 StackSingleBookShelf episodes that also fail in the organizers' replay |
| Direct (raw 500 Hz absolute-joint demo in its original env) | 35/40 succeed |

Converted replays reach success earlier than the dataset episode length (e.g. LiftPot ep0: 81 vs 116 frames) because the challenge loop stops at success; trailing dataset frames are post-success.

## 5. Data split registry (`data_manifests/demo_registry.json`, sha256 `66e88c64…0cabc`)

852 PR-variant episodes, each with uuid, family, variant, length, dataset episode index, reset seed, `parent_id=None`, `source_revision=918b4072…`. Split by uuid hash (salt `roboeval-entry-v1`): 776 train / 76 dev; no holdout demos (holdout is seed-based). Seed pools: train 100000–199999, dev 200000–200099, holdout 300000–300099, official 1000–1049 reserved (`configs/splits.json`, enforced by `roboeval_entry.data.seed_guard`). Descendants must inherit the parent split (`validate_lineage`).

## 6. Code delivered and tests

`src/roboeval_entry/`: `contracts/action_spec.py` (typed contract loader, fails on wrong commit/semantics), `control/guard.py` (finite/shape check, clip to guard bounds, gripper snap, clip accounting), `control/chunk.py` (chunk queue with reset), `policies/base.py` (`EpisodeContext` without seed/episode_index, `ObservationSchema` rejecting unexpected keys, `PolicyAdapter` protocol), `policies/{hold,router,official_bridge}.py`, `data/{seed_guard,splits,registry,io}.py`, `evaluation/{scoring,results_io}.py` (imports the official `scoring`/`spec` from the vendor tree; bootstrap + paired bootstrap). Scripts: `probe_contract.py`, `replay_parity.py`, `hardware_record.py` (sim env). `submission/` is the verbatim official template.

Tests: `PYTHONPATH=src python3.12 -m unittest discover -s tests -v` → 66 OK, 3 skipped (sim); `ROBOEVAL_SIM_TESTS=1` in the sim env → 66 OK, 0 skipped. Covered: action mapping/layout, guard (NaN, shape, clipping, gripper snapping, zero-residual identity), chunk episode boundaries, reset isolation (router instance reuse, two workers never share), seed/lineage leakage, official score aggregation (hold policy → 0, human medians → 1, fixture results.json reproduced to 1e-9), bootstrap sanity, live obs/action contract.

## 7. Design decisions taken in M0

- Owner decision: **openpi pi0.5 is the M1 main baseline, GR00T N1.7 secondary** (reverses DESIGN.md §1). Basis: official leaderboard (`configs/official_baselines.json`, fetched from the website JSON) mean success: Pi_0.5 0.405, ACT 0.278, Pi_0.5-LoRA 0.105, Diffusion 0.030, xVLA 0.015, GR00T 0.013. Those files do not publish an RDS and their `tl` values are inconsistent with `ct*20`, so they are reference points, not harness-identical numbers.
- Resolved for M1 (facts checked 2026-10-14): openpi main `215abfb2` (Apache-2.0), Python ≥3.11, jax 0.5.3/cuda12 + torch 2.7.1, numpy <2, base checkpoint `gs://openpi-assets/checkpoints/pi05_base`, full fine-tune needs >70 GB (fits one A100-80GB, tight), PyTorch training path available. **Blocker to design around:** openpi pins lerobot at `0cf86487` (CODEBASE_VERSION v2.1) while our dataset is LeRobot v3.0 → openpi's stock `LeRobotDataset` loader cannot read it; M1 needs our own parquet-backed dataset feeding openpi's model/training code. GR00T: `nvidia/GR00T-N1.7-3B` exists on HF (sha `2fc962b9`, ungated); Isaac-GR00T main `d2b7e75b`; license not yet reviewed.
- Demo split 90/10 train/dev by uuid; holdout seed-only.

## 8. Unresolved / blockers

- Submission `metadata.yaml` still holds template placeholders (team/contact/method) — need owner values; validator rejects `example.edu`.
- Organizer questions (`docs/ORGANIZER_QUESTIONS.md`) unsent; all approval-gated features stay disabled.
- Leaderboard baseline numbers are not RDS; our comparisons will use our own local dev/holdout RDS plus success/tp against those references.
- qvel in dataset `state` differs from the challenge env replay by up to ~4 rad/s in places; if a model consumes `state[18:]`, treat it as noisy or drop it.

## 9. Next step (M1) — runnable command

Set up the openpi environment and parquet dataset adapter, then run the small-set overfit gate before all-task training:

```bash
# planned (not yet implemented): in /root/roboeval-entry
bash scripts/setup_openpi_env.sh /workspace/roboeval/envs/openpi            # pin openpi commit, GIT_LFS_SKIP_SMUDGE=1 uv sync
/workspace/roboeval/envs/openpi/bin/python -m roboeval_entry.training.overfit_small --family LiftPot --episodes 4
```
