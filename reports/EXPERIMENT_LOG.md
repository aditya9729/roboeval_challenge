# Experiment and evaluation log

Append-only. One entry per run that touched the simulator, the GPU, or produced a score. Every entry
records: ID, date, purpose, exact command, inputs (commit/checkpoint/data hashes), seeds/split,
compute, outputs (paths + sha256), result, and verdict. Scores on non-official seeds are local
diagnostics, never official results. Negative results stay in the log.

Conventions: `E<NNN>` = experiment/evaluation run; `T<NNN>` = training run. Artifacts live under
`/workspace/roboeval/runs/<id>_<slug>/` unless noted. Repo commit refers to `/root/roboeval-entry`.

---

## E001 — official install check (template hold policy, LiftPot, 2 official seeds)
- Date: 2026-10-14. Purpose: organizers' install sanity check (Getting Started step 2). Not model selection.
- Command: `python -m roboeval.challenge.evaluate --submission examples/challenge_submission --tasks LiftPot --episodes 2 --output-dir /workspace/roboeval/runs/official_install_check --overwrite` (sim env, `MUJOCO_GL=egl`)
- Inputs: vendor `c8b7959`; template policy; seeds 1000, 1001 (official; sanity use only).
- Compute: 1 CPU process + EGL on A100; 22 s wall.
- Output: `runs/official_install_check/{results.json,logs/}`; results.json sha256 `eb2c661f…6b74e`.
- Result: success 0.000, tp 0.000, 0 errors, RDS 0.000. Verdict: plumbing OK (expected zero).

## E002 — non-official plumbing run (template hold policy, 8 families × dev seeds 200000–200001, 3 cams)
- Date: 2026-10-14. Purpose: verify the local wrapper + seed guard + all 8 envs with cameras.
- Command: `ROBOEVAL_VENDOR=… SIM_PYTHON=… EVAL_WORKERS=1 EVAL_GPUS=0 bash scripts/run_local_eval.sh ./submission configs/dev_smoke_seeds.json dev /workspace/roboeval/runs/m0_plumbing_dev_smoke`
- Inputs: vendor `c8b7959`; template policy; dev pool seeds.
- Compute: 1 worker, A100 EGL; 171 s wall (6 200 env steps).
- Output: `runs/m0_plumbing_dev_smoke/{results.json,logs/,stdout.txt}`; results.json sha256 `622f7edf…7c9a9`.
- Result: all 8 families 2/2 episodes, 0 errors, RDS 0.000 (hold policy; safety 1.0, efficiency 0.80–0.91, coordination 1.0, outcome 0). Verdict: plumbing OK.

## E003 — contract probe (LiftPot PR, train-pool seed 100000)
- Date: 2026-10-14. Purpose: measure gripper semantics, delta frame, speed clip, determinism, step cost.
- Command: `scripts/probe_contract.py` (sim env). Output: `runs/m0_contract_probe/probe.json` sha256 `447d1b5a…2158`, `contracts/probe_summary.md`.
- Result: gripper 0=open/1=closed; clip 0.10875 rad/step; same-seed bit-identical; 7.8 ms/step state-only, 17.5 ms/step 3 cams. Verdict: contract fixed in `contracts/action_spec.json`.

## E004 — demonstration replay parity (8 families × first 5 dataset episodes)
- Date: 2026-10-14. Purpose: converted (dataset ee-delta → challenge env) vs organizer reference, plus direct 500 Hz replay; dataset alignment check.
- Command: pinned math env, `scripts/replay_parity.py --episodes-per-family 5 --out /workspace/roboeval/runs/m0_replay_parity` (sim env).
- Inputs: dataset HF commit `918b4072…`; raw demos 1.0.0; vendor `c8b7959`.
- Output: `runs/m0_replay_parity/parity.{json,md}`; parity.json sha256 `1b4e4adb…658af`.
- Result: converted 38/40 success, 40/40 exact match with `human_reference_episodes.json`; direct 35/40; shifted pairing `frame[t-1] → actions[t]` confirmed (first-10-step qpos error 3e-4–6e-3 vs 6e-3–6e-2). Verdict: replay contract verified; alignment recorded in action_spec.

## T001 — overfit gate: pi0.5 base → 4 LiftPot PR training episodes (595 samples)
- Date: 2026-10-14. Purpose: catch mapping bugs in the data/model adapter; NOT a generalization claim.
- Command: `scripts/openpi_run.py inspect|norm-stats|train pi05_roboeval_overfit --exp-name gate01 --overwrite` (openpi env; config registered in `roboeval_entry.training.openpi_bridge`).
- Inputs: openpi `215abfb2`; base weights `pi05_base` converted to PyTorch bf16 (`/workspace/roboeval/models/pi05_base_pytorch/model.safetensors`, 7.23 GB, sha256 `f4e5e8c223a15c5c…`); dataset HF `918b4072…`, 4 train-split LiftPot episodes; obs=frame[t-1] → actions[t:t+16]; state = qpos[18]; images head/left_wrist/right_wrist 256→224.
- Config: Pi0Config(pi05=True, action_horizon=16), batch 16, 400 steps, AdamW lr 5e-5 (20 warmup), bf16, grad checkpointing, no EMA.
- Compute: 1× A100-80GB; peak 34.5 GB allocated / 35.2 GB reserved; ~3.4 s/step (data-loader bound: 42 ms/sample PNG-from-parquet); 27.5 min wall.
- Output: `runs/T001_overfit_liftpot4/{train_stdout.txt,loss_curve.json,gpu_mem.log}`; checkpoints `/workspace/roboeval/models/checkpoints/pi05_roboeval_overfit/gate01/{200,399,400}` (~20 GB each incl. optimizer).
- Result: loss first-20 mean 0.191 → last-20 mean 0.065 (min 0.038). Verdict: adapter trains; gate passed pending open-loop (E005) and closed-loop (E006) checks.

## E005 — open-loop memorization check of T001/gate01 (step 400), 40 training samples
- Date: 2026-10-15. Purpose: stronger overfit gate than loss: does the model reproduce its own training action chunks?
- Command: `scripts/openloop_check.py --config pi05_roboeval_overfit --checkpoint-dir /workspace/roboeval/models/checkpoints/pi05_roboeval_overfit/gate01/400 --n 40 --out /workspace/roboeval/runs/E005_openloop_overfit` (openpi env, in-process).
- Inputs: gate01/400 checkpoint; same 4 LiftPot train episodes; metric = MSE of first predicted action (raw 14-D) vs dataset action, compared with predict-the-training-mean baseline. Pass criterion (pre-registered): continuous dims 0–11 ratio < 0.5 and gripper agreement > 0.95.
- Output: `runs/E005_openloop_overfit/report.{json,md}`.
- Result: cont MSE pred 4.74e-4 vs mean-baseline 4.23e-4 → ratio **1.12 (FAIL)**; gripper agreement 0.981 (pass); seeded noise deterministic (same seed identical, different seed differs); 62 ms/inference.
- Verdict: NEGATIVE. Grippers learned, arm deltas ≈ mean. Hypotheses (unranked): undertraining at 400 steps; qpos-only state insufficient; normalization. Checked: quantile norm stats for actions are well-conditioned (q01/q99 ≈ ±0.02–0.11 for arm dims, 0/1 grippers) → normalization not the cause. Next: T001b (longer budget).

## T001b — overfit gate rerun, 2000 steps (gate02)
- Date: 2026-10-15. Purpose: test the undertraining hypothesis from E005 with the same data/adapter, 5× step budget.
- Command: `scripts/openpi_run.py train pi05_roboeval_overfit --exp_name gate02 --overwrite --num_train_steps 2000 --lr_schedule.decay_steps 2000 --save_interval 500 --log_interval 20` (openpi env, nohup, PID 118042).
- Inputs/config: identical to T001 except steps 400→2000 (lr constant 5e-5 after 20 warmup).
- Compute budget: 1× A100-80GB, ≤2 h wall (observed 3.3 s/step, data-loader bound).
- Output: `runs/T001b_overfit_liftpot4_2k/train_stdout.txt`; checkpoints `.../pi05_roboeval_overfit/gate02/{500,1000,1500,1999,2000}`.
- Result: 2000/2000 steps, exit 0, 1 h 51 min wall (3.34 s/step); loss 0.13 (step 28) → 0.048 (658) → 0.022 (2000). Checkpoints 500/1000/1500/1999/2000 present.
- Verdict: see E006.

## E006 — open-loop check of T001b/gate02 at steps 500 and 2000
- Date: 2026-10-15. Command: `scripts/openloop_check.py --config pi05_roboeval_overfit --checkpoint-dir .../gate02/{500,2000} --n 40 --out /workspace/roboeval/runs/E006_openloop_gate02_{500,2000}` (same criterion as E005).
- Result: step 500 → cont ratio 1.066 (FAIL), gripper 0.987; step 2000 → cont MSE 1.54e-4 vs baseline 4.23e-4, **ratio 0.365 (PASS)**, gripper 0.998, deterministic seeding OK.
- Data check (same run): arm deltas are smooth (lag-1 autocorr 0.62–0.78; 3-step moving average explains 89% of variance; ~600 distinct values/dim, no quantization) → the <0.5 criterion is attainable and is kept unchanged.
- Verdict: E005 failure attributed to undertraining (400 steps ≈ 11 epochs of 595 samples). Gate passed at 2000 steps.
- **Adapter defect found during review (source fact, openpi `215abfb`, `pi0_pytorch.py` L244/L276):** with `pi05=True` the action expert never receives the continuous state; pi0.5 ingests proprioception only as 256-bin discrete tokens in the prompt (`TokenizePrompt(discrete_state_input=True)`). Our configs set `discrete_state_input=False`, so gate01/gate02 trained **without any proprioception** (images + task text only). Fixed in `openpi_bridge.py` (both configs → `True`, pi0.5 default); verified the prompt now reads `..., State: <18 ints>;\nAction:`. gate01/gate02 checkpoints are retained as negative-control artifacts only.

## T001c — overfit gate with proprioception fixed (gate03)
- Date: 2026-10-15. Purpose: re-run the gate with `discrete_state_input=True`; expect ≥ gate02 fit at ≤ half the budget.
- Command: `scripts/openpi_run.py train pi05_roboeval_overfit --exp_name gate03 --overwrite --num_train_steps 1000 --lr_schedule.decay_steps 1000 --save_interval 500 --log_interval 20` (openpi env, nohup, PID 150076).
- Compute budget: 1× A100-80GB, ≤1 h wall. Output: `runs/T001c_overfit_liftpot4_state/train_stdout.txt`; checkpoints `.../gate03/{500,999,1000}`.
- Result: 1000/1000 steps, exit 0, 63 min wall (3.3–3.8 s/step, GPU ~95% util, 42 GB); loss 0.37 (step 1) → 0.056 (481) → 0.032 (1000).

## E008 — open-loop check of T001c/gate03 at steps 500 and 1000
- Date: 2026-10-15. Command: `scripts/openloop_check.py --config pi05_roboeval_overfit --checkpoint-dir .../gate03/{500,1000} --n 40 --out /workspace/roboeval/runs/E008_openloop_gate03_{500,1000}` (criterion unchanged from E005).
- Result: step 500 → ratio 0.640 (vs 1.066 for gate02/500 without state), gripper 0.988; step 1000 → **ratio 0.399 (PASS)**, gripper 0.998.
- Verdict: adapter gate PASSED with the corrected pi0.5 state path; fixing the state input roughly halves the steps needed to reach the gate. M1 full-data training is unblocked. Still not a generalization claim.

## E007 — closed-loop plumbing: gate02/2000 served over WebSocket into the official sim (LiftPot, dev seeds)
- Date: 2026-10-15. Purpose: first live run of `scripts/serve_pi05.py` + `Pi05RemoteAdapter` + `scripts/rollout_local.py` (diagnostic harness, NOT the official evaluator). Not a performance claim: 4-episode model, no proprioception (see E006 defect).
- Commands: `serve_pi05.py --config pi05_roboeval_overfit --checkpoint-dir .../gate02/2000 --port 8000` (openpi env); `MUJOCO_GL=egl PYTHONPATH=src rollout_local.py --family LiftPot --seeds 200000 200001 200002 --policy pi05_remote --execute-steps 4 --out /workspace/roboeval/runs/E007_rollout_gate02_liftpot_dev` (sim env); hold-policy control run in `runs/E007_rollout_hold_liftpot`.
- Result: 3/3 episodes ran to budget (200 steps) with 0 errors; success 0/3, tp 0.25 on seed 200001; infer 17 ms/step (173 ms first episode incl. warm-up); executed actions have per-dim |mean| 0.003–0.045 (≈ dataset std), grippers toggled 49–60% of steps; several IK non-convergence warnings from the env (reverts to previous joints). Hold control: tp 0, infer 0.05 ms.
- Verdict: serving/adapter/guard/sim path verified end to end. Performance evaluation deferred to the all-task model.

## T002 — all-task pi0.5 fine-tune, 8 PR variants, train split (pr8_v1)
- Date: 2026-10-15. Purpose: M1 imitation baseline. First all-eight-task run; no augmentation, no RL.
- Design choices: full fine-tune (no LoRA; webpage shows Pi_0.5 0.405 vs LoRA 0.105 success); batch 16 (keeps ~34 GB so dev-set eval of intermediate checkpoints can share the A100); 20 000 steps ≈ 2.9 epochs over 109 761 training samples; openpi default fine-tune schedule (warmup 1 000, peak 2.5e-5 cosine → 2.5e-6, AdamW, clip 1.0, bf16, grad ckpt, no EMA — unsupported on the PyTorch path); `discrete_state_input=True`; obs = frame[t-1] → actions[t:t+16]; state = qpos[18]; 3 cams 224².
- Data: registry train split only — 776 episodes / 110 537 frames (CubeHandover 91, LiftPot 73, LiftTray 138, PackBox 86, PickSingleBook 86, RotateValve 106, StackSingleBookShelf 67, StackTwoBlocks 129). Dev/holdout/official pools excluded by the registry filter.
- Norm stats: `openpi_run.py norm-stats pi05_roboeval_pr8` (8 m 23 s) → `/workspace/roboeval/models/openpi_assets/pi05_roboeval_pr8/roboeval/pr8_qpos18_h16/norm_stats.json` sha256 `b546acdf…da07c`.
- Command: `scripts/openpi_run.py train pi05_roboeval_pr8 --exp_name pr8_v1 --overwrite` (openpi env, nohup, PID 167893). Compute budget: 1× A100-80GB, ≈18 h (3.28 s/step, 34.1 GB peak). Checkpoints every 2 000 steps → `/workspace/roboeval/models/checkpoints/pi05_roboeval_pr8/pr8_v1/`.
- Output: `runs/T002_pr8_bs16_20k/{train_stdout.txt,gpu_mem.log}`. Early loss: 0.318 (0) → 0.290 (100) → 0.150 (155), lr still in warmup.
- Result: PENDING (ETA ~09:00 UTC 2026-10-16). Plan: open-loop + dev-seed closed-loop check of the 4k/8k checkpoints on all 8 families while training continues.
