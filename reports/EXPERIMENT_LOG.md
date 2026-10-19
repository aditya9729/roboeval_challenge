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

## E009 — official evaluator on dev seeds, T002/pr8_v1 step 2000 (10% of training), 8 families × 10 seeds
- Date: 2026-10-15. Purpose: first end-to-end run of `submission/policy/policy.py` (thin websocket client → served pi0.5) through the official `roboeval.challenge.evaluate` on non-official seeds. Early-checkpoint diagnostic; not a result claim.
- Command: `bash scripts/eval_dev_pi05.sh pi05_roboeval_pr8 /workspace/roboeval/models/checkpoints/pi05_roboeval_pr8/pr8_v1/2000 configs/dev_eval_seeds.json /workspace/roboeval/runs/E009_dev_pr8v1_2000_r3` (serves with openpi env; evaluator in sim env, 8 workers, EGL GPU 0, concurrent with T002 training).
- Harness defect fixed on the way (runs `_r1`/`_r2` errored 80/80 with `keepalive ping timeout`): the vendor sync websocket client pings every 20 s while the server's blocking `infer` under 8 workers + training exceeded it → `RelaxedWebsocketClient` (same msgpack protocol, `ping_interval=None`) in `policies/pi05_client.py`. Vendor untouched. Failed runs retained.
- Result (`_r3`, exit 0, 33 min wall, 0 error terminations, GPU peak 44.2 GB incl. training): overall score **0.015** (outcome 0.019, safety 0.808, efficiency 0.430, coordination 0.759); success 0/80; tp means RotateValve 0.15, PackBox 0.12, CubeHandover 0.10, LiftPot 0.05, LiftTray 0.05, StackTwoBlocks 0.03, StackSingleBookShelf 0.00, PickSingleBook 0.00. results.json sha256 `f561851a7bc3…dde654`.
- Verdict: evaluator ↔ policy ↔ server path verified with the official tool; the model at 2k steps shows partial progress only. Next checkpoints (4k, 8k, …) will be evaluated the same way.

## E011 — soft-inpainting mechanism check (gate03/1000, 10 training observations, inference only)
- Date: 2026-10-15. Purpose: verify the new `prefix_actions` path (serving.py `sample_actions_inpaint`, threshold 0.3 → prefix frozen on the flow path for the first 70% of the 10 Euler steps) before using it in rollouts. Inspired by Larchenko et al. 2512.06951 §7.1 (plain variant, no correlation-aware propagation).
- Command: `scripts/inpaint_check.py --config pi05_roboeval_overfit --checkpoint-dir .../gate03/1000 --n 10 --out /workspace/roboeval/runs/E011_inpaint_check` (openpi env). Chunk A with seed s; prefix = A[4:8]; chunk B with seed s+1, same observation, with/without prefix.
- Result: same seed + same prefix → identical output. Time-aligned |B[0:4] − prefix|: **inpaint 0.0066 vs plain 0.0150** (2.3× closer); free tail |B[4:12] − A[8:16]|: 0.0189 vs 0.0192 (unchanged). (First-pass report compared row-aligned B vs A, which is the wrong alignment; corrected.)
- Verdict: mechanism correct and deterministic. Effect on dev-set score is an open question → part of the inference sweep once the 8k/12k checkpoints land.

## E010 — dev-seed learning curve of T002/pr8_v1 (official evaluator, 8 families × 10 dev seeds per checkpoint; automated by `scripts/watch_eval_checkpoints.sh`)
- Summary table: `/workspace/roboeval/runs/E010_dev_pr8v1_summary.tsv`; per-checkpoint bundles `E010_dev_pr8v1_<step>/`.
- step 4000 (exit 0, 36 min): score **0.0335** (outcome 0.039, safety 0.847, efficiency 0.542, coordination 0.864), success 1/80 (StackTwoBlocks 1/10); tp: RotateValve 0.20, PackBox 0.16, StackTwoBlocks 0.13, LiftTray 0.12, CubeHandover 0.10, PickSingleBook 0.05, LiftPot 0.03, StackSingleBookShelf 0.03. (step 2000 = E009: 0.015, 0 successes.)
- Further rows appended automatically at 8k/12k/16k/20k.

## Prep for T003 — correlated flow-matching noise (Larchenko et al. §5), β = 0.5
- Date: 2026-10-15. `scripts/compute_action_corr.py --config pi05_roboeval_pr8 --beta 0.5` over 50 000 train chunks (normalized with the pr8 quantile stats, padded to 16×32=512): corr mean |off-diag| 0.014, lag-1 temporal corr of dim 0 = 0.794; M = 0.5·corr + 0.5·I eigenvalues [0.504, 12.2]. Asset: `/workspace/roboeval/models/openpi_assets/pi05_roboeval_pr8/roboeval/pr8_qpos18_h16/action_corr_beta0.5.npz`.
- Config `pi05_roboeval_pr8_corr` = pr8 config + class-level patch of `PI0Pytorch.sample_noise` (training) and the same transform on the seeded inference noise (serving). Shares pr8 norm stats (verified identical q01). Launch deferred until T002 completes.

## INCIDENT-01 — T002 killed at step 7999 by the `/workspace` 500 GB quota (2026-10-15 23:23 UTC)
- Symptom: PID 167893 vanished mid-write of checkpoint 8000 (`tmp_8000/optimizer.pt` 6.3 of 13.5 GB) with no traceback (stdout was on the same full volume). cgroup `memory.events` oom_kill=0; `dd` to `/workspace` returned `Disk quota exceeded`. `df` does not show the quota (404 TB network FS).
- Cause: 11 × 20 GB overfit-gate checkpoints (`pi05_roboeval_overfit/gate0{1,2,3}`, 215 GB) + 3 × 20 GB T002 checkpoints + data/cache. Each openpi checkpoint = 7.5 GB weights + 13.5 GB AdamW state.
- Fix (owner-approved 2026-10-16): deleted `tmp_8000_incomplete`, all gate checkpoints except `gate02/2000` and `gate03/1000` weights (optimizer states removed) → ~200 GB freed. T002 resumed from step 6000 (`openpi_run.py train pi05_roboeval_pr8 --exp_name pr8_v1 --resume`, PID 346522, 14:56 UTC). Note: the vendor loop re-seeds the shuffle on resume, so data order 6000→20000 differs from an uninterrupted run; steps 6000–7999 were trained twice (different batches). ~2 000 steps + ~14 h GPU lost (session suspended overnight).
- Also observed: container cgroup `cpu.max` = 13.6 cores (not the 128 reported by `nproc`). T002 runs 3.4 s/step alone and ~6.5 s/step while a dev evaluation (server + 8 sim workers) runs concurrently → each E010 checkpoint eval costs ≈ 40 min of 2× slower training. Retention rule going forward: keep optimizer state only for the latest checkpoint.
- T003 (correlated noise) auto-launch cancelled; owner decision 2026-10-15: HPO on a validated recipe first, then T003 as a twin of the tuned baseline.

## E010 (cont.) — step 8000
- step 8000 (exit 0, 34 min; trained 6000→8000 after the resume): score **0.0151** (outcome 0.018, safety 0.879, efficiency 0.504, coordination 0.776), success 0/80; tp: RotateValve 0.275, PackBox 0.06, LiftPot 0.05, StackTwoBlocks 0.03, CubeHandover 0.03, StackSingleBookShelf 0.025, LiftTray 0.00, PickSingleBook 0.00.
- Reading: no closed-loop learning signal yet at ~1.2 epochs (10 seeds/family → 1 success = 0.0125, so 4k vs 8k is within noise). Hypotheses, in order: (H1) undertrained — lerobot's pi0.5 reference recipe is bs64 × 30k ≈ 1.9 M samples vs our 320 k; (H2) too little data — only the 8 P+O variants (776 train eps) of the 30 allowed; (H3) a train→dev generalization gap (to be measured by E013 open-loop on dev demos). Not yet excluded: a defect visible only in closed loop.

## DATA-02 — all 30 dataset variants downloaded; registry extended (2026-10-16)
- Source fact: vendor `docs/challenge.md`: "Demonstrations from every variant may be used for training." Organizers' paper (arXiv 2507.00435v2, Table III, P+O variant, 50 eps) pi0.5 full-FT success per family: RotateValve 0.72, PackBox 0.54, LiftPot 0.52, CubeHandover 0.40, StackTwoBlocks 0.40, LiftTray 0.32, PickSingleBook 0.22, StackSingleBookShelf 0.12; LoRA variant clearly worse.
- Command: `hf download helen9975/roboeval_ee_delta_20hz_scaled --repo-type dataset --revision 918b4072… --include <22 variants> --local-dir /workspace/roboeval/data/roboeval_ee_delta_20hz_scaled --max-workers 4` (train env). Verified all 1 319 remote files present with matching sizes. Disk 232 → 299 GB.
- Registry: `scripts/extend_registry.py` (same uuid-hash split, salt `roboeval-entry-v1`; the 852 pre-existing records asserted byte-identical). Now **3 060 episodes: 2 778 train / 282 dev**, 411 158 frames (373 026 train). Per family (frames): PackBox 77.7k, StackTwoBlocks 59.5k, RotateValve 52.2k, PickSingleBook 48.8k, LiftTray 46.9k, StackSingleBookShelf 45.5k, CubeHandover 41.3k, LiftPot 39.2k. New sha256 `6de96add…0cbdcc`.
- Note: 394 of 3 060 episodes (107 of the original 852) are not in vendor `successful_demos_ee_delta_20hz.txt`; M0 applied no filter. A `successful_only` option is being added as a data-mix choice for HPO; default keeps everything.

## E013 — open-loop generalization diagnostic, T002/pr8_v1 4k & 8k, train vs dev demos (2026-10-16)
- Command: `scripts/openloop_check.py --config pi05_roboeval_pr8 --checkpoint-dir .../pr8_v1/{4000,8000} --split {train,dev} --n 48 --out /workspace/roboeval/runs/E013_openloop_<step>_<split>` (openpi env, GPU shared with T002). Mean-action baseline always from the train split.
- Result (single-sample continuous MSE ratio / gripper agreement): 4000 train 1.251 / 0.952; 4000 dev 1.245 / 0.930; 8000 train 1.130 / 0.965; 8000 dev 1.335 / 0.949. Grippers learned; a single sampled arm chunk is *worse* than the mean action on both splits.
- Follow-up with the K-sample displacement metric (`--k 4`, n 32, 8000 train → `E013_openloop_8000_train_k4/report.json`): single-sample cont 1.291, disp_mse_ratio_k1 1.237, **disp_mse_ratio_k4 0.700**. The mean of 4 samples tracks integrated displacement better than the baseline, so the model has signal but single draws are noise-dominated (expected for a generative model when the per-step conditional variance is high).
- Measurement on data (linear probe, 8 116 train frames, 4 P+O variants, 80/20 split): R² of arm deltas from qpos[18] alone 0.00–0.09 per dim; from qvel[18] 0.22–0.37 on translational/main rotation dims (dims 3,4,9,10 ≈ 0 either way). Our model state was qpos only → velocity unobservable from one frame.
- Decisions: (1) new state mode `qpos18_qvel18` (36 dims = full `proprioception`, a standard observation) for all new configs; (2) validation/HPO objective = `disp_mse_ratio_k4` (MSE of the K=4 sample mean on cumulative arm displacement over the 16-step chunk, vs the train mean-chunk baseline); single-sample metrics still logged. The old criterion rewarded mode collapse.

## INCIDENT-02 — T002 loader worker SIGKILLed at step 11 975 (2026-10-16 21:18 UTC)
- Cause: two stalled 16-worker norm-stats/throughput jobs (vendor `compute_norm_stats.py` spawns 16 DataLoader workers × 64 OpenBLAS threads) exhausted the container `pids.max`=8 704 and ~40 GB RSS; cgroup oom_kill=0, host-level kill. Checkpoints intact through 10 000. Resumed from 10 000 (`--resume`, PID 459469, `OPENBLAS_NUM_THREADS=4`, log `train_stdout_resume2.txt`); steps 10 000–11 975 retrained with a different shuffle. ~2 000 steps lost again. Rule: no multi-worker CPU jobs beside training; thread env vars pinned.
- Container limits now known: cpu.max 13.6 cores, pids.max 8 704, /workspace quota 500 GB.

## DATA-03 / config `pi05_roboeval_all` (2026-10-16)
- All 30 variants, train split (2 778 eps / 370 248 samples), state `qpos18_qvel18` (36), `max_token_len=256` (discretized state tokens fit), otherwise identical to `pi05_roboeval_pr8` (bs16, cosine 1k warmup → 2.5e-5 → 2.5e-6, no EMA). Assets `openpi_assets/pi05_roboeval_all/roboeval/all_qpos36_h16/`.
- Norm stats: `scripts/norm_stats_local.py` (serial, identical transform chain + RunningStats as the vendor script; placeholder images) over 370 248 samples, 915 s → `norm_stats.json` sha256 `a204f33c…c63d4` (state 36, actions 14).
- Loader: no image cache for the 22 new variants (disk). Uncached `__getitem__` 45.5 ms/sample single-process → ≈280 samples/s on 13 cores ≫ 4.7/s needed at 3.4 s/step; contended measurement during T002 was 2.3/s (pr8 cached 2.5/s) — contention-dominated. Decision: no cache build.
- Inference: `Pi05RemoteAdapter(state_mode)` slices `proprioception[:36]`; `stage_submission.py` derives `state_mode` from the checkpoint's norm_stats; `SeededPolicy` rejects mismatched state dims. Tests 109 OK / 11 skipped.

## HPO01 — Optuna TPE over the pi0.5 recipe (queued behind T002)
- Objective: `disp_mse_ratio_k4` on the 282 dev demos (val every 250 steps, median pruning), proxy runs of 1 500 steps (cosine completes within the proxy) from the pi0.5 base on `pi05_roboeval_all`. Search space: peak_lr log[1e-5,1e-4], warmup [200,2000], weight_decay log[1e-6,1e-1], ema {none,0.99,0.999}, batch {8,16,32}. TPE seed 0, multivariate; MedianPruner(4 startup, 2 warmup). Each trial keeps only `best/` weights (7.5 GB) + val_history.
- Command (queued via `scripts/launch_after.sh <T002 pid>`): `scripts/hpo_tpe.py --config pi05_roboeval_all --n-trials 16 --timeout-hours 30` → `/workspace/roboeval/runs/HPO01/`. Top-2 configs will be validated with the official evaluator on dev seeds before the full baseline run.
- 2026-10-19: the original launcher died with T002 in INCIDENT-03; re-queued (chain PID 521259) behind the resume-3 training process: E010 evals of 12000/16000/20000 first, then HPO01.

## INCIDENT-03 — /workspace quota hit again; T002 died at step 17 950 (2026-10-17 04:16 UTC), found 2026-10-19 14:30 UTC
- Symptom: GPU idle, no training/launcher/eval processes; `dd` → `Disk quota exceeded`; `tmp_18000/model.safetensors` truncated at 3.4 GB. cgroup oom_kill=0. E010 12000/16000 evals also failed (`NO_RESULTS`, server could not write logs). Session was suspended ~2 days, so the stall went unnoticed.
- Cause: the retention rule promised in INCIDENT-01 (optimizer state only for the latest checkpoint) had not been implemented; 8 full checkpoints × 20 GB = 160 GB, plus 47 GB of checkpoints left by the train-loop GPU smoke (`runs/SMOKE_val`) and data grown 97 → 165 GB after the 30-variant download. Total ≈ 451 GB + base weights/envs ≈ quota.
- Fix (under the owner-approved INCIDENT-01 retention rule): deleted `tmp_18000`, `runs/SMOKE_val`, and `optimizer.pt` in checkpoints 2000–14000 (weights kept); 16000 kept intact for resume. ≈145 GB freed; `models/checkpoints` 174 → 83 GB. Resumed from 16000 (`--resume`, PID 519039, log `train_stdout_resume3.txt`, 2.54 s/step, ETA ≈ 17:30 UTC); steps 16000–17950 retrained with a different shuffle. Loss lost: ≈1 950 steps + ~2.5 days wall clock.
- TODO before HPO01 produces checkpoints: enforce retention programmatically (keep-last-k optimizer states) in `train_loop.py` / a post-save hook, since the vendor loop cannot. HPO trials already keep only `best/` weights.

## INCIDENT-04 — E010 12000/16000/20000 evals failed at staging (2026-10-19)
- Symptom: `watch_eval_checkpoints.sh` rows 12000/16000/20000 → `NO_RESULTS`, each after ~100 s. T002 itself completed (checkpoint 20000 written, 19 999 final); HPO01 started 17:40 UTC as queued (trial 0 complete, value 1.085).
- Cause: `stage_submission._state_mode_from_ckpt` (added in DATA-03) globbed `assets/*/norm_stats.json`, but openpi writes `assets/<repo_id>/<asset_id>/norm_stats.json` (one level deeper). The unit test used the shallow layout, so it passed. Would also have broken the planned official-evaluator validation of HPO01 trials.
- Fix: recursive search under `assets/`; test uses the real nesting. Verified on `pr8_v1/20000` → `qpos18` and `HPO01/t000/.../best` → `qpos18_qvel18`. Stale NO_RESULTS rows removed (backup `E010_dev_pr8v1_summary.tsv.bak_20261019`); evals re-launched 20:58 UTC (watcher PID 635087, log `E010_dev_pr8v1_watch2.log`), running concurrently with HPO01 trial 1.
- Also observed: `unittest discover` on HEAD has 7 pre-existing errors (6 × `test_parquet_dataset` image-cache/index mismatch after the 30-variant registry extension; 1 × `test_serving_inpaint.test_euler_inpaint_schedule` numpy/torch `.device` mismatch). Not caused by this change; to be triaged separately.
