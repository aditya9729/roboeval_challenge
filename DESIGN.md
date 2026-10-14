# RoboEval: Robust Cousins + Targeted Practice
## Implementation design and coding-agent handoff

**Owner:** Nirvana Robotics  
**Version:** 1.0 | **Verified:** October 14, 2026  
**Target:** CoRL 2026 RoboEval Diagnostic Bimanual Challenge, Track 2  
**Status:** Engineering specification, not an implemented or validated robot policy. No training or RoboEval evaluation was run in preparing this handoff.

## 1. Decision and objective

Build a reproducible entry for all eight task families before expanding the research stack. The main path is **GR00T N1.7 imitation learning → replay-verified cousins → small off-policy residual controller → success-guided training resets**. Preserve the best independently validated model for each task. IMPRINT, ACCRUE, and Cosmos are separately gated experiments, not dependencies of the first submission.

Optimize the official RoboEval Diagnostic Score (RDS), not just success, generated-video volume, or imitation loss. Do not promise a winning score: the proposed architecture must earn its place through controlled local evaluations.

GR00T's official 3B checkpoint and fine-tuning repository provide a concrete starting point [S09, S10]. Use a compatible, pinned release and checkpoint revision. A pi0.7-inspired recipe is not a substitute for an available, tested implementation: no pi0.7 checkpoint or reproduction is assumed in this design. An openpi pi0.5 adapter is the fallback and the preferred matched baseline for the optional IMPRINT experiment [S11, S15].

### Working assumptions

The user will provide a Linux GPU training host, storage, team contact details, and an account for uploading artifacts. Discover the actual host resources; do not assume previous hardware allocations are still available. Start with one GPU and serial evaluation when necessary. Scale worker count from measured memory use and throughput, not from a copied command.

The competition embodiment is the supplied fixed-base bimanual Franka Panda, not R1 Lite. Do not spend this sprint on mobile-base control, humanoid retargeting, deformables, or sim-to-real deployment [S01, S04].

### What success looks like

A cold-start reproducible bundle covers all tasks, passes official validation and action replay, and improves local held-out RDS over an unaugmented base. Every claimed gain has a dataset manifest, checkpoint hash, training configuration, and evaluation record. The final policy uses only the declared observations and task context.

## 2. Submission contract

### Official requirements to preserve

| Item | Requirement |
|---|---|
| Deadline | November 1, 2026, 23:59 AoE; November 2, 2026, 06:59 EST in New York. |
| Evaluation | Eight PR variants, 50 episodes each; 400 episodes total at 20 Hz. |
| Upload | Archive link or portal upload, plus SHA-256; one team account. |
| Report | CoRL format; at most four pages plus references. |
| Bundle size | Keep below 5 GB; external weights use `checkpoint_url` and `--exclude-checkpoint`. |
| Submission rate | Two accepted entries per UTC day; the latest valid entry is ranked. |
| Reproduction | Linux x86-64 with AVX2 and the prescribed numerical environment. |

Sources: [S01–S04, S07]. The report must explain method, data, interface, diagnostics, and compute. This design document is NOT the required competition report. A report outline is included; experimental results must be filled from real runs [S03].

### Authoritative code snapshot

The challenge reference `corl2026-challenge` resolved to:

```text
c8b79598c9fb9021652128020517d1847abe4c95
```

That resolution was checked through the repository API [S08]. The fetched specification reports harness version 1.1.0; do not copy the older version shown in website JSON examples. Record the resolved commit, submodule SHAs, dependency locks, and source fingerprint. Recheck organizer announcements before final packaging. An upstream change requires explicit review and a fresh evaluation, not a silent update [S02, S04].

Two source details justify additional local safeguards: the retrieved specification leaves its official fingerprint constant unset, and the packager leaves its maximum-size constant unset. Keep our own clean-tree, expected-commit, and under-5-GB checks even when an upstream check does not enforce them [S04, S07]. Never exploit an omitted validator check.

### Submission layout

```text
submission/
  metadata.yaml
  report.pdf
  checkpoint/                 # or externally hosted immutable weights
  policy/
    policy.py                 # make_policy(), reset(), act()
    requirements.txt          # exact extra runtime dependencies
  results.json                # written ONLY by official evaluate
  logs/                       # written ONLY by official evaluate
  MANIFEST.sha256             # written ONLY by official package
```

Copy the official template rather than inventing metadata. Keep local analysis, training runs, datasets, credentials, and extra logs outside this directory. Declare pretrained models, added data, compute, policy seed, cameras, action mode, and test-time sampling honestly [S03, S05–S07].

## 3. Compliance boundaries

### Default permitted lane

Use the unmodified challenge environment, ordinary public resets with non-evaluation seeds, published demonstrations, standard camera/proprioceptive observations, and training-side image augmentation. Preserve reward and termination behavior. Training configuration sampling may select among ordinary resets; never feed reset identifiers to the learned policy.

The rules permit synthetic data and domain randomization, but prohibit privileged-state policy access, benchmark changes, evaluation-seed tuning, and manual metric editing [S02]. Do not interpret the general augmentation permission as blanket authorization for every training modification.

### Approval-gated lane

Before using geometry/dynamics modifications, privileged object poses or contacts for teachers, intermediate-state resets, changed training rewards, or simulator-state restoration, obtain written organizer clarification for the exact implementation. Record the answer and permitted scope in `docs/ORGANIZER_QUESTIONS.md`. These features default to disabled.

Continue the baseline, image augmentation, ordinary-reset cousin collection, and standard-signal RL while those questions remain open. No private-state policy input is allowed in any lane. A privileged teacher, if approved for training, must remain in a separate process/repository from submitted inference.

### Isolation requirements

Use an external, read-only challenge checkout and our own training repository. Do not patch benchmark code, monkey-patch runtime physics, replace assets during official evaluation, or alter rewards, success checks, metrics, time limits, or joint limits. Standard inference code must not receive an environment object, `info`, simulator handles, reset configuration, ground-truth object pose, or contact state.

Use environment seeds only for resetting and allowed stochastic reproducibility. Do not pass `episode.seed` or `episode_index` as model features, task routing keys, retrieval identifiers, or trajectory lookup keys. Task-family routing is allowed [S02, S03].

Public evaluation seeds are **1000–1049** [S04]. Proposed disjoint pools: training 100000–199999; development 200000–200099; final local holdout 300000–300099. Split source demonstrations by entire episode and group all descendants with their parent. All views and temporal windows of an episode remain in the same split.

Reserve official seeds for infrastructure sanity checks and the frozen reporting run, never checkpoint or hyperparameter selection. Our local holdout is not the organizers' private test set. A leak invalidates the local experiment and requires a new split.

## 4. Tasks, score, and selection

| Canonical family | PR environment suffix | Budget | Steps |
|---|---|---:|---:|
| LiftPot | PositionAndOrientation | 10 s | 200 |
| LiftTray | PositionAndOrientation | 10 s | 200 |
| StackTwoBlocks | PositionAndOrientation | 20 s | 400 |
| StackSingleBookShelf | PositionAndOrientation | 40 s | 800 |
| CubeHandover | PositionAndOrientation | 20 s | 400 |
| PackBox | PositionAndOrientation | 20 s | 400 |
| PickSingleBookFromTable | PositionAndOrientation | 15 s | 300 |
| RotateValve | PositionAndOrientation | 20 s | 400 |

These names and budgets come from `spec.py`; load that source at runtime and assert agreement with our manifest [S04]. LiftPot and CubeHandover are the pilot tasks, not the scope of the final entry. Establish a usable imitation baseline across all eight before spending the main budget on pilot RL.

From each task's mean metrics, compute:

```text
O = 0.7 * success + 0.3 * task_progression
RDS_task = O * (0.4 + 0.3 * safety + 0.2 * efficiency + 0.1 * coordination)
RDS_suite = mean(RDS_task over the eight tasks)
```

Behavioral normalization uses the packaged human references. Import `scoring.task_scores` / `scoring.suite_scores` rather than maintaining a competing implementation. Time contributes through completion time; trajectory length is reported but not counted twice. Compute bootstrap intervals by resampling episodes within each task, recalculating mean metrics, and rerunning the official scoring function [S06].

Use matched local seed sets for comparisons. Screen cheaply on development episodes, then run 100 local holdout episodes per task for finalists and three training seeds for the principal experiment where resources permit. These are proposed internal budgets, not challenge requirements. Report fewer runs honestly when necessary; never fabricate uncertainty.

A proposed promotion gate is positive paired-bootstrap evidence for suite RDS without a material task collapse. Flag any task's success decrease greater than five percentage points for review. The five-point threshold is an engineering choice, not a statistical guarantee. For task-specific dispatch, select on development data, freeze the routing map, then evaluate the complete routed bundle on untouched local holdout.

## 5. System architecture

```text
Official demonstrations ----> episode audit + split registry ---------+
                                                                    |
Ordinary-reset collection -> verified successful cousins -> BC store +-> base VLA
                         \-> all valid rollouts -----------> RL store |      |
                                                                    |      v
                                          optional memory -> action proposal
                                                                    |
                                     small residual actor <----------+
                                              |
                                      action contract guard
                                              |
                                      official environment
                                              |
                       outcome + executed actions + diagnostics
                                              |
                         training-only curriculum / correction queue
```

At official inference, the collection, reward, sampler, correction, and diagnostic paths are absent. The deployable policy consists only of the selected model(s), optional validated memory/residual components, preprocessing, and action conversion.

### Process topology

Keep the simulator/evaluator in the official Python 3.10 environment. Put modern policy training and serving in a separate environment with that model's pinned dependencies. The challenge documentation explicitly warns of LeRobot/NumPy incompatibility with the evaluator environment [S01, S16]. Do not resolve this by upgrading the official simulator's numerical stack.

Start with a deterministic local policy subprocess or compatible in-process inference. If using local RPC, specify request IDs, worker/episode session IDs, array dtypes/shapes, checkpoint hash, task context, seeded reset, response schema, timeout, and process shutdown. Bind locally; do not use pickle on an exposed network socket. Package a launch script that works on an organizer machine without an undisclosed running service.

Each worker needs isolated queues, memory, random state, and model routing. A shared GPU service may batch requests only with strict session isolation and a reproducibility test. The evaluator calls policy construction per worker, so copying a large-model loading command into many workers can exhaust VRAM [S05]. Start serially; increase concurrency only after measuring peak allocation.

The evaluation is synchronous, not a real-time control contest [S03]. Optimize simulation cost after correctness. Do not prioritize TensorRT, latency-aware temporal modeling, or asynchronous action execution before a reliable baseline.

## 6. Data and action contract

### Canonical interface

Default to head, left-wrist, and right-wrist RGB at 256 x 256, proprioception, and end-effector delta actions at 20 Hz. Keep depth disabled initially. Camera count, action mode, and resolution must match the submission metadata and training preprocessing [S03, S16].

The documented dataset has `state` length 36, `state_gripper` length 2, and `actions` length 14. Live RGB is CHW uint8 under `rgb_<camera>`; dataset camera keys are `cam_<camera>` [S01]. Treat these as expected dimensions to verify, NOT sufficient semantic documentation.

Before any training, produce `contracts/action_spec.json` from source inspection and replay. It must name every state/action coordinate, arm ordering, rotation representation, coordinate frame, gripper sign/range, delta scaling, clipping, and normalization. Do not assume that the 36-state vector is a simple arrangement of seven joint positions and velocities per arm. Do not multiply already-scaled demonstration actions by a second scaling factor.

Replay a small fixed sample of original demonstrations through the chosen adapter, comparing direct playback with converted playback. Investigate any disagreement in outcome or measured trajectory. Not every recorded demonstration necessarily succeeds under the challenge's time budget; log the observed rate, do not assert 100% replay success.

### Data records

`EpisodeRecord` must contain immutable episode ID, parent/source IDs, family/variant, split, source revision, controller contract hash, timestamps, observation references, actions actually sent, next observations, reward as returned, terminal reason, success, and data-quality status. Generated records additionally need generator version, configuration ID, seed, approval-scope ID if applicable, and verification results.

`TransitionRecord` must distinguish the base proposal, residual proposal, guarded command actually sent, environment-returned reward, next observation, and the number of executed steps. Preserve the actual next observation, including terminal observations. A simulator error is not a legitimate physics transition.

Store RGB as episode video/chunks with explicit frame-index and timestamp maps; numeric arrays as Parquet/NPZ or a compatible LeRobot representation. Choose formats after inspecting the pinned loaders. Keep raw artifacts immutable and derived datasets versioned. Store credentials outside manifests and code; use environment-based access to optional object storage. Train from a local cache rather than assuming remote per-frame random reads are fast.

### Learning masks

Successful verified demonstrations populate the positive behavior-cloning pool. Failed but physically valid rollouts populate RL replay and diagnostic/memory pools. They do not automatically become desirable BC targets. Corrective demonstrations are a separately labeled source. Invalid renderings, NaNs, corrupt frames, and unhealthy simulator episodes are quarantined with reasons.

Keep a meaningful original-demonstration fraction during augmented training; initially test 50% original and 50% verified generated trajectories. Sample families and episodes before timesteps so long tasks do not silently dominate. Fit normalization only on the training split. Evaluate data scaling at fixed optimizer budget as well as, separately, at fixed passes through the data.

## 7. Cousin generation and verification

### Stage C0: label-preserving visual cousins

Apply modest, temporally consistent appearance perturbations to training observations: brightness, contrast, sensor noise, and carefully bounded occlusion. Preserve geometry and action labels. Do not apply random flips, independent frame warps, or crops that remove the grasp target without proving the corresponding control transformation. Keep camera-specific augmentation metadata.

These are appearance augmentations, not new physical demonstrations. Measure their effect separately from new trajectory collection. Reserve an untouched nominal validation stream.

### Stage C1: spatial cousins through ordinary resets

Generate fresh non-evaluation initializations using the existing task distributions. Execute the current policy; retain successful physically valid episodes as new demonstrations and all valid failures as RL experience. Where the policy cannot solve a region, collect a corrective demonstration using the supported tools. First reproduce the original task/controller semantics before expanding coverage.

Start with 64 configurations per pilot task and grow in measured steps. This finite-pool size is a proposal for obtaining repeated outcomes under modest compute; it is not a paper setting. Every configuration must pass a full ordinary reset. Near-identical episodes should not dominate a dataset merely because generation is cheap.

### Stage C2: contact-aware geometric/dynamics cousins, approval required

Only after written authorization, use an isolated training factory to alter approved shape, material, friction, mass, or scene factors. Keep variation mild initially and tied to specific observed failures. A cousin must preserve task identity, affordances, reachable contacts, and physically valid placement.

For object-relative trajectory adaptation, a potential starting point is:

```text
T_new_world_eef = T_new_world_object * inverse(T_old_world_object) * T_old_world_eef
```

This proposal maps an object-relative pose; it is NOT proof of executable contact. Apply phase-specific frames, solve IK, check joint/self/environment constraints, and run the trajectory in physics. For dual-arm lifting, preserve the relative gripper-to-object transforms of both arms. For handover, the receiving arm must establish an actual stable hold before the giver releases. Never infer a grasp solely from a closed-gripper command.

Use trajectory repair or a new teacher rollout where simple warping fails. If the necessary teacher state is not approved, collect new demonstrations through observations instead. Do not preserve old action labels after geometry, camera extrinsics, or dynamics changes without re-execution and correct relabeling.

### Verification gates and scaling criterion

Accept an episode only after checking structural validity, timestamps, finite states/actions, successful execution under the intended task semantics, valid camera coverage, and reproducible lineage. When privileged physical checks are authorized, additionally check interpenetration, object support, contact continuity, reachable grasps, and rollout stability. Mark which checks were actually available rather than silently treating missing tests as passes.

Track candidate count, acceptance rate, successful episodes per GPU-hour, coverage per family/phase, replay stability, and holdout RDS. Scale accepted useful behaviors, not raw render count. Run original-only, appearance-only, and verified-trajectory augmentation as distinguishable ablations. Stop a generation branch that produces more data without improving holdout behavior.

## 8. Base policy and inference execution

### Baseline B0

Fine-tune the pinned GR00T N1.7 checkpoint with a new, explicit bimanual-Panda data/modality adapter. Use the official model's native training objective and supported fine-tuning path before changing its architecture [S09, S10]. First overfit a small training-only episode set to catch mapping bugs, then train all tasks. Small-set loss reduction is a debugging gate, not a generalization claim.

If native output horizon is configurable, test a 16-step prediction with four-step execution; otherwise keep the supported native horizon and execute the first four steps. At 20 Hz those are a 0.8-second proposal and 0.2-second open-loop execution. These are starting choices, not validated optima. Compare execution lengths 1, 2, and 4 on development data. Each `act()` still returns exactly one action.

Train and evaluate with the same action guard. Zero learned residual must reproduce the frozen base's guarded actions exactly. Do not average across discrete gripper open/close transitions. Arbitrary action smoothing can change contact behavior; promote it only after an ablation.

### Failure diagnosis before architectural changes

Classify failures as perception/localization, action-interface mismatch, missed grasp, slip, bimanual timing, premature release, repeated failed retry, unnecessary path/jerk, or exhausted budget. This diagnostic taxonomy is ours; actual official progression comes from the benchmark logs. Prioritize bugs and missing coverage before assuming a model-capacity problem.

Retain an openpi pi0.5 branch only when needed as a fallback or matched memory baseline. Do not train two large backbones, add memory, and change RL simultaneously in the first controlled experiment.

## 9. Off-policy RL and success-guided practice

### Residual baseline R0

Use a ResFiT-inspired small actor around a frozen base; this is an adaptation, not a claim to reproduce the entire ResFiT implementation. ResFiT motivates improving a black-box BC policy through off-policy residual control [S12]. Prefer a tested TD3/FastTD3 implementation after checking its action, replay, and target-update semantics.

For the first RL experiment, replan the base and execute one corrected action per control step. This avoids hidden pending-chunk state. A later chunked implementation must include the remaining proposal queue, its position, and required base history in the replay state.

```text
x_t = (fixed visual features, proprioception, task context,
       base proposal b_t, prior executed action, remaining step budget,
       required causal history)
u_t = actor(x_t)
a_sent_t = guard(b_t + bounded_residual(u_t))
```

The residual acts in verified controller-command coordinates. Initialize its final layer to zero. Derive per-dimension residual bounds from the verified contract, initially at 5% of finite command span. For unbounded coordinates, derive documented bounds from training data instead of taking an infinite span. Initially mask the gripper dimensions; enable gripper correction only after a dedicated stability test. Small rotational residuals must respect the verified action representation.

Train twin critics on the residual-control MDP, conditioned on the same base proposals as the actor. Retain both proposed and sent actions so clipping is not confused with execution. Freeze base/visual encoders within each RL round. Do not use cached features from a changed encoder without recomputing them.

### Rewards and terminal semantics

Use the unmodified environment-returned reward first. Inspect its implementation and verify whether it carries a useful success signal. Do not assume progress logs, contact state, or reward shaping are permitted training signals. If the native reward is unusable and no alternative is authorized, continue verified imitation/correction rather than silently inventing dense rewards.

For a finite-horizon challenge objective, the official episode budget is terminal. A collector interruption is a different event: do not convert it into task failure. Unhealthy states are quarantined; their terminal labels cannot be treated as normal next-state samples. For any later n-step version, discount by the number of actions actually executed, stop at termination, and do not bootstrap through success or the task budget.

Example scalar TD3 target for a k-step executed transition:

```text
G_t = sum_{j=0}^{k-1} gamma^j * r_{t+j}
y_t = G_t + gamma^k * bootstrap_mask * min(Q1_target, Q2_target)
```

For R0, k=1. Log critic error, return scale, residual magnitude, clipping rate, success and RDS. Prefer small monitored update-to-data ratios initially; do not assume an extreme ratio transfers from GPU-native benchmarks to this visual MuJoCo setting.

### SGS adaptation S0

SGS prioritizes configurations with intermediate recent success rather than spending uniformly on mastered and currently unreachable configurations [S13, S14]. Our first variant samples ordinary reset seeds from a fixed training pool, not privileged intermediate states. That change and the off-policy learner make it an SGS-inspired adaptation.

Within each task, maintain a rolling window of binary outcomes per configuration. The paper-inspired weighting is:

```text
log w_i = kappa*tau*log(p_i + epsilon)
        + kappa*(1-tau)*log(1 - p_i + epsilon)
P_sgs = softmax(log w / temperature)
```

Proposed initial settings: tau=0.5, kappa=2, temperature=1, window=20. Use uniform warm-up until every pool element has at least two valid attempts. Preserve exploration with a mixture: 50% SGS, 25% uniform, 25% successful-but-poor-quality cases. The latter uses approved training diagnostics only; otherwise redistribute that mass to uniform. Keep task allocation balanced.

The included reference sampler tests normalized finite probabilities, full support, reproducibility, and preference for intermediate success. It does not implement the robot learner or reproduce the paper's full reset system. Record policy-version windows so stale outcomes do not steer a new checkpoint indefinitely.

Maintain a separate correction queue for persistent failures. Inspired by Mulligan, spend human or teacher supervision where autonomous learning cannot bootstrap; do not confuse failure-focused demonstration collection with SGS autonomous reset allocation [S17].

## 10. Optional research branches

### IMPRINT: add only after a demonstrated memory failure

IMPRINT compresses visual, executed-action, and proprioceptive history into latent memory, pretrains it with reconstruction/progress objectives, then adapts it jointly with a policy [S15]. Run pi0.5 short-history versus pi0.5 + IMPRINT on identical data. A GR00T-versus-pi0.5 comparison cannot isolate memory. Porting IMPRINT to GR00T is a separate project, not a toggle.

Keep historical actions causal: previous executed commands only, never the future demonstration action or an unexecuted plan. Reset memory per episode; test interleaved worker sessions. Elapsed-time fraction is not successful completion, especially for failures and retries. Keep progress supervision explicitly labeled; do not assign failed episodes a completion target of one just because the recording ends.

If later authorized to reset to intermediate states, reconstruct valid causal memory using a matching replayed prefix or explicitly trained fresh-start reset. During RL, freeze memory or regenerate recurrent states from raw prefixes; changing memory encoders invalidates cached latent replay.

### ACCRUE: task transfer, not mandatory model consolidation

ACCRUE carries an actor–critic learner between tasks and retains prior behavior through actor distillation. Its project separates GR00T/FastSAC simulation from pi0.5/FastTD3 real-robot experiments; code is still marked coming soon in the checked project page [S18].

After single-task RL works, compare independent learners, warm-started task-specific checkpoints, and an ACCRUE-inspired continual branch. Try LiftTray → LiftPot → CubeHandover as a hypothesis about transfer. For each stage, measure new-task interaction efficiency and prior-task retention. Keep the strongest isolated checkpoints as a submission fallback. Do not silently train old-task critics on new-task rewards.

### Cosmos and Chronos-style ideas

Cosmos is not required for the executable-cousin pipeline. Treat any later generative video augmentation as a separate unverified data source until temporal, multiview, geometry, and label fidelity checks pass. Never use an imagined contact outcome as a verified RL reward. No specific Cosmos checkpoint is selected by this document.

History modeling can be tested through the IMPRINT branch. Timing-aware policy changes are not a first priority in synchronous evaluation. No Chronos reproduction is claimed, and no unnamed implementation should be imported without a source/release review.

## 11. Controlled experiment ladder

| ID | Variant | Main question |
|---|---|---|
| B0 | Original demonstrations + base VLA | Is the interface and baseline sound? |
| B1 | B0 + appearance augmentation | Does visual augmentation help nominal/generalized control? |
| B2 | B0 + verified trajectory cousins | Does executable coverage improve spatial robustness? |
| R0 | Best approved BC data + residual RL, uniform resets | Does RL add value? |
| S0 | Same as R0, same budget, SGS mixture | Does practice allocation add value? |
| M0 | pi0.5 short history vs matched pi0.5 + IMPRINT | Does memory fix diagnosed failures? |
| T0 | Independent vs warm-start vs ACCRUE-inspired learners | Is task transfer beneficial without regressions? |

The principal comparison is R0 versus S0. Match base checkpoint, data, environment steps, optimizer steps, cameras, action contract, and task allocation. Record wall-clock and GPU-hours separately. Compare data-scaling experiments under fixed optimization budget; do not attribute longer training to better augmentation.

All branches must use a common evaluation contract. Never compare RGB-only and privileged-state policies as though only the sampler changed. Include failures and errors in the reported coverage; do not delete bad episodes to improve the result.

## 12. Repository and agent work packages

```text
roboeval-entry/
  AGENTS.md                  # agent constraints
  DESIGN.md                  # this specification
  configs/                   # split pools, methods, tasks, approval flags
  src/roboeval_entry/
    contracts/               # verified observation/action schema
    data/                    # auditing, episode splits, stores, conversion
    cousins/                 # approved generation + acceptance gates
    policies/                # GR00T/openpi adapters, task router
    control/                 # guard, chunk queue, residual interface
    rl/                      # learner, replay, sampler, collectors
    evaluation/              # official-harness wrappers and diagnostics
    serving/                 # local worker-isolated inference
  tests/                     # unit, contract, integration, replay, leakage
  scripts/                   # setup, train, local eval, package
  reports/                   # local experiment summaries; not official results
  submission/                # staged final artifacts only
```

### M0 — environment, contracts, and audit

Inspect the pinned source and native model documentation; produce dependency locks, action/state contract, split registry, task manifest, and memory/throughput probe. Make the official template run on two non-official local seeds. Validate direct-versus-adapted demo replay. Exit evidence: command logs, resolved SHAs, replay comparison, and passing CPU/contract tests. Do not start expensive training before this gate.

### M1 — all-task imitation baseline

Implement the dataset and policy adapters, small-set overfit test, all-task training, and task routing. Run local development evaluation on every family. Save a complete usable baseline even if some tasks remain weak. Exit evidence: loadable checkpoint, eight-task diagnostics, camera/action contract tests, and cold-start inference.

### M2 — cousin coverage

Implement C0 and C1 first, lineage, acceptance/rejection reports, and correction collection. Add C2 only within written approval. Run B0/B1/B2 with fixed-budget comparisons. Exit evidence: verified dataset versions, leakage tests, and holdout improvement or a documented negative result.

### M3 — pilot RL and sampler

Implement R0 on LiftPot and CubeHandover, then S0 with the same interaction budget. Test zero-residual identity, reward/termination handling, replay action identity, deterministic resets, and sampler correctness. Exit evidence: learning curves, execution diagnostics, matched-seed comparison, and complete rollback checkpoints.

### M4 — scale or select a single research branch

Extend successful improvements to all families. Only then authorize M0-memory or T0-transfer work, chosen from failure evidence. Keep research branches separate so a failed integration cannot break the baseline. Exit evidence: frozen per-task checkpoint routing and full local holdout result.

### M5 — reproducibility and submission

Freeze artifacts, run complete official evaluation, write the report from measured results, validate with replay, cold-start on a clean host, package, check anonymous downloads, verify checksums, and obtain owner approval before upload. No agent submits or changes an accepted entry without explicit authorization.

Proposed calendar: M0/M1 on October 14–18; M2/M3 on October 19–25; all-task selection on October 26–28; freeze on October 29; packaging and reproducibility on October 30–31. These are project targets, not estimates of measured runtime. Optional branches are cut first when gates slip.

## 13. Acceptance tests and stop conditions

| Test | Required evidence |
|---|---|
| Source integrity | Expected commit, clean tracked tree/submodules, correct imported package path; no hidden patched install. |
| Seed separation | No official/dev/holdout seed or demonstration lineage in training. |
| Input contract | Exact keys, dtype, shape, camera order, preprocessing parity; no private fields. |
| Action contract | Verified units/order/sign/scaling; finite bounded output; conversion round-trip and replay parity. |
| Reset isolation | Chunk queues, memory, RNG and task routing reset; concurrent episodes cannot share state. |
| RL correctness | Zero-residual base identity, actual-action replay, target masks, executed-step discount, frozen feature versions. |
| Sampler correctness | Full support, normalized finite probabilities, warm-up, stale-window handling, reproducibility. |
| Cousin quality | Lineage, label fidelity, physics execution, correct approval scope, held-out coverage. |
| Score fidelity | Official aggregation; all eight families; no manually edited official result or log. |
| Cold start | Anonymous weight retrieval or staged files, exact revisions/checksums, launchable inference, no secrets. |
| Packaging | Report page limit, complete logs, replay validation, archive size, manifest and portal checksum. |

Stop a branch when its prerequisite fails. Missing weights, approval, or hardware must produce a precise blocker, not fabricated results or a silent substitute model. A no-op controller is allowed only for a plumbing test and must be labeled as such; it is not a completed baseline.

## 14. Deliverables and limitations of this handoff

This starter includes this design, a Word rendering, a kickoff prompt, agent constraints, proposed configurations, organizer questions, report outline, source registry, a standard-library reference sampler/seed guard, and their tests. It does not contain pretrained weights, a RoboEval adapter, a trained policy, or a validated competition bundle.

The agent's first deliverable is the M0 evidence bundle, followed by M1. Report exactly which commands ran, their exit codes, the produced artifacts, test outcomes, and unresolved issues. Do not claim to have reproduced a named paper merely because an inspired component has been implemented.

See `docs/SOURCES.md` for primary references and `KICKOFF_PROMPT.md` for the initial coding-agent instruction.
