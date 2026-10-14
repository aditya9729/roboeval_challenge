# Coding-agent kickoff

You are implementing Nirvana Robotics' CoRL 2026 RoboEval entry. Read `AGENTS.md`, `DESIGN.md`, `docs/SOURCES.md`, and the configs before editing code. This package is a specification plus small tested utilities, not a robot policy. Implement the design in milestone order; do not stop at another plan.

**First objective: finish M0, then start M1.** Pin the authoritative challenge reference to the expected commit in `configs/challenge.json`. Inspect the actual observation/action source, demonstration schema, model APIs, reward semantics, and environment dependencies. Produce an auditable action contract and data-split registry; never guess coordinate conventions or scaling.

Use GR00T N1.7 as the main baseline and retain a pi0.5 fallback. Separate policy dependencies from the official simulator environment. Copy the official submission template. Run a no-op plumbing check only on our non-official seed pool, then verify original versus converted demonstration replay. Build an adapter and small-set overfit test before expensive training. Establish an all-eight-task imitation baseline before pilot RL.

The implementation sequence is: all-task base → appearance augmentation and executable spatial cousins → ResFiT-inspired per-step residual RL → SGS-inspired reset allocation. Later compare IMPRINT and ACCRUE in isolated branches only after their prerequisites. Do not add Cosmos, change simulators, implement mobile-base control, or integrate all papers simultaneously.

Keep the official benchmark checkout unmodified. Never put ground-truth object poses, contacts, solver state, seed IDs, or episode IDs into the policy. Never train or select on official seeds 1000–1049. Group every generated descendant with its original demonstration's split. Geometry/dynamics modifications, privileged teachers, intermediate-state resets, and changed training rewards are disabled until documented organizer authorization. Continue ordinary-reset and imitation work while awaiting clarification.

Begin by running `python -m unittest discover -s tests -v` on the supplied reference utilities. They are not evidence of simulator compatibility. Extend them with tests for action mapping, reset isolation, replay identity, episode boundaries, feature-version consistency, and score aggregation. A no-op test is not a successful learned baseline.

For each milestone, produce the changed files, exact commands and exit codes, test results, artifact paths and hashes, hardware/compute record, and unresolved blockers. Use `reports/M0_EVIDENCE.md` for the first report. Do not invent results, checkpoint revisions, API calls, paper-reproduction claims, or approvals. Do not silently change models when a dependency fails. Do not upload or submit anything without the owner's explicit authorization.

Completion of the first assignment means: resolved dependency/source locks; verified interface contract; disjoint seed and data manifests; a two-episode non-official plumbing run; direct-versus-adapted demonstration replay evidence; a policy adapter skeleton with meaningful tests; and a clear runnable command for the next M1 training step.
