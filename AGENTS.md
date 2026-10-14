# Agent instructions

## Priority
Ship a reproducible eight-task RoboEval baseline, then improve it through measured experiments. Follow `DESIGN.md`. The main comparison is uniform-reset residual RL versus the same learner with SGS-inspired sampling.

## Non-negotiable boundaries
- Official benchmark code, assets, physics, reward, termination, metrics and limits remain unmodified.
- Training/model-selection data never includes official seeds 1000–1049 or local dev/holdout pools. Descendant data inherits the parent's split.
- Inference consumes only standard declared observations and allowed task context. Seeds may seed stochasticity, but are not learned inputs or lookup keys.
- Approval-gated features are disabled by default. Record written scope; lack of a validator check is not permission.
- No manually authored official `results.json`, episode logs, manifest, or fabricated report numbers.
- No external writes, account actions, public upload, or competition submission without explicit owner approval.
- No secrets in source, documents, model bundles, shell command logs, or dataset manifests.

## Engineering
Read the actual pinned APIs before implementing adapters. Fail on an unexpected contract or missing checkpoint. Keep training and evaluation environments separate. Pin all repositories/checkpoints/dependencies and record licenses separately for code and weights. Do not assume a model name implies that a matching checkpoint exists.

Start with CPU tests and small-data plumbing. Run training only after action and replay contracts pass. Keep all eight tasks represented. Use typed interfaces, finite-value checks, atomic outputs, meaningful errors, deterministic seeds, and artifact hashes. Test exact actions sent, not intended plans.

Use one implementation lane at a time. Reviewers verify tests and compliance independently of the component author when multiple agents are available. Do not spawn unbounded jobs or spend compute without a configured budget.

## Completion reports
Report commands actually run, exit codes, test counts, evidence paths, compute usage, and blockers. Separate source facts, design choices, hypotheses, and measurements. Preserve negative results. Never label scaffolding as a trained or evaluated robot system.
