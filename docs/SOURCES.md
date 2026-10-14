# Primary-source registry

Checked October 14, 2026. Source-backed challenge requirements and method descriptions are identified in `DESIGN.md` by these keys. Proposed hyperparameters, architecture choices, test budgets, and schedules are our engineering choices, not official rules or reported paper results.

## Official challenge

[S01] RoboEval challenge overview and getting started. Dates, data, interface and installation guidance.  
`https://robo-eval.github.io/challenge.html`

[S02] Evaluation and rules. Source precedence, eligibility, allowed/prohibited behavior, seeds and ranking.  
`https://robo-eval.github.io/challenge-rules.html`

[S03] Submission guidelines. Bundle/report requirements, portal, policy API, size guidance and reproduction.  
`https://robo-eval.github.io/challenge-submission.html`

[S04] Pinned challenge specification. Task keys, budgets, seeds, cameras and metric keys.  
`https://github.com/Robo-Eval/RoboEval/blob/c8b79598c9fb9021652128020517d1847abe4c95/roboeval/challenge/spec.py`

[S05] Pinned evaluator implementation. Worker behavior, reset, policy seeding, action recording and CLI flags.  
`https://github.com/Robo-Eval/RoboEval/blob/c8b79598c9fb9021652128020517d1847abe4c95/roboeval/challenge/evaluate.py`

[S06] Pinned scoring implementation. Human normalization and RDS aggregation.  
`https://github.com/Robo-Eval/RoboEval/blob/c8b79598c9fb9021652128020517d1847abe4c95/roboeval/challenge/scoring.py`

[S07] Pinned packaging implementation. Validation, replay, external checkpoints and manifest construction.  
`https://github.com/Robo-Eval/RoboEval/blob/c8b79598c9fb9021652128020517d1847abe4c95/roboeval/challenge/package.py`

[S08] Challenge-reference resolution via repository API. Resolved SHA was c8b79598c9fb9021652128020517d1847abe4c95.  
`https://api.github.com/repos/Robo-Eval/RoboEval/commits/corl2026-challenge`

[S16] Repository challenge guide. Command workflow, numerical reproducibility and policy contract.  
`https://github.com/Robo-Eval/RoboEval/blob/c8b79598c9fb9021652128020517d1847abe4c95/docs/challenge.md`

## Models and research methods

[S09] NVIDIA GR00T N1.7 3B model card. Verify the exact weight revision and applicable license before downloading or redistributing.  
`https://huggingface.co/nvidia/GR00T-N1.7-3B`

[S10] NVIDIA Isaac-GR00T official repository. Training, embodiment/modality adaptation and serving. Pin the compatible commit; do not treat the moving default branch as reproducible.  
`https://github.com/NVIDIA/Isaac-GR00T`

[S11] Physical Intelligence openpi official repository. The inspected README lists pi0, pi0-FAST and pi0.5 models/checkpoints. This design does not assume a pi0.7 release.  
`https://github.com/Physical-Intelligence/openpi`

[S12] ResFiT, Residual Off-Policy RL for Finetuning Behavior Cloning Policies. Black-box BC base with lightweight off-policy residual control.  
`https://residual-offpolicy-rl.github.io/`

[S13] Success-Guided Sampling project. Adaptive allocation among fixed training configurations; code marked coming soon when checked.  
`https://sgs-rl.github.io/`

[S14] A Balanced Data Diet: Addressing the Exploration Bottleneck in Mega-Scale RL for Robot Control. arXiv:2610.12465. The starter's off-policy, ordinary-reset mixture is an adaptation rather than a paper reproduction.  
`https://arxiv.org/abs/2610.12465`

[S15] IMPRINT, Memory Without Curation: Learning Multi-Stream Latent Memory for Robot Policies. arXiv:2610.14406v1. Consult the method and appendix for causal alignment, reconstruction/progress supervision and backbone integration.  
`https://arxiv.org/html/2610.14406v1`

[S17] Mulligan: Performance-Guided Data Collection for Efficient On-Robot Learning. arXiv:2610.05882. Motivation for a distinct supervised correction queue.  
`https://arxiv.org/abs/2610.05882`

[S18] ACCRUE official project. Continual actor-critic inheritance and prior-behavior retention; distinct simulation and hardware setups. Code marked coming soon when checked.  
`https://accrue-rl.github.io/`

## Submission portal

The guidelines link to:  
`https://huggingface.co/spaces/helen9975/roboeval-challenge`

No portal write, registration, upload or organizer message was performed while preparing this handoff. Recheck the official source and announcements before a final submission. A code constant left unset does not waive a published rule.
