# Competition report outline — not submission-ready

The official report is a CoRL-format PDF, at most four pages plus references (S03). This outline and the implementation design are not substitutes for that PDF. No results are available in this starter.

## Page 1: task, method, and contribution
Describe the bimanual-Panda task suite and objective. State the implemented backbone, exact checkpoint revision, camera/action contract, action execution scheme, and only the algorithmic components actually used. Clearly call adapted methods "inspired by" unless faithfully reproduced.

## Page 2: training data and optimization
State source demonstrations, split methodology, accepted generated trajectories, correction data, pretraining, and licenses. Explain cousin verification and excluded failures. Describe residual learning and reset sampling, including all departures from source methods. Record optimizer settings, interaction budget, hardware and GPU-hours from logs.

## Page 3: results and ablations
Use measured eight-task RDS, success and progression, plus safety/efficiency/coordination. Compare base, verified cousins, uniform-reset RL and SGS under matched budgets. Include uncertainty where supported. Identify local versus official seed sets clearly. Never select checkpoints using official results.

## Page 4: diagnostics, limitations, and reproduction
Analyze missed grasps, slips, coordination, recovery and excess motion using actual rollouts. State negative results and omitted branches. Describe environment/model locks, test-time compute, task routing, cold-start reproduction, and artifact availability.

## References
Cite RoboEval and the specific pretrained models, datasets and methods used. Review model-code and model-weight licensing separately before checking the open-source-award field.

## Required preflight
No placeholder result tables, invented gains, or unexecuted experiments; correct page count; exact interface; truthful GPU-hour reporting; matching bundle hashes; CoRL formatting; readable diagnostic plots. Generate `submission/report.pdf` only after measurements exist.
