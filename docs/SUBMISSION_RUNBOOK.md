# Submission runbook

This is an operational checklist for the owner/agent, not a prevalidated submission. Follow primary sources S02–S07. Use the official template and schema, not this document as a replacement validator.

## Development

Keep a clean external challenge checkout, exact dependency locks, disjoint datasets, source lineage, and separate local evaluation outputs. Configure a single GPU/worker first. The copied template is only a no-op smoke test until actual inference is implemented.

```bash
# Run in the starter root, with the official simulator environment activated.
export ROBOEVAL_VENDOR=/absolute/path/to/RoboEval
export SIM_PYTHON=/absolute/path/to/simulator-env/bin/python
export EVAL_WORKERS=1
export EVAL_GPUS=0
cp -r "$ROBOEVAL_VENDOR/examples/challenge_submission" ./submission
# Fill template metadata and implement/load the actual model before real evaluation.
bash scripts/run_local_eval.sh ./submission configs/dev_smoke_seeds.json dev ./runs/smoke
```

The local wrapper evaluates all families on the supplied seed list. For a smaller initial two-episode pilot, use the official evaluator directly with `--tasks LiftPot`, the same checked local seed file, and an external output directory. Local-seed runs cannot be submitted as official results.

## Freeze

Record the selected model/checkpoint hashes, camera/action contract, per-task routing, policy seed, dependency locks, data manifests, compute record and approval records. Model selection ends before the official run. Owner reviews the freeze. Avoid `--resume` unless all these hashes match the interrupted run.

## Official evaluation

```bash
# This explicit variable is an internal guard, not organizer approval.
export ALLOW_OFFICIAL_EVALUATION=1
bash scripts/run_official_eval.sh ./submission
```

Confirm coverage of every required episode and metric. Investigate runtime errors rather than manually replacing logs. Do not tune against this run. Write the four-page-plus-references report from actual measurements and place it at `submission/report.pdf`.

## Package and reproduce

```bash
# Large weights: set checkpoint_url in metadata and provide deterministic retrieval.
bash scripts/package_submission.sh ./submission ./team-method-v1.tar.gz --exclude-checkpoint
# For a sufficiently small bundle with included weights, omit the final flag.
```

The wrapper runs official package validation with ten action replays and adds a website-size check. This does not replace a cold-start checkpoint/inference test. Reproduce in a clean environment; verify every external dependency and weight revision. Ensure anonymous downloads work and that the downloaded archive matches the printed SHA-256.

## Owner-only upload

Use one team account in the official Hugging Face portal. Review the archive/report/checksum and access rights before submission. Observe the deadline and UTC daily limit. The most recent valid submission replaces the previous ranked entry; do not upload a worse experimental checkpoint as a casual test. Keep public artifacts immutable during verification.

## Required report evidence

No invented training runs, accepted cousin counts, GPU-hours, or results. Declare extra data, inference compute and routing. The implementation design is not the competition report; the starter archive is not the competition archive.
