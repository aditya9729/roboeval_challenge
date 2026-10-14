#!/usr/bin/env bash
# Does not upload. Run only after owner-approved model/configuration freeze.
set -euo pipefail
if [[ $# -ne 1 ]]; then echo "Usage: $0 SUBMISSION" >&2; exit 2; fi
if [[ "${ALLOW_OFFICIAL_EVALUATION:-0}" != 1 ]]; then
  echo "Freeze model selection first; then explicitly set ALLOW_OFFICIAL_EVALUATION=1" >&2
  exit 2
fi
: "${ROBOEVAL_VENDOR:?Set ROBOEVAL_VENDOR}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${SIM_PYTHON:-python}"
"$PY" "$ROOT/tools/assert_vendor.py" "$ROBOEVAL_VENDOR"
ARGS=(--submission "$1" --workers "${EVAL_WORKERS:-1}")
if [[ -n "${EVAL_GPUS:-}" ]]; then ARGS+=(--gpus "$EVAL_GPUS"); fi
export MUJOCO_GL="${MUJOCO_GL:-egl}"
"$PY" -m roboeval.challenge.evaluate "${ARGS[@]}"
# No --resume/--overwrite by default: avoid mixing or destroying prior evidence.
