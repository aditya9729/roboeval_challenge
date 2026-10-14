#!/usr/bin/env bash
# Requires an installed official simulator environment and a staged policy bundle.
set -euo pipefail
if [[ $# -ne 4 ]]; then
  echo "Usage: $0 SUBMISSION SEEDS_JSON {dev|holdout} OUTPUT_DIR" >&2; exit 2
fi
: "${ROBOEVAL_VENDOR:?Set ROBOEVAL_VENDOR to the external clean checkout}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${SIM_PYTHON:-python}"
"$PY" "$ROOT/tools/assert_vendor.py" "$ROBOEVAL_VENDOR"
"$PY" "$ROOT/tools/check_seed_file.py" "$2" --split "$3"
ARGS=(--submission "$1" --seeds-file "$2" --output-dir "$4" --workers "${EVAL_WORKERS:-1}")
if [[ -n "${EVAL_GPUS:-}" ]]; then ARGS+=(--gpus "$EVAL_GPUS"); fi
export MUJOCO_GL="${MUJOCO_GL:-egl}"
"$PY" -m roboeval.challenge.evaluate "${ARGS[@]}"
# A non-official-seed result is analysis-only and must not be packaged as official.
