#!/usr/bin/env bash
# Source preparation only; does not install dependencies or alter existing trees.
set -euo pipefail
if [[ $# -ne 1 ]]; then
  echo "Usage: bash scripts/prepare_vendor.sh /absolute/path/to/RoboEval" >&2
  exit 2
fi
TARGET="$1"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${CONFIG_PYTHON:-python3}"
EXPECTED="$($PY -c 'import json,sys; print(json.load(open(sys.argv[1]))["expected_commit"])' "$ROOT/configs/challenge.json")"
if [[ "$TARGET" != /* ]]; then echo "Use an absolute destination path" >&2; exit 2; fi
if [[ -e "$TARGET" ]]; then
  echo "Destination already exists; refusing to overwrite or reset it: $TARGET" >&2
  exit 2
fi
git clone --no-checkout https://github.com/Robo-Eval/RoboEval.git "$TARGET"
git -C "$TARGET" checkout --detach "$EXPECTED"
git -C "$TARGET" submodule update --init --recursive
[[ "$(git -C "$TARGET" rev-parse HEAD)" == "$EXPECTED" ]]
[[ -z "$(git -C "$TARGET" status --porcelain --untracked-files=normal)" ]]
echo "Prepared $TARGET at $EXPECTED. Install using the official Python 3.10 instructions."
