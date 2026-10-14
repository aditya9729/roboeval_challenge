#!/usr/bin/env bash
# Local packaging only. No publication or portal submission is performed.
set -euo pipefail
if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: $0 SUBMISSION OUTPUT_TAR_GZ [--exclude-checkpoint]" >&2; exit 2
fi
if [[ $# -eq 3 && "$3" != "--exclude-checkpoint" ]]; then echo "Unknown option" >&2; exit 2; fi
: "${ROBOEVAL_VENDOR:?Set ROBOEVAL_VENDOR}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${SIM_PYTHON:-python}"
[[ -f "$1/report.pdf" ]] || { echo "Missing actual competition report.pdf" >&2; exit 2; }
[[ ! -e "$2" ]] || { echo "Output archive exists; choose a new versioned path" >&2; exit 2; }
"$PY" "$ROOT/tools/assert_vendor.py" "$ROBOEVAL_VENDOR"
ARGS=("$1" --output "$2" --replay 10)
if [[ $# -eq 3 ]]; then ARGS+=(--exclude-checkpoint); fi
"$PY" -m roboeval.challenge.package "${ARGS[@]}"
"$PY" - "$2" "$ROOT/configs/challenge.json" <<'PY'
import hashlib, json, sys
from pathlib import Path
p = Path(sys.argv[1])
limit = json.loads(Path(sys.argv[2]).read_text())["archive_limit_bytes_exclusive"]
if p.stat().st_size >= limit:
    raise SystemExit("Archive is at/over the website's 5 GB limit; do not submit")
h = hashlib.sha256()
with p.open("rb") as f:
    for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
        h.update(block)
print("Local size check passed. Archive SHA-256:", h.hexdigest())
print("Owner must review and authorize any upload/submission.")
PY
