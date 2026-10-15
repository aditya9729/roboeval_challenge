#!/usr/bin/env bash
# Wait for WAIT_PID to exit, then run CMD. Usage: launch_after.sh PID -- CMD...
set -uo pipefail
WAIT_PID="$1"; shift
[ "${1:-}" = "--" ] && shift
while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
echo "[launch_after] $WAIT_PID exited at $(date -u +%FT%TZ); running: $*"
"$@"
echo "[launch_after] done rc=$?"
