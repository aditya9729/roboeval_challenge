#!/usr/bin/env bash
# Serve a pi05 checkpoint (openpi env) and run the official evaluator on a
# local seed file (sim env). Usage: eval_dev_pi05.sh CONFIG CKPT_DIR SEEDS_JSON OUT_DIR [PORT]
set -uo pipefail
CONFIG="$1"; CKPT="$2"; SEEDS="$3"; OUT="$4"; PORT="${5:-8000}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OPENPI_PY=/workspace/roboeval/envs/openpi/bin/python
SIM_PY=/workspace/roboeval/envs/sim/bin/python
mkdir -p "$OUT"

"$OPENPI_PY" "$ROOT/scripts/serve_pi05.py" --config "$CONFIG" \
  --checkpoint-dir "$CKPT" --port "$PORT" --host 127.0.0.1 \
  > "$OUT/server.log" 2>&1 &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null' EXIT

"$SIM_PY" - "$PORT" <<'EOF'
import sys, time
port = int(sys.argv[1])
from openpi_client import websocket_client_policy
deadline = time.time() + 600
while True:
    try:
        c = websocket_client_policy.WebsocketClientPolicy(host="127.0.0.1", port=port)
        print("server metadata:", c.get_server_metadata())
        break
    except Exception as e:
        if time.time() > deadline:
            raise SystemExit(f"server not ready after 600s: {e}")
        time.sleep(5)
EOF
READY=$?
[ $READY -ne 0 ] && exit $READY

"$SIM_PY" "$ROOT/scripts/stage_submission.py" --config "$CONFIG" \
  --checkpoint-dir "$CKPT" --port "$PORT" || exit $?

ROBOEVAL_VENDOR=/workspace/roboeval/vendor/RoboEval \
SIM_PYTHON="$SIM_PY" EVAL_WORKERS=8 EVAL_GPUS=0 MUJOCO_GL=egl \
  bash "$ROOT/scripts/run_local_eval.sh" "$ROOT/submission" "$SEEDS" dev "$OUT"
