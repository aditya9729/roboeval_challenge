#!/usr/bin/env bash
# Watch a checkpoint run dir; evaluate each requested step on dev seeds as it
# appears. Usage: watch_eval_checkpoints.sh RUN_DIR OUT_PREFIX STEPS...
set -uo pipefail
RUN_DIR="$1"; OUT_PREFIX="$2"; shift 2
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SUMMARY="${OUT_PREFIX}_summary.tsv"
FAMS="LiftPot LiftTray StackTwoBlocks StackSingleBookShelf CubeHandover PackBox PickSingleBookFromTable RotateValve"

[ -f "$SUMMARY" ] || printf 'step\texit\twall_s\tscore\toutcome\tsafety\tefficiency\tcoordination\tsuccess\t%s\n' \
  "$(for f in $FAMS; do printf '%s_success\t%s_tp\t' "$f" "$f"; done)" >> "$SUMMARY"

for STEP in "$@"; do
  CKPT="$RUN_DIR/$STEP/model.safetensors"
  echo "[watch] waiting for $CKPT"
  while :; do
    if [ -f "$CKPT" ]; then
      S1=$(stat -c %s "$CKPT"); sleep 60
      S2=$(stat -c %s "$CKPT" 2>/dev/null || echo 0)
      [ "$S1" = "$S2" ] && [ "$S2" != "0" ] && break
    fi
    sleep 60
  done
  echo "[watch] $CKPT stable; evaluating"
  T0=$(date +%s)
  bash "$ROOT/scripts/eval_dev_pi05.sh" pi05_roboeval_pr8 "$RUN_DIR/$STEP" \
    "$ROOT/configs/dev_eval_seeds.json" "${OUT_PREFIX}_${STEP}"
  RC=$?
  WALL=$(( $(date +%s) - T0 ))
  python3 - "$STEP" "$RC" "$WALL" "${OUT_PREFIX}_${STEP}/results.json" >> "$SUMMARY" <<'EOF'
import json, sys
step, rc, wall, rpath = sys.argv[1:5]
FAMS = ["LiftPot","LiftTray","StackTwoBlocks","StackSingleBookShelf",
        "CubeHandover","PackBox","PickSingleBookFromTable","RotateValve"]
try:
    d = json.load(open(rpath))
    sc = d["score"]["overall"]; pf = d["summary"]["per_family"]
    succ = sum(pf[f]["success"] for f in FAMS) / len(FAMS)
    cols = [sc["score"], sc["outcome"], sc["safety"], sc["efficiency"],
            sc["coordination"], succ]
    for f in FAMS:
        cols += [pf[f]["success"], pf[f]["tp"]]
    row = "\t".join(f"{c:.6g}" for c in cols)
except Exception as e:
    row = f"NO_RESULTS({e})"
print(f"{step}\t{rc}\t{wall}\t{row}")
EOF
  echo "[watch] $STEP done rc=$RC wall=${WALL}s"
done
echo "[watch] all steps processed"
