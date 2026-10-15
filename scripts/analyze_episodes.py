"""Episode-log analysis across run dirs (py3.12/sim env, numpy only)."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

RAW_MEAN_KEYS = ["completion_time", "slip_count", "env_collision_count",
                 "self_collision_count", "target_distance"]
FAM_ORDER = ["LiftPot", "LiftTray", "StackTwoBlocks", "StackSingleBookShelf",
             "CubeHandover", "PackBox", "PickSingleBookFromTable", "RotateValve"]


def _family_runs(run_dir: Path):
    for fam_dir in sorted((run_dir / "logs").glob("*_PositionAndOrientation")):
        eps = []
        for ep_file in sorted(fam_dir.glob("episode_*.json")):
            ep = json.loads(ep_file.read_text())
            npz = ep_file.with_suffix(".npz")
            ep["_actions"] = np.load(npz)["actions"] if npz.exists() else None
            eps.append(ep)
        yield fam_dir.name.replace("_PositionAndOrientation", ""), eps


def _stats(eps):
    raw = [e["raw"] for e in eps]
    out = {
        "n": len(eps),
        "success": float(np.mean([r.get("success", 0.0) for r in raw])),
        "tp": float(np.mean([r.get("subtask_progress", 0.0) for r in raw])),
        "steps": float(np.mean([e["steps"] for e in eps])),
        "terminations": dict(Counter(e["termination"] for e in eps)),
        "stages": dict(Counter(
            str(max([int(k) for k, v in r.get("task_stage_reached", {}).items()
                     if v], default=0)) for r in raw)),
    }
    for k in RAW_MEAN_KEYS:
        vals = []
        for r in raw:
            v = r.get(k, 0.0)
            if isinstance(v, dict):
                nums = [x for x in v.values() if isinstance(x, (int, float))]
                v = float(np.mean(nums)) if nums else 0.0
            vals.append(float(v))
        out[k] = float(np.mean(vals))
    toggles, closed_frac, pos_abs, rot_abs = [], [], [], []
    for e in eps:
        a = e["_actions"]
        if a is None or not len(a):
            continue
        g = (a[:, 12:14] > 0.5).astype(int)
        toggles.append(int(np.abs(np.diff(g, axis=0)).sum()))
        closed_frac.append(float(g.mean()))
        pos_abs.append(float(np.abs(a[:, [0, 1, 2, 6, 7, 8]]).mean()))
        rot_abs.append(float(np.abs(a[:, [3, 4, 5, 9, 10, 11]]).mean()))
    out["grip_toggles"] = float(np.mean(toggles)) if toggles else float("nan")
    out["grip_closed_frac"] = (float(np.mean(closed_frac))
                               if closed_frac else float("nan"))
    out["pos_abs"] = float(np.mean(pos_abs)) if pos_abs else float("nan")
    out["rot_abs"] = float(np.mean(rot_abs)) if rot_abs else float("nan")
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dirs", nargs="+", type=Path)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    runs = {}
    for rd in args.run_dirs:
        runs[rd.name] = {fam: _stats(eps) for fam, eps in _family_runs(rd)}

    fams = sorted({f for r in runs.values() for f in r},
                  key=lambda f: FAM_ORDER.index(f) if f in FAM_ORDER else 99)
    labels = list(runs)
    lines = ["# Episode analysis", "",
             "runs: " + ", ".join(f"`{l}`" for l in labels), ""]

    def row(fam, key, fmt="{:.3f}"):
        return "| " + key + " | " + " | ".join(
            fmt.format(runs[l].get(fam, {}).get(key, float("nan")))
            if isinstance(runs[l].get(fam, {}).get(key), (int, float))
            else str(runs[l].get(fam, {}).get(key)) for l in labels) + " |"

    for fam in fams:
        lines += [f"## {fam}", "",
                  "| metric | " + " | ".join(labels) + " |",
                  "|---|" + "---|" * len(labels)]
        for k, fmt in [("n", "{}"), ("success", "{:.3f}"), ("tp", "{:.3f}"),
                       ("steps", "{:.1f}"), ("completion_time", "{:.2f}"),
                       ("slip_count", "{:.2f}"), ("env_collision_count", "{:.2f}"),
                       ("self_collision_count", "{:.2f}"),
                       ("target_distance", "{:.4f}"),
                       ("grip_toggles", "{:.1f}"),
                       ("grip_closed_frac", "{:.3f}"), ("pos_abs", "{:.5f}"),
                       ("rot_abs", "{:.5f}")]:
            lines.append(row(fam, k, fmt))
        for key, name in [("terminations", "terminations"),
                          ("stages", "task_stage_reached (max) hist")]:
            cells = []
            for l in labels:
                d = runs[l].get(fam, {}).get(key, {})
                cells.append(", ".join(f"{k}:{v}" for k, v in sorted(d.items())))
            lines.append(f"| {name} | " + " | ".join(cells) + " |")
        lines.append("")

    lines += ["## score blocks", "", "| field | " + " | ".join(labels) + " |",
              "|---|" + "---|" * len(labels)]
    for field in ["score", "outcome", "safety", "efficiency", "coordination"]:
        cells = []
        for l in labels:
            res = Path(args.run_dirs[labels.index(l)]) / "results.json"
            v = (json.loads(res.read_text())["score"]["overall"].get(field)
                 if res.exists() else None)
            cells.append(f"{v:.5f}" if isinstance(v, float) else str(v))
        lines.append(f"| {field} | " + " | ".join(cells) + " |")

    text = "\n".join(lines) + "\n"
    args.out.write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
