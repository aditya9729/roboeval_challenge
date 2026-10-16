"""Optuna TPE HPO driver over train_roboeval.py trials (openpi env).

Objective: min cont_mse_ratio from each trial's val_history.jsonl.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = "/workspace/roboeval/envs/openpi/bin/python"
HPO_DIR = Path("/workspace/roboeval/runs/HPO01")
TRIAL_STEPS = 1500
VAL_INTERVAL = 250


def suggest(trial) -> dict:
    return {
        "peak_lr": trial.suggest_float("peak_lr", 1e-5, 1e-4, log=True),
        "warmup_steps": trial.suggest_int("warmup_steps", 200, 2000, step=100),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-1,
                                            log=True),
        "ema_decay": trial.suggest_categorical("ema_decay",
                                               [None, 0.99, 0.999]),
        "batch_size": trial.suggest_categorical("batch_size", [8, 16, 32]),
    }


def trial_cmd(p: dict, trial_dir: Path, exp_name: str) -> list[str]:
    cmd = [PY, str(ROOT / "scripts/train_roboeval.py"), "pi05_roboeval_pr8",
           "--exp_name", exp_name, "--overwrite",
           "--num_train_steps", str(TRIAL_STEPS),
           "--lr_schedule.decay_steps", str(TRIAL_STEPS),
           "--lr_schedule.peak_lr", repr(p["peak_lr"]),
           "--lr_schedule.warmup_steps", str(p["warmup_steps"]),
           "--optimizer.weight_decay", repr(p["weight_decay"]),
           "--batch_size", str(p["batch_size"]),
           "--save_interval", str(TRIAL_STEPS),
           "--checkpoint_base_dir", str(trial_dir / "ckpts"),
           "--val.interval", str(VAL_INTERVAL)]
    if p["ema_decay"] is not None:
        cmd += ["--ema_decay", repr(p["ema_decay"])]
    return cmd


def cleanup_ckpt_dir(ckpt_dir: Path) -> None:
    """Keep only best/ + val_history + state; delete numeric step dirs."""
    for d in ckpt_dir.iterdir() if ckpt_dir.exists() else []:
        if d.is_dir() and (d.name.isdigit() or d.name.startswith("tmp_")):
            shutil.rmtree(d)


def run_trial(trial) -> float:
    import optuna
    exp = f"hpo01_t{trial.number:03d}"
    tdir = HPO_DIR / f"t{trial.number:03d}"
    tdir.mkdir(parents=True, exist_ok=True)
    p = suggest(trial)
    ckpt_dir = tdir / "ckpts" / "pi05_roboeval_pr8" / exp
    hist = ckpt_dir / "val_history.jsonl"
    cmd = trial_cmd(p, tdir, exp)
    (tdir / "params.json").write_text(json.dumps(p))
    with open(tdir / "stdout.txt", "w") as out:
        proc = subprocess.Popen(cmd, stdout=out,
                                stderr=subprocess.STDOUT, cwd=ROOT)
    best = float("inf")
    seen = set()
    try:
        while proc.poll() is None:
            time.sleep(10)
            if hist.exists():
                for line in hist.read_text().splitlines():
                    r = json.loads(line)
                    if r["step"] in seen:
                        continue
                    seen.add(r["step"])
                    obj = r["cont_mse_ratio"]
                    best = min(best, obj)
                    trial.report(obj, r["step"])
            if trial.should_prune():
                proc.terminate()
                try:
                    proc.wait(60)
                except subprocess.TimeoutExpired:
                    proc.kill()
                # keep artifacts, mark pruned
                (tdir / "PRUNED").write_text(
                    f"step={max(seen) if seen else -1}")
                raise optuna.TrialPruned()
    finally:
        if proc.poll() is None:
            proc.terminate()
        cleanup_ckpt_dir(ckpt_dir)
    # tail any final history lines
    if hist.exists():
        for line in hist.read_text().splitlines():
            r = json.loads(line)
            if r["step"] not in seen:
                seen.add(r["step"])
                best = min(best, r["cont_mse_ratio"])
    if best == float("inf"):
        raise optuna.TrialPruned("no validation recorded")
    return best


def main() -> int:
    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-trials", type=int, default=20)
    ap.add_argument("--timeout-hours", type=float, default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.dry_run:
        study = optuna.create_study(direction="minimize",
                                    sampler=TPESampler(seed=0,
                                                       multivariate=True))
        for i in range(5):
            t = study.ask()
            print(json.dumps(suggest(t)))
            study.tell(t, 0.5)
        return 0

    HPO_DIR.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=0, multivariate=True),
        pruner=MedianPruner(n_startup_trials=4, n_warmup_steps=2),
        storage=f"sqlite:///{HPO_DIR}/study.db",
        study_name="HPO01_pr8", load_if_exists=True)

    def record(study, trial):
        with open(HPO_DIR / "trials.tsv", "a") as f:
            f.write("\t".join([
                str(trial.number), str(trial.state.name),
                str(trial.value), json.dumps(trial.params)]) + "\n")

    study.optimize(run_trial, n_trials=args.n_trials,
                   timeout=(args.timeout_hours * 3600
                            if args.timeout_hours else None),
                   callbacks=[record])
    done = [t for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE]
    if len(done) >= 4:
        imp = optuna.importance.get_param_importances(study)
        (HPO_DIR / "importance.json").write_text(json.dumps(imp, indent=1))
        print("param importances:", imp)
    print("best:", study.best_trial.number, study.best_value,
          study.best_params)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
