"""Checkpoint retention pruner (stdlib-only, any env).

openpi run dir layout: <run>/<step>/{model.safetensors,optimizer.pt,
metadata.pt,assets/}; in-progress saves live in tmp_<step>/.

Retention rule:
  - optimizer.pt is kept only in the newest K COMPLETE numeric step
    checkpoints (default K=1) and deleted everywhere else. Non-numeric
    complete dirs (e.g. "best") never count toward K — their optimizer is
    pruned too (inference only needs safetensors + assets).
  - tmp_* dirs older than --tmp-max-age (default 1 h) are removed
    (interrupted saves).

model.safetensors / metadata.pt / assets/ are never touched.
"""
from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path


def _is_complete(d: Path) -> bool:
    return d.is_dir() and not d.name.startswith("tmp_") \
        and (d / "model.safetensors").exists()


def prune_run_dir(run_dir: Path, keep_optimizer: int = 1,
                  tmp_max_age_s: float = 3600.0,
                  dry_run: bool = False,
                  log=print) -> list[tuple[Path, int]]:
    """Apply the retention rule; returns [(deleted_path, bytes)]."""
    run_dir = Path(run_dir)
    deleted: list[tuple[Path, int]] = []
    if not run_dir.is_dir():
        return deleted

    numeric = sorted(
        (int(d.name) for d in run_dir.iterdir()
         if d.name.isdigit() and _is_complete(d)),
        reverse=True)
    keep_steps = set(numeric[:max(0, keep_optimizer)])

    for d in sorted(run_dir.iterdir()):
        if d.name.startswith("tmp_"):
            age = time.time() - d.stat().st_mtime
            if d.is_dir() and age > tmp_max_age_s:
                size = sum(f.stat().st_size for f in d.rglob("*")
                           if f.is_file())
                if not dry_run:
                    shutil.rmtree(d)
                deleted.append((d, size))
                log(f"delete stale tmp dir {d} ({size / 1e9:.2f} GB)")
            continue
        if not _is_complete(d):
            continue
        if d.name.isdigit() and int(d.name) in keep_steps:
            continue
        opt = d / "optimizer.pt"
        if opt.exists():
            size = opt.stat().st_size
            if not dry_run:
                opt.unlink()
            deleted.append((opt, size))
            log(f"delete {opt} ({size / 1e9:.2f} GB)")
    return deleted


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir")
    p.add_argument("--keep-optimizer", type=int, default=1)
    p.add_argument("--tmp-max-age", type=float, default=3600.0,
                   help="seconds; tmp_* dirs older than this are deleted")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--watch", type=float, default=None,
                   help="loop every SECONDS")
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    while True:
        prune_run_dir(run_dir, keep_optimizer=args.keep_optimizer,
                      tmp_max_age_s=args.tmp_max_age,
                      dry_run=args.dry_run)
        if args.watch is None:
            break
        time.sleep(args.watch)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
