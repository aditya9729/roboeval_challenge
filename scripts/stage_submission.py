"""Stage the submission bundle (stdlib-only; any env).

Copies the policy package + action contract into submission/ and writes
checkpoint/pi05_server.json describing the inference server endpoint.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def _copy_tree(src: Path, dst: Path) -> None:
    def ignore(dir_, names):
        return {n for n in names
                if n == "__pycache__" or n.endswith(".pyc")
                or (Path(dir_) == src and n == "training")}
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=ignore)


def _atomic_write(path: Path, data: str) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp_")
    with os.fdopen(fd, "w") as f:
        f.write(data)
    os.replace(tmp, path)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint-dir", required=True)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--execute-steps", type=int, default=4)
    p.add_argument("--gripper-rule", action="store_true")
    p.add_argument("--gripper-rule-params", default=None)
    p.add_argument("--inpaint-steps", type=int, default=0)
    args = p.parse_args()

    ckpt = Path(args.checkpoint_dir)
    weights = ckpt / "model.safetensors"
    if not weights.exists():
        raise FileNotFoundError(weights)

    sub = ROOT / "submission"
    _copy_tree(ROOT / "src" / "roboeval_entry",
               sub / "policy" / "roboeval_entry")
    shutil.copy2(ROOT / "contracts" / "action_spec.json",
                 sub / "policy" / "action_spec.json")

    info = {
        "config": args.config,
        "checkpoint_dir": str(ckpt),
        "host": args.host,
        "port": args.port,
        "execute_steps": args.execute_steps,
        "inpaint_steps": args.inpaint_steps,
        "model_sha256": _sha256(weights),
        "gripper_rule": (json.loads(args.gripper_rule_params)
                         if args.gripper_rule and args.gripper_rule_params
                         else ({} if args.gripper_rule else None)),
        "staged_at": datetime.now(timezone.utc).isoformat(),
    }
    out = sub / "checkpoint" / "pi05_server.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(out, json.dumps(info, indent=1))
    print(json.dumps(info, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
