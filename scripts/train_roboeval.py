"""pi0.5 training with dev-split validation (openpi env).

Usage: train_roboeval.py <config> [tyro overrides] [--val.<k> v ...]
  tyro overrides: --exp_name X --overwrite --resume --num_train_steps N
                  --lr_schedule.peak_lr F --optimizer.weight_decay F etc.
  --val.* args are stripped before tyro sees them. All take a value
  except --val.enabled / --val.disabled (flags).
"""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path


def _split_val_args(argv: list[str]) -> tuple[list[str], dict]:
    val_kwargs, rest = {}, []
    bools = {"enabled", "disabled"}
    i = 0
    while i < len(argv):
        a = argv[i]
        if a.startswith("--val."):
            key = a[len("--val."):]
            if key in bools:
                val_kwargs["enabled"] = (key == "enabled")
            elif "=" in a:
                k, v = a[len("--val."):].split("=", 1)
                val_kwargs[k] = v
            else:
                val_kwargs[key] = argv[i + 1]
                i += 1
        else:
            rest.append(a)
        i += 1
    return rest, val_kwargs


def main() -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from roboeval_entry.training.train_loop import ValidationConfig

    rest, val_kwargs = _split_val_args(sys.argv[1:])
    fields = {f.name: f for f in dataclasses.fields(ValidationConfig)}
    typed = {}
    for k, v in val_kwargs.items():
        f = fields[k]
        typed[k] = (int(v) if f.type == "int" else float(v)
                    if f.type == "float" else v)
    val_cfg = ValidationConfig(**typed)

    sys.argv = [sys.argv[0], *rest]
    import openpi.training.config as _config
    from roboeval_entry.training.openpi_bridge import (
        install, maybe_install_corr_noise)
    install()
    config = _config.cli()
    maybe_install_corr_noise(config.name)

    from roboeval_entry.training.train_loop import train_loop
    train_loop(config, val_cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
