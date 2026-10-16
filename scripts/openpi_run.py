"""openpi glue CLI (openpi env: /workspace/roboeval/envs/openpi/bin/python).

Subcommands:
  inspect <config>                 dataset summary, raw/transformed sample, timings
  norm-stats <config> [--max-frames N]
  train <config> --exp-name X [--overwrite] [extra train_pytorch args]
"""
from __future__ import annotations

import argparse
import runpy
import sys
import time
from pathlib import Path

OPENPI_ROOT = Path("/workspace/roboeval/vendor/openpi")


def _install():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from roboeval_entry.training.openpi_bridge import install

    install()


def maybe_install_corr_noise(config_name: str):
    from roboeval_entry.training.openpi_bridge import maybe_install_corr_noise
    return maybe_install_corr_noise(config_name)


def cmd_inspect(config_name: str) -> int:
    import numpy as np
    import openpi.training.data_loader as dl
    import openpi.training.config as _config

    config = _config.get_config(config_name)
    data_config = config.data.create(config.assets_dirs, config.model)
    dataset = dl.create_torch_dataset(data_config, config.model.action_horizon,
                                      config.model)
    print("summary:", dataset.summary())

    s0 = dataset[0]
    print("raw sample:")
    for k, v in s0.items():
        a = np.asarray(v) if not isinstance(v, str) else None
        print(f"  {k}: shape={getattr(a, 'shape', None)} dtype={getattr(a, 'dtype', None)}"
              if a is not None else f"  {k}: {v!r}")

    tds = dl.transform_dataset(dataset, data_config, skip_norm_stats=True)
    ts = tds[0]
    print("transformed sample:")
    def show(prefix, x):
        if isinstance(x, dict):
            for k, v in x.items():
                show(f"{prefix}{k}/", v)
        else:
            a = np.asarray(x)
            print(f"  {prefix[:-1]}: shape={a.shape} dtype={a.dtype}")
    show("", ts)

    t0 = time.perf_counter()
    for i in range(200):
        dataset[i % len(dataset)]
    ms = (time.perf_counter() - t0) / 200 * 1000
    print(f"raw __getitem__: {ms:.2f} ms/sample (200 samples, single process)")

    loader = dl.create_data_loader(config, framework="pytorch", shuffle=True,
                                   num_batches=50, skip_norm_stats=True)
    t0 = time.perf_counter()
    n = 0
    for obs, actions in loader:
        n += obs.to_dict()["state"].shape[0] if hasattr(obs, "to_dict") else 0
    wall = time.perf_counter() - t0
    print(f"loader throughput: {n / wall:.1f} samples/s "
          f"(50 batches x {config.batch_size}, {config.num_workers} workers)")
    return 0


def cmd_norm_stats(config_name: str, extra: list[str]) -> int:
    sys.argv = [str(OPENPI_ROOT / "scripts/compute_norm_stats.py"),
                "--config-name", config_name, *extra]
    runpy.run_path(str(OPENPI_ROOT / "scripts/compute_norm_stats.py"),
                   run_name="__main__")
    return 0


def cmd_train(config_name: str, extra: list[str]) -> int:
    sys.argv = [str(OPENPI_ROOT / "scripts/train_pytorch.py"), config_name, *extra]
    runpy.run_path(str(OPENPI_ROOT / "scripts/train_pytorch.py"),
                   run_name="__main__")
    return 0


def main() -> int:
    _install()
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_inspect = sub.add_parser("inspect")
    p_inspect.add_argument("config")
    p_norm = sub.add_parser("norm-stats")
    p_norm.add_argument("config")
    p_norm.add_argument("rest", nargs=argparse.REMAINDER)
    p_train = sub.add_parser("train")
    p_train.add_argument("config")
    p_train.add_argument("rest", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if args.cmd == "inspect":
        maybe_install_corr_noise(args.config)
        return cmd_inspect(args.config)
    if args.cmd == "norm-stats":
        maybe_install_corr_noise(args.config)
        import roboeval_entry.training.openpi_bridge as _bridge
        _bridge.STATS_ONLY = True
        return cmd_norm_stats(args.config, args.rest)
    if args.cmd == "train":
        maybe_install_corr_noise(args.config)
        return cmd_train(args.config, args.rest)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
