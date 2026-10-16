"""Serial norm-stats computation (openpi env), equivalent to vendor
scripts/compute_norm_stats.py but without torch DataLoader spawn workers —
the cgroup pids.max budget cannot host num_workers x OpenBLAS thread pools.
Same transform chain (repack + data_transforms.inputs + RemoveStrings) and
the same openpi.shared.normalize.RunningStats math; writes the same file.

Usage: norm_stats_local.py <config>
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np


class _RemoveStrings:
    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items()
                if not np.issubdtype(np.asarray(v).dtype, np.str_)}


def main() -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    import roboeval_entry.training.openpi_bridge as bridge

    bridge.install()
    bridge.STATS_ONLY = True  # 1x1 placeholder images; stats use state/actions
    bridge.maybe_install_corr_noise(sys.argv[1])

    import openpi.shared.normalize as normalize
    import openpi.training.config as _config
    import openpi.training.data_loader as dl
    import tqdm

    config = _config.get_config(sys.argv[1])
    data_config = config.data.create(config.assets_dirs, config.model)
    dataset = dl.create_torch_dataset(
        data_config, config.model.action_horizon, config.model)
    tds = dl.TransformedDataset(
        dataset,
        [*data_config.repack_transforms.inputs,
         *data_config.data_transforms.inputs,
         _RemoveStrings()])

    stats = {k: normalize.RunningStats() for k in ("state", "actions")}
    t0 = time.time()
    for i in tqdm.trange(len(tds), desc="Computing stats"):
        item = tds[i]
        for k in stats:
            stats[k].update(np.asarray(item[k])[None])
    norm_stats = {k: s.get_statistics() for k, s in stats.items()}
    output_path = config.assets_dirs / data_config.repo_id
    print(f"Writing stats to: {output_path}")
    normalize.save(output_path, norm_stats)
    print(f"done in {time.time() - t0:.1f}s over {len(tds)} samples")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
