"""Seeded pi0.5 inference server (openpi env). Localhost only."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    from roboeval_entry.training.openpi_bridge import (
        install, maybe_install_corr_noise)
    install()
    corr_noise = maybe_install_corr_noise(args.config)

    import openpi.policies.policy_config as policy_config
    import openpi.serving.websocket_policy_server as ws
    import openpi.training.config as _config
    from roboeval_entry.training.serving import SeededPolicy, model_sha256

    config = _config.get_config(args.config)
    policy = policy_config.create_trained_policy(config, args.checkpoint_dir)
    seeded = SeededPolicy(policy, action_horizon=config.model.action_horizon,
                          action_dim=config.model.action_dim,
                          corr_noise=corr_noise)
    server = ws.WebsocketPolicyServer(
        policy=seeded, host=args.host, port=args.port,
        metadata={"config": args.config,
                  "checkpoint_dir": args.checkpoint_dir,
                  "model_sha256": model_sha256(args.checkpoint_dir)})
    print(f"serving {args.config} @ {args.host}:{args.port}", flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
