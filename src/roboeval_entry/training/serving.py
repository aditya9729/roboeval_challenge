"""Seeded policy wrapper for deterministic openpi inference (openpi env)."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np


class SeededPolicy:
    """Wraps an openpi Policy; pops ``noise_seed`` from obs to drive the
    diffusion noise deterministically. Without a seed it falls through."""

    def __init__(self, policy, action_horizon: int = 16, action_dim: int = 32):
        self._policy = policy
        self.action_horizon = action_horizon
        self.action_dim = action_dim

    def _noise(self, seed: int) -> np.ndarray:
        import torch

        g = torch.Generator().manual_seed(seed)
        n = torch.randn((self.action_horizon, self.action_dim), generator=g)
        return n.numpy().astype(np.float32)

    def infer(self, obs: dict, **kwargs: Any) -> dict:
        obs = dict(obs)
        seed = obs.pop("noise_seed", None)
        if seed is not None and "noise" not in kwargs:
            kwargs["noise"] = self._noise(int(seed))
        return self._policy.infer(obs, **kwargs)

    def __getattr__(self, name: str):
        return getattr(self._policy, name)


def model_sha256(checkpoint_dir: str | Path) -> str:
    path = Path(checkpoint_dir) / "model.safetensors"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()
