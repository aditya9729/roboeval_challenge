"""Correlated action noise for flow matching (Larchenko et al. §5).

CPU-importable (numpy only); torch is imported lazily inside methods.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np

_ORIG_SAMPLE_NOISE = None


def _sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def _normalize_chunk(chunk: np.ndarray, q01: np.ndarray,
                     q99: np.ndarray) -> np.ndarray:
    """openpi transforms.Normalize._normalize_quantile (no clipping)."""
    return (chunk - q01[:14]) / (q99[:14] - q01[:14] + 1e-6) * 2.0 - 1.0


def estimate_action_correlation(
    variants: Sequence[str],
    split: str,
    registry_path: str | Path,
    norm_stats_path: str | Path,
    action_horizon: int = 16,
    action_dim: int = 32,
    max_chunks: int = 50_000,
    seed: int = 0,
) -> dict:
    """Accumulate mean/covariance of normalized flattened action chunks.

    Returns dict with 'sigma' (512,512 covariance), 'corr' (correlation with
    identity rows/cols for zero-variance padded dims), 'n', 'dims'.
    """
    from roboeval_entry.training.parquet_dataset import RoboEvalParquetDataset
    from roboeval_entry.training.openpi_bridge import DATASET_ROOT

    raw = json.loads(Path(norm_stats_path).read_text())
    stats = raw.get("norm_stats", raw)["actions"]
    q01 = np.asarray(stats["q01"], dtype=np.float64)
    q99 = np.asarray(stats["q99"], dtype=np.float64)

    ds = RoboEvalParquetDataset(
        dataset_root=DATASET_ROOT, variants=variants, split=split,
        registry_path=registry_path, action_horizon=action_horizon,
        actions_only=True)
    n_total = len(ds)
    rng = np.random.default_rng(seed)
    idxs = (rng.choice(n_total, size=min(max_chunks, n_total), replace=False)
            if n_total > max_chunks else np.arange(n_total))

    d = action_horizon * action_dim
    s1 = np.zeros(d, np.float64)
    s2 = np.zeros((d, d), np.float64)
    for idx in idxs:
        chunk = ds[int(idx)]["actions"]  # (H,14)
        normed = _normalize_chunk(chunk.astype(np.float64), q01, q99)
        flat = np.zeros((action_horizon, action_dim), np.float64)
        flat[:, :14] = normed
        v = flat.reshape(-1)
        s1 += v
        s2 += np.outer(v, v)
    n = len(idxs)
    mean = s1 / n
    sigma = s2 / n - np.outer(mean, mean)
    sd = np.sqrt(np.clip(np.diag(sigma), 0, None))
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = sigma / np.outer(sd, sd)
    zero_var = sd == 0
    corr[:, zero_var] = 0.0
    corr[zero_var, :] = 0.0
    corr[np.diag_indices_from(corr)] = 1.0
    return {"sigma": sigma, "corr": corr, "n": n, "dims": d,
            "norm_stats_sha256": _sha256(norm_stats_path)}


class CorrelatedNoise:
    def __init__(self, corr: np.ndarray, beta: float = 0.5,
                 norm_stats_sha256: str | None = None):
        corr = np.asarray(corr, dtype=np.float64)
        m = beta * corr + (1.0 - beta) * np.eye(corr.shape[0])
        self.corr = corr
        self.beta = float(beta)
        self.M = m
        self.L = np.linalg.cholesky(m)
        self.norm_stats_sha256 = norm_stats_sha256

    def transform(self, white):
        import torch

        lead = white.shape[:-2]
        flat = white.reshape(*lead, -1)
        L = torch.from_numpy(self.L).to(device=white.device,
                                      dtype=white.dtype)
        out = L @ flat.unsqueeze(-1)
        return out.reshape(*white.shape)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp.npz")
        np.savez(tmp, corr=self.corr, beta=np.float64(self.beta),
                 L=self.L, M=self.M,
                 norm_stats_sha256=np.asarray(self.norm_stats_sha256 or ""))
        tmp.rename(path)

    @classmethod
    def load(cls, path: str | Path) -> "CorrelatedNoise":
        z = np.load(path, allow_pickle=False)
        cn = cls(z["corr"], float(z["beta"]),
                 str(z["norm_stats_sha256"]) or None)
        np.testing.assert_allclose(cn.L, z["L"], atol=1e-8)
        return cn


def patch_pi0_sample_noise(cn: CorrelatedNoise) -> None:
    """Idempotently wrap PI0Pytorch.sample_noise with correlated noise."""
    global _ORIG_SAMPLE_NOISE
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch

    if _ORIG_SAMPLE_NOISE is None:
        _ORIG_SAMPLE_NOISE = PI0Pytorch.sample_noise
    orig = _ORIG_SAMPLE_NOISE

    def wrapped(self, shape, device):
        return cn.transform(orig(self, shape, device))

    PI0Pytorch.sample_noise = wrapped


def unpatch_pi0_sample_noise() -> None:
    global _ORIG_SAMPLE_NOISE
    if _ORIG_SAMPLE_NOISE is not None:
        from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
        PI0Pytorch.sample_noise = _ORIG_SAMPLE_NOISE
        _ORIG_SAMPLE_NOISE = None
