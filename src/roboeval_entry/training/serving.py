"""Seeded policy wrapper for deterministic openpi inference (openpi env)."""
from __future__ import annotations

import functools
import hashlib
from pathlib import Path
from typing import Any

import numpy as np


class SeededPolicy:
    """Wraps an openpi Policy; pops ``noise_seed`` from obs to drive the
    diffusion noise deterministically. ``prefix_actions`` (k,14 raw actions)
    enables plain soft-inpainting of the chunk boundary (no correlation-aware
    propagation). Without either key it falls through to the wrapped policy."""

    def __init__(self, policy, action_horizon: int = 16, action_dim: int = 32,
                 inpaint_time_threshold: float = 0.3, corr_noise=None):
        self._policy = policy
        self.action_horizon = action_horizon
        self.action_dim = action_dim
        self.inpaint_time_threshold = inpaint_time_threshold
        self.corr_noise = corr_noise

    def _noise(self, seed: int) -> np.ndarray:
        import torch

        g = torch.Generator().manual_seed(seed)
        n = torch.randn((1, self.action_horizon, self.action_dim),
                        generator=g)
        if self.corr_noise is not None:
            n = self.corr_noise.transform(n)
        return n.numpy().astype(np.float32)[0]

    def _find_normalize(self):
        def walk(node):
            for t in getattr(node, "transforms", ()) or ():
                found = walk(t)
                if found is not None:
                    return found
            return node if type(node).__name__ == "Normalize" else None

        return walk(self._policy._input_transform)

    def normalize_prefix(self, prefix: np.ndarray) -> np.ndarray:
        prefix = np.asarray(prefix, dtype=np.float32)
        if prefix.ndim != 2 or prefix.shape[1] != 14:
            raise ValueError(f"prefix_actions must be (k,14), got {prefix.shape}")
        k = prefix.shape[0]
        if not 1 <= k < self.action_horizon:
            raise ValueError(f"prefix length {k} must be in [1, {self.action_horizon})")
        if not np.all(np.isfinite(prefix)):
            raise ValueError("prefix_actions contains non-finite values")
        norm = self._find_normalize()
        return normalize_actions(prefix, norm.norm_stats["actions"],
                                 norm.use_quantiles, self.action_dim)

    def infer(self, obs: dict, **kwargs: Any) -> dict:
        obs = dict(obs)
        seed = obs.pop("noise_seed", None)
        prefix = obs.pop("prefix_actions", None)
        if seed is not None and "noise" not in kwargs:
            kwargs["noise"] = self._noise(int(seed))
        if prefix is None:
            return self._policy.infer(obs, **kwargs)

        import torch

        a_prefix = torch.from_numpy(self.normalize_prefix(prefix)).to(
            self._policy._pytorch_device)
        sampler = functools.partial(
            sample_actions_inpaint, model=self._policy._model,
            a_prefix_norm=a_prefix, threshold=self.inpaint_time_threshold)
        orig = self._policy._sample_actions
        self._policy._sample_actions = sampler
        try:
            return self._policy.infer(obs, **kwargs)
        finally:
            self._policy._sample_actions = orig

    def __getattr__(self, name: str):
        return getattr(self._policy, name)


def normalize_actions(prefix: np.ndarray, stats, use_quantiles: bool,
                      action_dim: int = 32) -> np.ndarray:
    """Normalize (k,14) actions like openpi's transforms.Normalize, then
    zero-pad to the model action dim."""
    if use_quantiles:
        q01 = np.asarray(stats.q01)[..., :14]
        q99 = np.asarray(stats.q99)[..., :14]
        normed = (prefix - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0
    else:
        mean = np.asarray(stats.mean)[..., :14]
        std = np.asarray(stats.std)[..., :14]
        normed = (prefix - mean) / (std + 1e-6)
    out = np.zeros((prefix.shape[0], action_dim), np.float32)
    out[:, :14] = normed
    return out


def _euler_loop(denoise, x_t, noise, a_prefix, threshold, num_steps):
    """Euler integration with prefix inpainting for time > threshold.

    ``denoise(x_t, expanded_time) -> v_t`` is the only model-facing call.
    ``time`` semantics mirror PI0Pytorch.sample_actions.
    """
    import torch

    k = a_prefix.shape[1]
    dt = -1.0 / num_steps
    time = torch.tensor(1.0, dtype=torch.float32, device=x_t.device)
    while time >= -dt / 2:
        expanded_time = time.expand(x_t.shape[0])
        v_t = denoise(x_t, expanded_time)
        x_t = x_t + dt * v_t
        time += dt
        if time > threshold:
            x_t[:, :k, :] = time * noise[:, :k, :] + (1 - time) * a_prefix
    return x_t


def sample_actions_inpaint(device, observation, *, model, noise=None,
                           a_prefix_norm=None, threshold: float = 0.3,
                           num_steps: int = 10):
    """PI0Pytorch.sample_actions with prefix inpainting (copy of vendor body)."""
    import torch
    from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks

    bsize = observation.state.shape[0]
    if noise is None:
        actions_shape = (bsize, model.config.action_horizon,
                         model.config.action_dim)
        noise = model.sample_noise(actions_shape, device)

    images, img_masks, lang_tokens, lang_masks, state = \
        model._preprocess_observation(observation, train=False)
    prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
        images, img_masks, lang_tokens, lang_masks)
    prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
    prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
    prefix_att_2d_masks_4d = model._prepare_attention_masks_4d(
        prefix_att_2d_masks)
    model.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001
    _, past_key_values = model.paligemma_with_expert.forward(
        attention_mask=prefix_att_2d_masks_4d,
        position_ids=prefix_position_ids,
        past_key_values=None,
        inputs_embeds=[prefix_embs, None],
        use_cache=True,
    )

    if a_prefix_norm.dim() == 2:
        a_prefix_norm = a_prefix_norm[None]
    a_prefix_norm = a_prefix_norm.to(device=device, dtype=noise.dtype)

    def denoise(x_t, expanded_time):
        return model.denoise_step(state, prefix_pad_masks, past_key_values,
                                  x_t, expanded_time)

    return _euler_loop(denoise, noise, noise, a_prefix_norm, threshold,
                       num_steps)


def model_sha256(checkpoint_dir: str | Path) -> str:
    path = Path(checkpoint_dir) / "model.safetensors"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()
