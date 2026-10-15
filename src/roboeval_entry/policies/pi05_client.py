"""Remote pi0.5 PolicyAdapter speaking to a WebsocketPolicyServer (sim env).

Determinism contract: the official evaluator seeds Python's ``random`` module
right before ``policy.reset`` (see evaluate._seed_policy_rngs), so drawing
``noise_base`` from ``random`` inside ``reset`` is the sanctioned seed path —
the adapter never sees an env seed itself.
"""
from __future__ import annotations

import random
from typing import Mapping, Sequence

import numpy as np

from roboeval_entry.control.chunk import ActionChunkQueue
from roboeval_entry.policies.base import EpisodeContext


class Pi05RemoteAdapter:
    def __init__(self, host: str = "127.0.0.1", port: int = 8000,
                 execute_steps: int = 4,
                 cameras: Sequence[str] = ("head", "left_wrist", "right_wrist"),
                 client=None):
        self.execute_steps = execute_steps
        self.cameras = tuple(cameras)
        if client is None:
            from openpi_client import websocket_client_policy

            client = websocket_client_policy.WebsocketClientPolicy(
                host=host, port=port)
        self._client = client
        self._queue = ActionChunkQueue(execute_steps)
        self._prompt = ""
        self._noise_base = 0
        self._calls = 0

    def reset(self, ctx: EpisodeContext) -> None:
        self._prompt = ctx.instruction
        self._queue.reset()
        self._noise_base = random.getrandbits(62)
        self._calls = 0

    def act(self, obs: Mapping[str, np.ndarray]) -> np.ndarray:
        if self._queue.remaining == 0:
            request = {
                "observation/head_image": np.moveaxis(
                    obs["rgb_head"], 0, -1),
                "observation/left_wrist_image": np.moveaxis(
                    obs["rgb_left_wrist"], 0, -1),
                "observation/right_wrist_image": np.moveaxis(
                    obs["rgb_right_wrist"], 0, -1),
                "observation/state": np.asarray(
                    obs["proprioception"][:18], dtype=np.float32),
                "prompt": self._prompt,
                "noise_seed": (self._noise_base + self._calls) % (2 ** 62),
            }
            self._calls += 1
            result = self._client.infer(request)
            chunk = np.asarray(result["actions"], dtype=np.float32)
            if chunk.shape != (16, 14):
                raise ValueError(f"expected actions (16,14), got {chunk.shape}")
            self._queue.push(chunk)
        return self._queue.pop()
