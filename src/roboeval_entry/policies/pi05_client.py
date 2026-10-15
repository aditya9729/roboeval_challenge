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


class RelaxedWebsocketClient:
    """Same wire protocol as openpi_client.WebsocketClientPolicy but with the
    client's keepalive pings disabled; under multiple eval workers the async
    server runs infer on its event loop and can exceed the 20 s default ping
    timeout. Reimplemented here so vendor code stays untouched."""

    def __init__(self, host: str, port: int):
        import websockets.sync.client
        from openpi_client import msgpack_numpy

        self._packer = msgpack_numpy.Packer()
        self._ws = websockets.sync.client.connect(
            f"ws://{host}:{port}", compression=None, max_size=None,
            ping_interval=None)
        self._server_metadata = msgpack_numpy.unpackb(self._ws.recv())

    def get_server_metadata(self) -> dict:
        return self._server_metadata

    def infer(self, obs: dict) -> dict:
        from openpi_client import msgpack_numpy

        self._ws.send(self._packer.pack(obs))
        response = self._ws.recv()
        if isinstance(response, str):
            raise RuntimeError(f"Error in inference server:\n{response}")
        return msgpack_numpy.unpackb(response)


class Pi05RemoteAdapter:
    def __init__(self, host: str = "127.0.0.1", port: int = 8000,
                 execute_steps: int = 4,
                 cameras: Sequence[str] = ("head", "left_wrist", "right_wrist"),
                 client=None):
        self.execute_steps = execute_steps
        self.cameras = tuple(cameras)
        if client is None:
            client = RelaxedWebsocketClient(host=host, port=port)
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
                f"observation/{cam}_image": np.moveaxis(obs[f"rgb_{cam}"], 0, -1)
                for cam in self.cameras
            }
            request.update({
                "observation/state": np.asarray(
                    obs["proprioception"][:18], dtype=np.float32),
                "prompt": self._prompt,
                "noise_seed": (self._noise_base + self._calls) % (2 ** 62),
            })
            self._calls += 1
            result = self._client.infer(request)
            chunk = np.asarray(result["actions"], dtype=np.float32)
            if chunk.shape != (16, 14):
                raise ValueError(f"expected actions (16,14), got {chunk.shape}")
            self._queue.push(chunk)
        return self._queue.pop()
