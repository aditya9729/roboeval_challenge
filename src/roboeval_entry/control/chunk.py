"""Temporal action chunk queue for chunked policies."""
from __future__ import annotations

import numpy as np

from roboeval_entry.contracts.action_spec import ActionLayout


class ActionChunkQueue:
    def __init__(self, execute_steps: int):
        if execute_steps < 1:
            raise ValueError("execute_steps must be >= 1")
        self.execute_steps = execute_steps
        self._queue: list[np.ndarray] = []

    def push(self, chunk: np.ndarray) -> None:
        a = np.asarray(chunk)
        if a.ndim != 2 or a.shape[1] != ActionLayout.DIM:
            raise ValueError(f"chunk shape {a.shape} must be (H, {ActionLayout.DIM})")
        if a.shape[0] < self.execute_steps:
            raise ValueError(
                f"chunk horizon {a.shape[0]} < execute_steps {self.execute_steps}")
        if not np.all(np.isfinite(a)):
            raise ValueError("chunk contains non-finite values")
        self._queue.extend(row.astype(np.float32, copy=True)
                          for row in a[: self.execute_steps])

    def pop(self) -> np.ndarray:
        if not self._queue:
            raise RuntimeError("pop() on empty ActionChunkQueue")
        return self._queue.pop(0)

    @property
    def remaining(self) -> int:
        return len(self._queue)

    def reset(self) -> None:
        self._queue.clear()
