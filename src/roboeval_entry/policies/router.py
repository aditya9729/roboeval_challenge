"""Per-family policy routing: one fresh adapter instance per family."""
from __future__ import annotations

from typing import Callable, Mapping

import numpy as np

from roboeval_entry.policies.base import EpisodeContext, PolicyAdapter


class TaskRouter:
    """Factories run at most once per family; adapters persist across episodes
    of their family (models load once) and clear per-episode state in reset."""

    def __init__(self, factories: Mapping[str, Callable[[], PolicyAdapter]]):
        self._factories = dict(factories)
        self._adapters: dict[str, PolicyAdapter] = {}
        self._current: PolicyAdapter | None = None

    def reset(self, ctx: EpisodeContext) -> None:
        if ctx.family not in self._factories:
            raise KeyError(f"no policy factory for family {ctx.family!r}")
        adapter = self._adapters.get(ctx.family)
        if adapter is None:
            adapter = self._factories[ctx.family]()
            self._adapters[ctx.family] = adapter
        adapter.reset(ctx)
        self._current = adapter

    def act(self, obs: Mapping[str, np.ndarray]) -> np.ndarray:
        if self._current is None:
            raise RuntimeError("act() called before reset()")
        return self._current.act(obs)
