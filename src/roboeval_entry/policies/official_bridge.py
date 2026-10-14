"""Bridge between the official evaluator's Policy protocol and our adapters."""
from __future__ import annotations

from typing import Dict

import numpy as np

from roboeval_entry.control.guard import ActionGuard
from roboeval_entry.policies.base import (
    EpisodeContext,
    ObservationSchema,
    PolicyAdapter,
)


class OfficialPolicyBridge:
    """Implements reset(EpisodeSpec)/act(observation) for the official evaluator."""

    def __init__(self, adapter: PolicyAdapter, schema: ObservationSchema, guard: ActionGuard):
        self.adapter = adapter
        self.schema = schema
        self.guard = guard

    def reset(self, episode) -> None:
        self.adapter.reset(EpisodeContext.from_episode_spec(episode))

    def act(self, observation: Dict[str, np.ndarray]) -> np.ndarray:
        self.schema.validate(observation)
        action = self.adapter.act(observation)
        return self.guard(np.asarray(action, dtype=np.float32))
