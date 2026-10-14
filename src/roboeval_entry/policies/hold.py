"""Plumbing-only hold-still policy. NOT a learned baseline.

Emits zero ee-delta and echoes the observed gripper state so the grippers
stay where they are. Exists to exercise the evaluate pipeline end to end.
"""
from __future__ import annotations

from typing import Mapping

import numpy as np

from roboeval_entry.contracts.action_spec import ActionLayout
from roboeval_entry.policies.base import EpisodeContext


class HoldPolicy:
    def __init__(self) -> None:
        self._ctx: EpisodeContext | None = None

    def reset(self, ctx: EpisodeContext) -> None:
        self._ctx = ctx

    def act(self, obs: Mapping[str, np.ndarray]) -> np.ndarray:
        action = np.zeros(ActionLayout.DIM, dtype=np.float32)
        action[ActionLayout.LEFT_GRIP] = obs["proprioception_grippers"][0]
        action[ActionLayout.RIGHT_GRIP] = obs["proprioception_grippers"][1]
        return action
