"""Failed-grasp gripper correction rule.

Command semantics: 0 = open, 1 = closed. `proprioception_grippers` reads 1.0
fully open, 0.0 fully closed on empty air, intermediate while holding an
object. If a commanded close shows the gripper fully closed on nothing for the
whole window, the grasp failed: force open briefly, then cool down.
"""
from __future__ import annotations

import numpy as np

from roboeval_entry.contracts.action_spec import ActionLayout

_GRIPPERS = (ActionLayout.LEFT_GRIP, ActionLayout.RIGHT_GRIP)


class GripperCorrectionRule:
    def __init__(self, close_steps: int = 10, closed_threshold: float = 0.02,
                 open_steps: int = 6, cooldown_steps: int = 20):
        self.close_steps = close_steps
        self.closed_threshold = closed_threshold
        self.open_steps = open_steps
        self.cooldown_steps = cooldown_steps
        self.reset()

    def reset(self) -> None:
        self._close_streak = np.zeros(2, np.int64)
        self._empty_streak = np.zeros(2, np.int64)
        self._open_left = np.zeros(2, np.int64)
        self._cooldown_left = np.zeros(2, np.int64)
        self.triggers = np.zeros(2, np.int64)
        self.override_steps = np.zeros(2, np.int64)

    def apply(self, action: np.ndarray, gripper_obs: np.ndarray):
        out = np.asarray(action, dtype=np.float32).copy()
        obs = np.asarray(gripper_obs, dtype=np.float64)
        triggered = False
        for i, dim in enumerate(_GRIPPERS):
            if self._open_left[i] > 0:
                out[dim] = 0.0
                self._open_left[i] -= 1
                self.override_steps[i] += 1
                self._close_streak[i] = 0
                self._empty_streak[i] = 0
                continue
            if self._cooldown_left[i] > 0:
                self._cooldown_left[i] -= 1
            if out[dim] >= 0.5:
                self._close_streak[i] += 1
            else:
                self._close_streak[i] = 0
            if obs[i] <= self.closed_threshold:
                self._empty_streak[i] += 1
            else:
                self._empty_streak[i] = 0
            if (self._close_streak[i] >= self.close_steps
                    and self._empty_streak[i] >= self.close_steps
                    and self._cooldown_left[i] == 0):
                out[dim] = 0.0
                self._open_left[i] = self.open_steps - 1
                self._cooldown_left[i] = self.cooldown_steps
                self.triggers[i] += 1
                self.override_steps[i] += 1
                self._close_streak[i] = 0
                self._empty_streak[i] = 0
                triggered = True
        return out, triggered
