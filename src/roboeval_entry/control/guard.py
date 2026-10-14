"""Action guard: the last thing a policy action passes through before env.step."""
from __future__ import annotations

import numpy as np

from roboeval_entry.contracts.action_spec import (
    ActionContractError,
    ActionLayout,
    ActionSpec,
)


class ActionGuard:
    def __init__(self, spec: ActionSpec):
        if spec.guard_bounds is None:
            raise ActionContractError(
                "guard_bounds are unset in the action spec; run the data statistics "
                "step to fill them before constructing an ActionGuard"
            )
        self.spec = spec
        self.low, self.high = spec.guard_bounds.arrays()
        self.last_clip_mask = np.zeros(ActionLayout.DIM, dtype=bool)
        self.clip_counts = np.zeros(ActionLayout.DIM, dtype=np.int64)

    def __call__(self, action: np.ndarray) -> np.ndarray:
        a = np.asarray(action)
        if a.shape != (ActionLayout.DIM,):
            raise ActionContractError(
                f"action shape {a.shape} != ({ActionLayout.DIM},)")
        a = a.astype(np.float64)
        if self.spec.finite_required and not np.all(np.isfinite(a)):
            raise ActionContractError("action contains non-finite values")
        out = np.clip(a, self.low, self.high)
        if self.spec.gripper.discrete:
            out[ActionLayout.LEFT_GRIP] = float(round(out[ActionLayout.LEFT_GRIP]))
            out[ActionLayout.RIGHT_GRIP] = float(round(out[ActionLayout.RIGHT_GRIP]))
        self.last_clip_mask = out != a
        self.clip_counts += self.last_clip_mask
        return out.astype(np.float32)
