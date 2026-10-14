from __future__ import annotations

import unittest

import numpy as np

from roboeval_entry.contracts.action_spec import (
    ActionContractError,
    ActionSpec,
    Bounds,
    GripperSpec,
)
from roboeval_entry.control.guard import ActionGuard


def make_spec() -> ActionSpec:
    return ActionSpec(
        version=1,
        source_commit="c8b79598c9fb9021652128020517d1847abe4c95",
        control_frequency_hz=20,
        layout=tuple(f"d{i}" for i in range(14)),
        rotation_representation="rotvec",
        rotation_composition="premultiply_pelvis_frame",
        position_frame="pelvis",
        delta_reference="actuator_ctrl_targets",
        env_bounds=Bounds(tuple([-1.0] * 14), tuple([1.0] * 14)),
        guard_bounds=Bounds(tuple([-0.5] * 12 + [0.0, 0.0]),
                            tuple([0.5] * 12 + [1.0, 1.0])),
        guard_bounds_provenance=None,
        gripper=GripperSpec(command_range=(0.0, 1.0), discrete=True,
                            open_command=0, closed_command=1, verified=True),
        joint_speed_limit_rad_per_step=0.10875,
        finite_required=True,
    )


class GuardTests(unittest.TestCase):
    def test_nan_rejected(self):
        g = ActionGuard(make_spec())
        a = np.zeros(14, dtype=np.float32)
        a[3] = np.nan
        with self.assertRaises(ActionContractError):
            g(a)

    def test_shape_rejected(self):
        g = ActionGuard(make_spec())
        with self.assertRaises(ActionContractError):
            g(np.zeros((14, 1), dtype=np.float32))
        with self.assertRaises(ActionContractError):
            g(np.zeros(13, dtype=np.float32))

    def test_clipping_with_mask(self):
        g = ActionGuard(make_spec())
        a = np.full(14, 10.0, dtype=np.float32)
        out = g(a)
        np.testing.assert_allclose(out[:12], 0.5)
        self.assertTrue(g.last_clip_mask[:12].all())
        self.assertEqual(int(g.clip_counts[0]), 1)

    def test_gripper_snapping(self):
        g = ActionGuard(make_spec())
        a = np.zeros(14, dtype=np.float32)
        a[12], a[13] = 0.49, 0.51
        out = g(a)
        self.assertEqual(out[12], 0.0)
        self.assertEqual(out[13], 1.0)

    def test_zero_residual_identity(self):
        g = ActionGuard(make_spec())
        b = np.linspace(-0.4, 0.4, 14, dtype=np.float32)
        b[12], b[13] = 0.0, 1.0
        np.testing.assert_array_equal(g(b), g(b + np.zeros(14)))

    def test_refuses_unset_guard_bounds(self):
        spec = make_spec()
        spec = ActionSpec(**{**spec.__dict__, "guard_bounds": None})
        with self.assertRaises(ActionContractError):
            ActionGuard(spec)


if __name__ == "__main__":
    unittest.main()
