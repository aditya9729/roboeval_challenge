"""CPU tests for GripperCorrectionRule."""
import unittest

import numpy as np

from roboeval_entry.control.gripper_rule import GripperCorrectionRule


def _act(l=1.0, r=1.0):
    a = np.zeros(14, np.float32)
    a[12], a[13] = l, r
    return a


class GripperRuleTests(unittest.TestCase):
    def test_trigger_override_cooldown_retrigger(self):
        r = GripperCorrectionRule(close_steps=10, closed_threshold=0.02,
                                  open_steps=6, cooldown_steps=20)
        obs = np.zeros(2, np.float32)
        fired = []
        for _ in range(60):
            _, trig = r.apply(_act(), obs)
            fired.append(trig)
        self.assertEqual(fired.index(True), 9)          # triggers at step 10
        fired_idx = [i for i, t in enumerate(fired) if t]
        self.assertEqual(fired_idx, [9, 34, 59])
        self.assertEqual(r.triggers.tolist(), [3, 3])
        self.assertEqual(r.override_steps.tolist(), [13, 13])

    def test_no_trigger_when_holding_or_open_cmd(self):
        r = GripperCorrectionRule()
        held = np.array([0.3, 0.3], np.float32)
        for _ in range(50):
            _, t1 = r.apply(_act(), held)          # object held
            _, t2 = r.apply(_act(0.0, 0.0), np.zeros(2))  # never commanded
        self.assertFalse(t1 or t2)
        self.assertEqual(r.triggers.tolist(), [0, 0])

    def test_arms_independent(self):
        r = GripperCorrectionRule(close_steps=10, open_steps=6,
                                  cooldown_steps=20)
        obs = np.array([0.0, 0.5], np.float32)   # right holds an object
        for _ in range(15):
            out, _ = r.apply(_act(), obs)
        self.assertEqual(r.triggers.tolist(), [1, 0])
        # during the open burst left is forced 0, right untouched
        self.assertEqual(out[12], 0.0)
        self.assertEqual(out[13], 1.0)

    def test_reset_clears(self):
        r = GripperCorrectionRule()
        for _ in range(15):
            r.apply(_act(), np.zeros(2))
        r.reset()
        self.assertEqual(r.triggers.tolist(), [0, 0])
        self.assertEqual(r._close_streak.tolist(), [0, 0])

    def test_input_not_mutated(self):
        r = GripperCorrectionRule()
        a = _act()
        before = a.copy()
        obs = np.zeros(2)
        for _ in range(12):
            r.apply(a, obs)
        np.testing.assert_array_equal(a, before)
        np.testing.assert_array_equal(obs, np.zeros(2))


if __name__ == "__main__":
    unittest.main()
