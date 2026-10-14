from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from roboeval_entry.contracts.action_spec import (
    ActionContractError,
    ActionLayout,
    StateLayout,
    load_action_spec,
)

REPO = Path(__file__).resolve().parents[1]
SPEC_PATH = REPO / "contracts" / "action_spec.json"
EXPECTED_COMMIT = json.loads(
    (REPO / "configs" / "challenge.json").read_text())["expected_commit"]


class ActionSpecTests(unittest.TestCase):
    def test_loads(self):
        spec = load_action_spec(SPEC_PATH, expected_commit=EXPECTED_COMMIT)
        self.assertEqual(spec.dim, 14)
        self.assertEqual(spec.control_frequency_hz, 20)
        self.assertEqual(spec.rotation_representation, "rotvec")

    def test_wrong_commit_rejected(self):
        with self.assertRaises(ActionContractError):
            load_action_spec(SPEC_PATH, expected_commit="0" * 40)

    def test_layout_slices_cover_once(self):
        flat = [i for s in (ActionLayout.LEFT_POS, ActionLayout.LEFT_ROT,
                            ActionLayout.RIGHT_POS, ActionLayout.RIGHT_ROT)
                for i in range(*s.indices(14))]
        flat += [ActionLayout.LEFT_GRIP, ActionLayout.RIGHT_GRIP]
        self.assertEqual(sorted(flat), list(range(ActionLayout.DIM)))

    def test_state_layout_disjoint(self):
        flat = [i for s in (StateLayout.LEFT_ARM_QPOS, StateLayout.LEFT_FINGER_QPOS,
                            StateLayout.RIGHT_ARM_QPOS, StateLayout.RIGHT_FINGER_QPOS)
                for i in range(*s.indices(36))]
        self.assertEqual(len(flat), len(set(flat)))
        self.assertEqual(max(flat), StateLayout.QVEL_OFFSET - 1)

    def test_env_bounds_shape(self):
        spec = load_action_spec(SPEC_PATH)
        self.assertIsNotNone(spec.env_bounds)
        low, high = spec.env_bounds.arrays()
        self.assertEqual(low.shape, (14,))
        self.assertTrue(np.all(low < high))

    def test_dataset_alignment_required(self):
        data = json.loads(SPEC_PATH.read_text())
        del data["dataset_alignment"]
        with tempfile.NamedTemporaryFile("w", suffix=".json") as f:
            json.dump(data, f)
            f.flush()
            with self.assertRaises(ActionContractError):
                load_action_spec(f.name)

    def test_dataset_alignment_exact_strings(self):
        spec = load_action_spec(SPEC_PATH)
        data = json.loads(SPEC_PATH.read_text())
        data["dataset_alignment"]["frame_t_state_is"] = "observation_before_actions_t"
        with tempfile.NamedTemporaryFile("w", suffix=".json") as f:
            json.dump(data, f)
            f.flush()
            with self.assertRaises(ActionContractError):
                load_action_spec(f.name)
        self.assertIsNotNone(spec)

    def test_null_bounds_allowed_but_shape_checked(self):
        data = json.loads(SPEC_PATH.read_text())
        data["guard_bounds"] = {"low": [0.0] * 5, "high": [1.0] * 5}
        with tempfile.NamedTemporaryFile("w", suffix=".json") as f:
            json.dump(data, f)
            f.flush()
            with self.assertRaises(ActionContractError):
                load_action_spec(f.name)


if __name__ == "__main__":
    unittest.main()
