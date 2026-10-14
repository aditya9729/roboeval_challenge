"""Simulator-dependent contract tests; run only with ROBOEVAL_SIM_TESTS=1
inside the sim env (numpy 1.26 / mujoco)."""
from __future__ import annotations

import json
import os
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]

SIM = os.environ.get("ROBOEVAL_SIM_TESTS") == "1"


@unittest.skipUnless(SIM, "set ROBOEVAL_SIM_TESTS=1 in the sim env")
class SimContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from roboeval.challenge.submission import Interface, make_env

        cls.env = make_env("LiftPot", "PositionAndOrientation",
                           Interface(absolute=False, ee=True))

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def test_obs_schema_matches(self):
        from roboeval_entry.policies.base import ObservationSchema

        obs, _ = self.env.reset(seed=100000)
        ObservationSchema(cameras=()).validate(obs)

    def test_action_bounds_match_contract(self):
        from roboeval_entry.contracts.action_spec import load_action_spec

        spec = load_action_spec(REPO / "contracts" / "action_spec.json")
        low, high = spec.env_bounds.arrays()
        np.testing.assert_allclose(low, self.env.action_space.low, rtol=1e-6)
        np.testing.assert_allclose(high, self.env.action_space.high, rtol=1e-6)

    def test_bridge_runs_five_steps(self):
        from roboeval_entry.contracts.action_spec import load_action_spec
        from roboeval_entry.control.guard import ActionGuard
        from roboeval_entry.policies.base import ObservationSchema
        from roboeval_entry.policies.hold import HoldPolicy
        from roboeval_entry.policies.official_bridge import OfficialPolicyBridge

        spec = load_action_spec(REPO / "contracts" / "action_spec.json")
        if spec.guard_bounds is None:
            self.skipTest("guard_bounds not yet filled")
        bridge = OfficialPolicyBridge(
            HoldPolicy(), ObservationSchema(cameras=()), ActionGuard(spec))
        obs, _ = self.env.reset(seed=100000)
        episode = type("E", (), dict(
            family="LiftPot", variation="PositionAndOrientation",
            env_name="LiftPotPositionAndOrientation", instruction="x",
            episode_index=0, seed=100000, policy_seed=0,
            control_frequency=20, max_steps=200,
            action_space=self.env.action_space))()
        bridge.reset(episode)
        for _ in range(5):
            a = bridge.act(obs)
            self.assertEqual(a.shape, (14,))
            self.assertEqual(a.dtype, np.float32)
            self.assertTrue(np.all(np.isfinite(a)))
            obs, *_ = self.env.step(a)


if __name__ == "__main__":
    unittest.main()
