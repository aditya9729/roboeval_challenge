from __future__ import annotations

import dataclasses
import unittest
from types import SimpleNamespace

import numpy as np

from roboeval_entry.contracts.action_spec import ActionLayout
from roboeval_entry.policies.base import (
    EpisodeContext,
    ObservationContractError,
    ObservationSchema,
)
from roboeval_entry.policies.hold import HoldPolicy
from roboeval_entry.policies.router import TaskRouter


def fake_episode(family="LiftPot"):
    return SimpleNamespace(
        family=family, variation="PositionAndOrientation", env_name="X",
        instruction="do the thing", episode_index=3, seed=1000,
        policy_seed=0, control_frequency=20, max_steps=200,
        action_space=SimpleNamespace(shape=(14,)),
    )


def good_obs(cameras=("head",)):
    obs = {
        "proprioception": np.zeros(36, dtype=np.float32),
        "proprioception_grippers": np.ones(2, dtype=np.float32),
        "proprioception_floating_base": np.zeros(0, dtype=np.float32),
        "proprioception_floating_base_actions": np.zeros(0, dtype=np.float32),
    }
    for c in cameras:
        obs[f"rgb_{c}"] = np.zeros((3, 256, 256), dtype=np.uint8)
    return obs


class EpisodeContextTests(unittest.TestCase):
    def test_no_seed_fields(self):
        names = {f.name for f in dataclasses.fields(EpisodeContext)}
        for forbidden in ("seed", "episode_index", "policy_seed", "env_name",
                          "action_space"):
            self.assertNotIn(forbidden, names)

    def test_from_episode_spec(self):
        ctx = EpisodeContext.from_episode_spec(fake_episode())
        self.assertEqual(ctx.family, "LiftPot")
        self.assertEqual(ctx.max_steps, 200)
        self.assertEqual(ctx.action_dim, 14)


class ObservationSchemaTests(unittest.TestCase):
    def test_accepts_valid(self):
        ObservationSchema(cameras=("head",)).validate(good_obs())

    def test_rejects_extra_key(self):
        obs = good_obs()
        obs["object_pose"] = np.zeros(7, dtype=np.float32)
        with self.assertRaises(ObservationContractError):
            ObservationSchema(cameras=("head",)).validate(obs)

    def test_rejects_missing(self):
        obs = good_obs()
        del obs["rgb_head"]
        with self.assertRaises(ObservationContractError):
            ObservationSchema(cameras=("head",)).validate(obs)

    def test_rejects_wrong_dtype(self):
        obs = good_obs()
        obs["rgb_head"] = obs["rgb_head"].astype(np.float32)
        with self.assertRaises(ObservationContractError):
            ObservationSchema(cameras=("head",)).validate(obs)

    def test_state_only_schema(self):
        ObservationSchema(cameras=()).validate(good_obs(cameras=()))


class HoldPolicyTests(unittest.TestCase):
    def test_zeros_plus_gripper_copy(self):
        p = HoldPolicy()
        p.reset(EpisodeContext.from_episode_spec(fake_episode()))
        obs = good_obs()
        obs["proprioception_grippers"] = np.array([0.0, 1.0], dtype=np.float32)
        a = p.act(obs)
        self.assertEqual(a.shape, (14,))
        np.testing.assert_array_equal(a[:12], np.zeros(12))
        self.assertEqual(a[ActionLayout.LEFT_GRIP], 0.0)
        self.assertEqual(a[ActionLayout.RIGHT_GRIP], 1.0)


class StubAdapter:
    constructed = 0

    def __init__(self):
        type(self).constructed += 1
        self.resets = 0
        self.history: list = []

    def reset(self, ctx):
        self.resets += 1
        self.history.clear()

    def act(self, obs):
        self.history.append(obs)
        return np.zeros(14, dtype=np.float32)


class TaskRouterTests(unittest.TestCase):
    def setUp(self):
        StubAdapter.constructed = 0

    def test_same_family_reuses_adapter_and_clears_state(self):
        router = TaskRouter({"LiftPot": StubAdapter})
        router.reset(EpisodeContext.from_episode_spec(fake_episode("LiftPot")))
        router.act(good_obs())
        self.assertEqual(len(router._adapters["LiftPot"].history), 1)
        router.reset(EpisodeContext.from_episode_spec(fake_episode("LiftPot")))
        self.assertEqual(StubAdapter.constructed, 1)
        adapter = router._adapters["LiftPot"]
        self.assertEqual(adapter.resets, 2)
        self.assertEqual(adapter.history, [])

    def test_different_families_get_own_instances(self):
        router = TaskRouter({"LiftPot": StubAdapter, "LiftTray": StubAdapter})
        router.reset(EpisodeContext.from_episode_spec(fake_episode("LiftPot")))
        router.reset(EpisodeContext.from_episode_spec(fake_episode("LiftTray")))
        self.assertEqual(StubAdapter.constructed, 2)
        self.assertIsNot(router._adapters["LiftPot"], router._adapters["LiftTray"])

    def test_routers_never_share_instances(self):
        r1 = TaskRouter({"LiftPot": StubAdapter})
        r2 = TaskRouter({"LiftPot": StubAdapter})
        r1.reset(EpisodeContext.from_episode_spec(fake_episode("LiftPot")))
        r2.reset(EpisodeContext.from_episode_spec(fake_episode("LiftPot")))
        self.assertIsNot(r1._adapters["LiftPot"], r2._adapters["LiftPot"])

    def test_act_before_reset_raises(self):
        router = TaskRouter({"LiftPot": StubAdapter})
        with self.assertRaises(RuntimeError):
            router.act(good_obs())

    def test_unknown_family(self):
        router = TaskRouter({})
        with self.assertRaises(KeyError):
            router.reset(EpisodeContext.from_episode_spec(fake_episode("Nope")))


if __name__ == "__main__":
    unittest.main()
