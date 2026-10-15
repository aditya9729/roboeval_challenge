"""CPU test for Pi05RemoteAdapter with a fake websocket client."""
import random
import unittest

import numpy as np

from roboeval_entry.policies.base import EpisodeContext
from roboeval_entry.policies.pi05_client import Pi05RemoteAdapter


class _FakeClient:
    def __init__(self):
        self.requests = []

    def infer(self, request):
        self.requests.append(dict(request))
        return {"actions": np.full((16, 14), len(self.requests), np.float32)}


def _obs():
    return {
        "rgb_head": np.moveaxis(np.full((256, 256, 3), 1, np.uint8), -1, 0),
        "rgb_left_wrist": np.moveaxis(np.full((256, 256, 3), 2, np.uint8), -1, 0),
        "rgb_right_wrist": np.moveaxis(np.full((256, 256, 3), 3, np.uint8), -1, 0),
        "proprioception": np.arange(36, dtype=np.float32),
        "proprioception_grippers": np.zeros(2, np.float32),
    }


CTX = EpisodeContext(family="LiftPot", variation="PositionAndOrientation",
                     instruction="do the thing", control_frequency=20,
                     max_steps=300, action_dim=14)


class Pi05RemoteAdapterTests(unittest.TestCase):
    def test_request_and_chunking(self):
        client = _FakeClient()
        adapter = Pi05RemoteAdapter(execute_steps=4, client=client)
        adapter.reset(CTX)
        for _ in range(4):
            adapter.act(_obs())
        self.assertEqual(len(client.requests), 1)
        req = client.requests[0]
        # CHW -> HWC and channel values preserved
        self.assertEqual(req["observation/head_image"].shape, (256, 256, 3))
        self.assertEqual(req["observation/left_wrist_image"][0, 0, 0], 2)
        self.assertEqual(req["observation/right_wrist_image"][0, 0, 0], 3)
        np.testing.assert_array_equal(req["observation/state"],
                                      np.arange(18, dtype=np.float32))
        self.assertEqual(req["prompt"], "do the thing")
        self.assertIn("noise_seed", req)
        # 5th act triggers second infer; noise_seed increments
        adapter.act(_obs())
        self.assertEqual(len(client.requests), 2)
        self.assertEqual(
            (client.requests[1]["noise_seed"] - client.requests[0]["noise_seed"])
            % (2 ** 62), 1)

    def test_reset_redraws_and_clears(self):
        client = _FakeClient()
        adapter = Pi05RemoteAdapter(execute_steps=4, client=client)
        random.seed(1234)
        adapter.reset(CTX)
        base1 = adapter._noise_base
        adapter.act(_obs())
        random.seed(1234)
        adapter.reset(CTX)
        self.assertEqual(adapter._noise_base, base1)  # deterministic re-draw
        self.assertEqual(adapter._queue.remaining, 0)
        self.assertEqual(adapter._calls, 0)
        random.seed(9999)
        adapter.reset(CTX)
        self.assertNotEqual(adapter._noise_base, base1)


class GripperRuleWiringTests(unittest.TestCase):
    def test_trigger_forces_replan(self):
        from roboeval_entry.control.gripper_rule import GripperCorrectionRule

        client = _FakeClient()
        rule = GripperCorrectionRule(close_steps=10, closed_threshold=0.02,
                                     open_steps=6, cooldown_steps=20)
        adapter = Pi05RemoteAdapter(execute_steps=4, client=client,
                                    gripper_rule=rule)
        adapter.reset(CTX)
        obs = _obs()  # grippers read 0.0 = closed on empty air
        # fake chunks command gripper=1.0 -> failed grasp detected at step 10
        for _ in range(10):
            adapter.act(obs)
        self.assertEqual(rule.triggers.tolist(), [1, 1])
        adapter.act(obs)   # queue cleared -> extra infer
        self.assertEqual(len(client.requests), 4)  # steps 1,5,9 -> 3 infers + this


if __name__ == "__main__":
    unittest.main()
