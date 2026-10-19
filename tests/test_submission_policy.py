"""CPU test for the staged submission policy (fake websocket client)."""
import hashlib
import importlib.util
import json
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
POLICY_PY = ROOT / "submission" / "policy" / "policy.py"


def _load_policy_mod():
    spec = importlib.util.spec_from_file_location("submission_policy", POLICY_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _FakeClient:
    def __init__(self, sha: str):
        self.sha = sha
        self.requests = []

    def get_server_metadata(self):
        return {"model_sha256": self.sha}

    def infer(self, request):
        self.requests.append(dict(request))
        a = np.zeros((16, 14), np.float32)
        a[:, 12:14] = 1.0
        return {"actions": a}


@dataclass
class _FakeEpisode:
    family: str = "LiftPot"
    variation: str = "PositionAndOrientation"
    env_name: str = "LiftPotPositionAndOrientationEnv"
    instruction: str = "Lift the pot."
    episode_index: int = 0
    seed: int = 200000
    policy_seed: int = 0
    control_frequency: int = 20
    max_steps: int = 300
    action_space: object = None

    def __post_init__(self):
        class _Space:
            shape = (14,)
        self.action_space = _Space()


METADATA = {"interface": {"observation": {"cameras": [
    {"name": "head", "resolution": [256, 256]},
    {"name": "left_wrist", "resolution": [256, 256]},
    {"name": "right_wrist", "resolution": [256, 256]}]}}}


def _obs():
    obs = {
        "proprioception": np.arange(36, dtype=np.float32),
        "proprioception_grippers": np.zeros(2, np.float32),
        "proprioception_floating_base": np.zeros(0, np.float32),
        "proprioception_floating_base_actions": np.zeros(0, np.float32),
    }
    for cam in ("head", "left_wrist", "right_wrist"):
        obs[f"rgb_{cam}"] = np.zeros((3, 256, 256), np.uint8)
    return obs


class SubmissionPolicyTests(unittest.TestCase):
    def setUp(self):
        self.mod = _load_policy_mod()
        self.tmp = tempfile.TemporaryDirectory()
        ckpt = Path(self.tmp.name)
        weights = ckpt / "model.safetensors"
        weights.write_bytes(b"fake-weights")
        self.sha = hashlib.sha256(b"fake-weights").hexdigest()
        info = {"host": "127.0.0.1", "port": 8000, "execute_steps": 4,
                "model_sha256": self.sha, "config": "pi05_roboeval_pr8",
                "checkpoint_dir": str(ckpt)}
        (ckpt / "pi05_server.json").write_text(json.dumps(info))
        self.ckpt = ckpt

    def tearDown(self):
        self.tmp.cleanup()

    def _policy(self, sha=None):
        return _FakeClient(sha or self.sha), self.mod.make_policy(
            self.ckpt, METADATA, client=_FakeClient(sha or self.sha))

    def test_act_shape_bounds_and_request(self):
        policy = self.mod.make_policy(self.ckpt, METADATA,
                                      client=_FakeClient(self.sha))
        client = policy.adapter._client
        policy.reset(_FakeEpisode())
        a = policy.act(_obs())
        self.assertEqual(a.shape, (14,))
        self.assertEqual(a.dtype, np.float32)
        self.assertTrue(np.all(np.isfinite(a)))
        self.assertTrue((a[-2:] >= 0).all() and (a[-2:] <= 1).all())
        # one infer per execute_steps=4 acts
        for _ in range(3):
            policy.act(_obs())
        self.assertEqual(len(client.requests), 1)
        policy.act(_obs())
        self.assertEqual(len(client.requests), 2)
        req = client.requests[0]
        self.assertEqual(set(req), {
            "observation/head_image", "observation/left_wrist_image",
            "observation/right_wrist_image", "observation/state",
            "prompt", "noise_seed"})
        self.assertEqual(req["observation/head_image"].shape, (256, 256, 3))
        self.assertEqual(req["observation/head_image"].dtype, np.uint8)
        self.assertEqual(req["observation/state"].shape, (18,))
        self.assertEqual(req["prompt"], "Lift the pot.")

    def test_sha_mismatch_raises(self):
        with self.assertRaises(RuntimeError):
            self.mod.make_policy(self.ckpt, METADATA,
                                 client=_FakeClient("deadbeef"))

    def test_reset_clears_queue(self):
        client = _FakeClient(self.sha)
        policy = self.mod.make_policy(self.ckpt, METADATA, client=client)
        policy.reset(_FakeEpisode())
        policy.act(_obs())
        self.assertEqual(len(client.requests), 1)
        policy.reset(_FakeEpisode())
        policy.act(_obs())
        self.assertEqual(len(client.requests), 2)  # queue cleared -> new infer

    def test_state_mode_from_server_info(self):
        info = json.loads((self.ckpt / "pi05_server.json").read_text())
        info["state_mode"] = "qpos18_qvel18"
        (self.ckpt / "pi05_server.json").write_text(json.dumps(info))
        client = _FakeClient(self.sha)
        policy = self.mod.make_policy(self.ckpt, METADATA, client=client)
        policy.reset(_FakeEpisode())
        policy.act(_obs())
        self.assertEqual(
            client.requests[0]["observation/state"].shape, (36,))

    def test_stage_derives_state_mode(self):
        spec = importlib.util.spec_from_file_location(
            "stage_sub", ROOT / "scripts" / "stage_submission.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        for n, want in ((18, "qpos18"), (36, "qpos18_qvel18")):
            ns = self.ckpt / "assets" / "roboeval" / "rid" / "norm_stats.json"
            ns.parent.mkdir(parents=True, exist_ok=True)
            ns.write_text(json.dumps({"norm_stats": {
                "state": {"q01": [0.0] * n, "q99": [1.0] * n},
                "actions": {"q01": [0.0] * 14, "q99": [1.0] * 14}}}))
            self.assertEqual(mod._state_mode_from_ckpt(self.ckpt), want)
        import shutil
        shutil.rmtree(self.ckpt / "assets")
        with self.assertRaises(FileNotFoundError):
            mod._state_mode_from_ckpt(self.ckpt)


if __name__ == "__main__":
    unittest.main()
