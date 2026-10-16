"""CPU tests for RoboEvalParquetDataset (requires pyarrow + pillow)."""
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    from PIL import Image

    HAVE_DEPS = True
except ImportError:
    HAVE_DEPS = False


def _png(color) -> bytes:
    img = Image.fromarray(np.full((8, 8, 3), color, np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@unittest.skipUnless(HAVE_DEPS, "pyarrow/pillow not installed")
class ParquetDatasetTests(unittest.TestCase):
    def setUp(self):
        from roboeval_entry.training.parquet_dataset import RoboEvalParquetDataset

        self.DS = RoboEvalParquetDataset
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        vdir = root / "LiftPotPositionAndOrientation"
        (vdir / "data" / "chunk-000").mkdir(parents=True)
        (vdir / "meta").mkdir(parents=True)
        self.uuids = ["uu-train-a", "uu-train-b", "uu-dev"]
        rows = []
        for ep, uuid in enumerate(self.uuids):
            for t in range(6):
                rows.append({
                    "episode_index": ep, "frame_index": t, "uuid": uuid,
                    "task_index": ep,
                    "state": list(np.linspace(ep * 10, ep * 10 + t, 36,
                                              dtype=np.float32)),
                    "actions": list(np.full(14, ep * 100 + t, np.float32)),
                    "cam_head": {"bytes": _png(10 + ep * 40 + t), "path": "h"},
                    "cam_left_wrist": {"bytes": _png(100 + t), "path": "l"},
                    "cam_right_wrist": {"bytes": _png(200 + t), "path": "r"},
                })
        schema = pa.schema([
            ("episode_index", pa.int64()), ("frame_index", pa.int64()),
            ("uuid", pa.string()), ("task_index", pa.int64()),
            ("state", pa.list_(pa.float32(), 36)),
            ("actions", pa.list_(pa.float32(), 14)),
            ("cam_head", pa.struct([("bytes", pa.binary()),
                                    ("path", pa.string())])),
            ("cam_left_wrist", pa.struct([("bytes", pa.binary()),
                                          ("path", pa.string())])),
            ("cam_right_wrist", pa.struct([("bytes", pa.binary()),
                                           ("path", pa.string())])),
        ])
        pq.write_table(pa.Table.from_pylist(rows, schema=schema),
                       vdir / "data" / "chunk-000" / "file-000.parquet")
        tasks = pa.table({"task_index": [0, 1, 2],
                          "__index_level_0__": ["do A", "do B", "do C"]})
        pq.write_table(tasks, vdir / "meta" / "tasks.parquet")
        self.registry = root / "registry.json"
        self._write_registry(["uu-train-a", "uu-train-b"], ["uu-dev"])
        self.root = root

    def _write_registry(self, train, dev):
        self.registry.write_text(json.dumps({
            "episodes": [{"episode_id": u, "split": "train"} for u in train]
            + [{"episode_id": u, "split": "dev"} for u in dev]}))

    def tearDown(self):
        self.tmp.cleanup()

    def test_counts_split_and_alignment(self):
        ds = self.DS(self.root, ["LiftPotPositionAndOrientation"], "train",
                     self.registry, action_horizon=4)
        # 2 train episodes x (6-1) samples; dev episode excluded
        self.assertEqual(len(ds), 10)
        self.assertEqual(ds.summary()["num_episodes"], 2)
        self.assertTrue(all(ds._episodes[i]["uuid"] != "uu-dev" for i in (0, 1)))

        s = ds[0]  # episode 0, t=1 -> obs frame 0
        np.testing.assert_array_equal(s["observation/state"],
                                      np.linspace(0, 0, 36, dtype=np.float32)[:18])
        self.assertEqual(s["observation/head_image"][0, 0, 0], 10)
        self.assertEqual(s["observation/left_wrist_image"][0, 0, 0], 100)
        self.assertEqual(s["observation/right_wrist_image"][0, 0, 0], 200)
        np.testing.assert_array_equal(s["actions"][:, 0],
                                      np.array([1, 2, 3, 4], np.float32))
        self.assertFalse(s["actions_is_pad"].any())
        self.assertEqual(s["prompt"], "do A")
        self.assertEqual(s["episode_uuid"], "uu-train-a")
        self.assertEqual(s["t"], 1)

    def test_padding(self):
        ds = self.DS(self.root, ["LiftPotPositionAndOrientation"], "train",
                     self.registry, action_horizon=3)
        s = ds[3]  # episode 0, t=4 -> actions[4:7], T=6 -> 2 real + 1 pad
        np.testing.assert_array_equal(s["actions"][:, 0],
                                      np.array([4, 5, 5], np.float32))
        np.testing.assert_array_equal(s["actions_is_pad"],
                                      np.array([False, False, True]))

    def test_state_mode_qvel(self):
        ds = self.DS(self.root, ["LiftPotPositionAndOrientation"], "train",
                     self.registry, action_horizon=4,
                     state_mode="qpos18_qvel18")
        s = ds[0]
        self.assertEqual(s["observation/state"].shape, (36,))
        np.testing.assert_array_equal(
            s["observation/state"],
            np.linspace(0, 0, 36, dtype=np.float32))
        with self.assertRaises(ValueError):
            self.DS(self.root, ["LiftPotPositionAndOrientation"], "train",
                    self.registry, action_horizon=4, state_mode="bogus")

    def test_episode_limit(self):
        ds = self.DS(self.root, ["LiftPotPositionAndOrientation"], "train",
                     self.registry, action_horizon=2, episode_limit=1)
        self.assertEqual(len(ds), 5)
        self.assertEqual(ds._episodes[0]["uuid"], "uu-train-a")

    def test_successful_only(self):
        sf = self.root / "succ.txt"
        sf.write_text("# header\nuu-train-a\n")
        ds = self.DS(self.root, ["LiftPotPositionAndOrientation"], "train",
                     self.registry, action_horizon=2, successful_only=True,
                     successful_file=sf)
        self.assertEqual(ds.summary()["num_episodes"], 1)
        self.assertEqual(ds._episodes[0]["uuid"], "uu-train-a")
        # default keeps everything
        ds2 = self.DS(self.root, ["LiftPotPositionAndOrientation"], "train",
                      self.registry, action_horizon=2)
        self.assertEqual(ds2.summary()["num_episodes"], 2)
        with self.assertRaises(FileNotFoundError):
            self.DS(self.root, ["LiftPotPositionAndOrientation"], "train",
                    self.registry, action_horizon=2, successful_only=True,
                    successful_file=self.root / "nope.txt")

    def test_missing_uuid_raises(self):
        self._write_registry(["uu-train-a"], ["uu-dev"])
        with self.assertRaises(KeyError):
            self.DS(self.root, ["LiftPotPositionAndOrientation"], "train",
                    self.registry, action_horizon=2)


if __name__ == "__main__":
    unittest.main()
