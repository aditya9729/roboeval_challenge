"""CPU tests for scripts/prune_checkpoints.py retention rules."""
import importlib.util
import os
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location(
        "prune_checkpoints", ROOT / "scripts" / "prune_checkpoints.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class PruneCheckpointsTests(unittest.TestCase):
    def setUp(self):
        self.mod = _load()
        self.tmp = tempfile.TemporaryDirectory()
        self.run = Path(self.tmp.name)
        for name in ["1000", "2000", "best"]:
            d = self.run / name
            (d / "assets").mkdir(parents=True)
            (d / "model.safetensors").write_bytes(b"W" * 10)
            (d / "optimizer.pt").write_bytes(b"O" * 20)
            (d / "metadata.pt").write_bytes(b"M")

    def tearDown(self):
        self.tmp.cleanup()

    def test_keep_newest_optimizer(self):
        self.mod.prune_run_dir(self.run, keep_optimizer=1, log=lambda m: None)
        self.assertTrue((self.run / "2000" / "optimizer.pt").exists())
        self.assertFalse((self.run / "1000" / "optimizer.pt").exists())
        # 'best' never counts toward K and its optimizer is pruned too
        self.assertFalse((self.run / "best" / "optimizer.pt").exists())
        # weights/metadata/assets untouched everywhere
        for name in ["1000", "2000", "best"]:
            d = self.run / name
            self.assertTrue((d / "model.safetensors").exists())
            self.assertTrue((d / "metadata.pt").exists())
            self.assertTrue((d / "assets").is_dir())

    def test_keep_two(self):
        self.mod.prune_run_dir(self.run, keep_optimizer=2, log=lambda m: None)
        self.assertTrue((self.run / "2000" / "optimizer.pt").exists())
        self.assertTrue((self.run / "1000" / "optimizer.pt").exists())

    def test_tmp_dirs(self):
        fresh = self.run / "tmp_3000"
        stale = self.run / "tmp_4000"
        for d in (fresh, stale):
            d.mkdir()
            (d / "model.safetensors").write_bytes(b"W")
        old = time.time() - 7200
        os.utime(stale, (old, old))
        self.mod.prune_run_dir(self.run, keep_optimizer=1, log=lambda m: None)
        self.assertTrue(fresh.exists())      # < 1h old, kept
        self.assertFalse(stale.exists())     # > 1h old, removed

    def test_dry_run_deletes_nothing(self):
        out = self.mod.prune_run_dir(self.run, keep_optimizer=1,
                                     dry_run=True, log=lambda m: None)
        self.assertTrue(out)  # reported deletions
        self.assertTrue((self.run / "1000" / "optimizer.pt").exists())

    def test_incomplete_dir_ignored(self):
        (self.run / "500").mkdir()  # no model.safetensors -> not a checkpoint
        self.mod.prune_run_dir(self.run, keep_optimizer=1, log=lambda m: None)
        self.assertTrue((self.run / "500").exists())


if __name__ == "__main__":
    unittest.main()
