from __future__ import annotations

import unittest
import uuid as uuid_mod

from roboeval_entry.data.splits import (
    assign_demo_split,
    validate_lineage,
    validate_seeds,
)


class DemoSplitTests(unittest.TestCase):
    def test_deterministic(self):
        u = "abc123"
        self.assertEqual(assign_demo_split(u), assign_demo_split(u))

    def test_fraction(self):
        uuids = [uuid_mod.uuid4().hex for _ in range(10000)]
        frac = sum(assign_demo_split(u) == "dev" for u in uuids) / len(uuids)
        self.assertAlmostEqual(frac, 0.1, delta=0.02)


class SeedGuardReexportTests(unittest.TestCase):
    def test_cross_pool_rejected(self):
        with self.assertRaises(ValueError):
            validate_seeds([300000], "dev")

    def test_official_rejected_everywhere(self):
        for split in ("train", "dev", "holdout"):
            with self.subTest(split=split), self.assertRaises(ValueError):
                validate_seeds([1000], split)

    def test_lineage(self):
        validate_lineage([
            {"episode_id": "a", "parent_id": None, "split": "train"},
            {"episode_id": "b", "parent_id": "a", "split": "train"},
        ])
        with self.assertRaises(ValueError):
            validate_lineage([
                {"episode_id": "a", "parent_id": None, "split": "train"},
                {"episode_id": "b", "parent_id": "a", "split": "dev"},
            ])


if __name__ == "__main__":
    unittest.main()
