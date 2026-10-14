from __future__ import annotations
import math
import json
from pathlib import Path
import unittest
from random import Random
from tools.seed_guard import validate_seeds, validate_lineage, POOLS, OFFICIAL
from tools.sgs_reference import Settings, probabilities, choose


class SeedGuardTests(unittest.TestCase):
    def test_config_pool_alignment(self):
        config = json.loads((Path(__file__).resolve().parents[1] / "configs/splits.json").read_text())
        self.assertEqual({k: tuple(config[k]) for k in POOLS}, POOLS)
        self.assertEqual(tuple(config["official_reserved"]), OFFICIAL)

    def test_valid_pool(self):
        self.assertEqual(validate_seeds([100000, 100001], "train"), [100000, 100001])

    def test_official_rejected(self):
        with self.assertRaises(ValueError):
            validate_seeds([1000], "train")

    def test_cross_split_rejected(self):
        with self.assertRaises(ValueError):
            validate_seeds([200000], "train")

    def test_duplicate_rejected(self):
        with self.assertRaises(ValueError):
            validate_seeds([100000, 100000], "train")

    def test_wrong_types_rejected(self):
        for value in (True, "100000", 100000.0):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_seeds([value], "train")

    def test_empty_rejected(self):
        with self.assertRaises(ValueError):
            validate_seeds([], "dev")

    def test_lineage_valid(self):
        validate_lineage([
            {"episode_id": "a", "parent_id": None, "split": "train"},
            {"episode_id": "b", "parent_id": "a", "split": "train"},
        ])

    def test_lineage_split_leak(self):
        with self.assertRaises(ValueError):
            validate_lineage([
                {"episode_id": "a", "parent_id": None, "split": "train"},
                {"episode_id": "b", "parent_id": "a", "split": "dev"},
            ])

    def test_unknown_parent(self):
        with self.assertRaises(ValueError):
            validate_lineage([{"episode_id": "b", "parent_id": "a", "split": "train"}])

    def test_lineage_cycle(self):
        with self.assertRaises(ValueError):
            validate_lineage([
                {"episode_id": "a", "parent_id": "b", "split": "train"},
                {"episode_id": "b", "parent_id": "a", "split": "train"},
            ])


class SamplerTests(unittest.TestCase):
    def setUp(self):
        self.histories = {"easy": [1] * 20, "frontier": [0, 1] * 10, "hard": [0] * 20}

    def test_normalized_and_full_support(self):
        p = probabilities(self.histories)
        self.assertAlmostEqual(sum(p.values()), 1.0)
        self.assertTrue(all(math.isfinite(v) and v > 0 for v in p.values()))

    def test_intermediate_preferred(self):
        p = probabilities(self.histories)
        self.assertGreater(p["frontier"], p["easy"])
        self.assertGreater(p["frontier"], p["hard"])

    def test_warmup_uniform(self):
        p = probabilities({"a": [], "b": [1, 1]})
        self.assertEqual(p, {"a": 0.5, "b": 0.5})

    def test_quality_priority(self):
        baseline = probabilities(self.histories)
        changed = probabilities(self.histories, quality_priority={"easy": 1.0})
        self.assertGreater(changed["easy"], baseline["easy"])

    def test_reproducible_sampling(self):
        p = probabilities(self.histories)
        a, b = Random(42), Random(42)
        self.assertEqual([choose(p, a) for _ in range(30)], [choose(p, b) for _ in range(30)])

    def test_rolling_window(self):
        p = probabilities({"a": [0] * 20 + [1] * 20, "b": [1] * 20})
        self.assertEqual(p, {"a": 0.5, "b": 0.5})

    def test_invalid_settings(self):
        for settings in (Settings(temperature=0), Settings(uniform_mass=0),
                         Settings(tau=2), Settings(epsilon=float("nan"))):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                probabilities(self.histories, settings)

    def test_invalid_outcome(self):
        with self.assertRaises(ValueError):
            probabilities({"a": [0, 2]})

    def test_unknown_quality_id(self):
        with self.assertRaises(ValueError):
            probabilities(self.histories, quality_priority={"not_in_pool": 1.0})

    def test_empty_pool(self):
        with self.assertRaises(ValueError):
            probabilities({})


if __name__ == "__main__":
    unittest.main()
