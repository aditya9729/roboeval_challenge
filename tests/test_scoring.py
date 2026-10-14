from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np

from roboeval_entry.evaluation.scoring import (
    bootstrap_suite,
    import_official_scoring,
    paired_bootstrap_delta,
    rds_from_episode_metrics,
    suite_rds,
    vendor_root,
)
from roboeval_entry.evaluation.results_io import load_episode_metrics

VENDOR = vendor_root()
FIXTURE_RESULTS = Path("/workspace/roboeval/runs/official_install_check/results.json")
ALL_ZERO = {k: 0.0 for k in (
    "success", "tp", "bavd", "bgvd", "cpl", "ct", "ecc",
    "jpl", "mcj", "mjj", "opl", "scc", "sc", "tl")}


@unittest.skipUnless((VENDOR / "roboeval/challenge/scoring.py").exists(),
                     "vendor tree not available")
class ScoringTests(unittest.TestCase):
    def test_hold_still_scores_zero(self):
        out = rds_from_episode_metrics("LiftPot", [dict(ALL_ZERO)])
        self.assertEqual(out["score"], 0.0)

    def test_perfect_episode_scores_one(self):
        scoring = import_official_scoring()
        ref = scoring.human_reference()["LiftPot"]
        ep = dict(ref)
        ep["success"] = 1.0
        ep["tp"] = 1.0
        ep["ct"] = ref["ct"]
        ep["tl"] = ref["tl"]
        out = rds_from_episode_metrics("LiftPot", [ep])
        self.assertAlmostEqual(out["score"], 1.0, places=9)

    @unittest.skipUnless(FIXTURE_RESULTS.exists(), "install-check fixture missing")
    def test_matches_evaluator_results_json(self):
        results = json.loads(FIXTURE_RESULTS.read_text())
        per_family = load_episode_metrics(FIXTURE_RESULTS.parent / "logs")
        for family, task in results["score"]["tasks"].items():
            ours = rds_from_episode_metrics(family, per_family[family])
            self.assertAlmostEqual(ours["score"], task["score"], places=9)

    def test_bootstrap_ci(self):
        rng = np.random.default_rng(0)
        eps = []
        for i in range(10):
            e = dict(ALL_ZERO)
            e["success"] = float(rng.integers(0, 2))
            e["tp"] = e["success"]
            eps.append(e)
        out = bootstrap_suite({"LiftPot": eps}, n_boot=200, seed=0)
        self.assertTrue(np.isfinite(out["ci95"]).all())
        self.assertLessEqual(out["ci95"][0], out["mean"])
        self.assertLessEqual(out["mean"], out["ci95"][1])
        self.assertIn("LiftPot", out["per_task_ci"])

    def test_paired_delta_identical_is_zero(self):
        eps = [dict(ALL_ZERO, success=float(i % 2)) for i in range(6)]
        out = paired_bootstrap_delta({"LiftPot": eps}, {"LiftPot": eps},
                                     n_boot=50, seed=0)
        self.assertEqual(out["delta"], 0.0)
        self.assertTrue((np.asarray(out["ci95"]) == 0.0).all())


if __name__ == "__main__":
    unittest.main()
