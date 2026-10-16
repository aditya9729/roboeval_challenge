"""CPU tests: PlateauAnnealer, val RNG isolation, HPO helpers."""
import importlib.util
import json
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from roboeval_entry.training.train_loop import (
    PlateauAnnealer, ValidationConfig, openloop_objective, validation_noise)


def _load_script(name):
    path = Path(__file__).resolve().parents[1] / "scripts" / name
    spec = importlib.util.spec_from_file_location(name[:-3], path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class PlateauAnnealerTests(unittest.TestCase):
    def _a(self, **kw):
        cfg = dict(patience=3, anneal_factor=0.5, min_lr_scale=0.1,
                   max_anneals=2, rel_tol=0.01)
        cfg.update(kw)
        return PlateauAnnealer(**cfg)

    def test_improvement_resets_patience(self):
        a = self._a()
        self.assertEqual(a.update(1.0), "improved")
        self.assertEqual(a.update(1.0), "no_improve")
        self.assertEqual(a.update(1.0), "no_improve")
        self.assertEqual(a.update(0.9), "improved")
        self.assertEqual(a.bad, 0)

    def test_rel_tol_boundary(self):
        a = self._a()
        a.update(1.0)
        # improvement must EXCEED rel_tol: 0.99 = exactly 1% -> not improved
        self.assertEqual(a.update(0.99), "no_improve")
        self.assertEqual(a.update(0.98), "improved")
        # 0.985 vs best 0.99: not >1% improvement -> no_improve
        self.assertEqual(a.update(0.985), "no_improve")

    def test_anneal_floor_and_stop(self):
        a = self._a()
        a.update(1.0)
        for _ in range(3):
            a.update(1.0)
        self.assertEqual(a.anneals, 1)
        self.assertEqual(a.lr_scale, 0.5)
        for _ in range(3):
            a.update(1.0)
        self.assertEqual(a.anneals, 2)
        self.assertEqual(a.lr_scale, 0.25)
        # anneals exhausted -> stop
        for _ in range(3):
            r = a.update(1.0)
        self.assertEqual(r, "stop")

    def test_resume_state(self):
        a = self._a()
        a.update(0.8); a.update(1.0); a.update(1.0); a.update(1.0)  # anneal
        d = a.to_dict()
        b = PlateauAnnealer.from_dict(d, patience=3, anneal_factor=0.5,
                                    min_lr_scale=0.1, max_anneals=2,
                                    rel_tol=0.01)
        self.assertEqual((b.best, b.bad, b.anneals, b.lr_scale),
                         (a.best, a.bad, a.anneals, a.lr_scale))
        self.assertEqual(b.update(0.7), "improved")

    def test_openloop_objective_beats_baseline(self):
        rng = np.random.default_rng(0)
        targets = rng.normal(size=(8, 16, 14)) * 0.01
        mean = np.zeros(14)
        preds = targets.copy()
        ratio, grip = openloop_objective(preds, targets, mean)
        self.assertAlmostEqual(ratio, 0.0, places=6)
        self.assertEqual(grip, 1.0)
        ratio2, _ = openloop_objective(mean[None, None].repeat(8, 0).reshape(
            8, 1, 14).repeat(16, 1), targets, mean)
        self.assertAlmostEqual(ratio2, 1.0, places=5)


class ValRngIsolationTests(unittest.TestCase):
    def setUp(self):
        try:
            import torch  # noqa
        except ImportError:
            self.skipTest("torch not available")

    def test_validation_does_not_consume_train_rng(self):
        import torch

        def run(with_val: bool):
            torch.manual_seed(0)
            draws = []
            for step in range(20):
                if with_val and step % 5 == 0:
                    validation_noise(0, 4, 16, 32)  # val-side draws
                draws.append(torch.randn(4))       # "train step" draws
            return torch.stack(draws)

        torch.testing.assert_close(run(True), run(False))

    def test_validation_noise_deterministic(self):
        import torch
        a, _ = validation_noise(0, 4, 16, 32)
        b, _ = validation_noise(0, 4, 16, 32)
        c, _ = validation_noise(1, 4, 16, 32)
        self.assertTrue(torch.equal(a, b))
        self.assertFalse(torch.equal(a, c))
        self.assertEqual(a.shape, (4, 1, 16, 32))


class HpoHelperTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hpo = _load_script("hpo_tpe.py")

    def test_cleanup_keeps_best_only(self):
        with tempfile.TemporaryDirectory() as td:
            ck = Path(td)
            for name in ["250", "500", "best", "tmp_9"]:
                (ck / name).mkdir()
            (ck / "val_history.jsonl").write_text("{}")
            self.hpo.cleanup_ckpt_dir(ck)
            left = sorted(p.name for p in ck.iterdir())
            self.assertEqual(left, ["best", "val_history.jsonl"])

    def test_dry_run_deterministic_and_in_range(self):
        try:
            import optuna  # noqa
            from optuna.samplers import TPESampler
        except ImportError:
            self.skipTest("optuna not available")
        study = optuna.create_study(
            direction="minimize", sampler=TPESampler(seed=0, multivariate=True))
        p1 = [self.hpo.suggest(study.ask()) for _ in range(5)]
        study2 = optuna.create_study(
            direction="minimize", sampler=TPESampler(seed=0, multivariate=True))
        p2 = [self.hpo.suggest(study2.ask()) for _ in range(5)]
        self.assertEqual(p1, p2)
        for p in p1:
            self.assertTrue(1e-5 <= p["peak_lr"] <= 1e-4)
            self.assertIn(p["batch_size"], [8, 16, 32])
            self.assertIn(p["ema_decay"], [None, 0.99, 0.999])
            self.assertTrue(200 <= p["warmup_steps"] <= 2000)
            self.assertTrue(1e-6 <= p["weight_decay"] <= 1e-1)


if __name__ == "__main__":
    unittest.main()
