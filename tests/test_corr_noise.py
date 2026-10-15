"""CPU tests for correlated noise (numpy + torch; openpi only for patch test)."""
import os
import tempfile
import unittest

import numpy as np

from roboeval_entry.training.corr_noise import (
    CorrelatedNoise, estimate_action_correlation)


def _torch():
    try:
        import torch
        return torch
    except ImportError:
        raise unittest.SkipTest("torch not available")


class CorrelatedNoiseTests(unittest.TestCase):
    def test_identity_corr_is_identity_transform(self):
        torch = _torch()
        cn = CorrelatedNoise(np.eye(512), beta=0.5)
        w = torch.randn(3, 16, 32)
        np.testing.assert_allclose(cn.transform(w).numpy(), w.numpy(),
                                   atol=1e-5)

    def test_block_correlation_sampling(self):
        torch = _torch()
        corr = np.eye(512)
        i = np.arange(0, 4)  # dim0, steps 0-3
        corr[np.ix_(i, i)] = 0.9
        np.fill_diagonal(corr[np.ix_(i, i)], 1.0)
        cn = CorrelatedNoise(corr, beta=0.5)
        g = torch.Generator().manual_seed(0)
        w = torch.randn((20_000, 16, 32), generator=g)
        out = cn.transform(w).reshape(20_000, 512).double().numpy()
        emp = np.cov(out.T)
        np.testing.assert_allclose(emp[:4, :4], cn.M[:4, :4], atol=0.05)
        np.testing.assert_allclose(np.diag(emp)[4:], 1.0, atol=0.08)

    def test_save_load_roundtrip(self):
        cn = CorrelatedNoise(np.eye(512) * 0.9 + 0.1, beta=0.5,
                             norm_stats_sha256="abc")
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "cn.npz")
            cn.save(path)
            cn2 = CorrelatedNoise.load(path)
        np.testing.assert_array_equal(cn2.L, cn.L)
        self.assertEqual(cn2.beta, cn.beta)
        self.assertEqual(cn2.norm_stats_sha256, "abc")

    def test_patch_unpatch_restores(self):
        try:
            from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
        except ImportError:
            self.skipTest("openpi not importable")
        from roboeval_entry.training.corr_noise import (
            patch_pi0_sample_noise, unpatch_pi0_sample_noise)
        orig = PI0Pytorch.sample_noise
        cn = CorrelatedNoise(np.eye(512), beta=0.5)
        patch_pi0_sample_noise(cn)
        self.assertIsNot(PI0Pytorch.sample_noise, orig)
        patch_pi0_sample_noise(cn)  # idempotent
        unpatch_pi0_sample_noise()
        self.assertIs(PI0Pytorch.sample_noise, orig)

    def test_zero_variance_dims_identity_rows(self):
        # emulate estimate path: sigma with zero-variance padded dims
        torch = _torch()
        sigma = np.ones((4, 4))
        sigma[2:, :] = sigma[:, 2:] = 0.0
        np.fill_diagonal(sigma, [1.0, 1.0, 0.0, 0.0])
        sd = np.sqrt(np.clip(np.diag(sigma), 0, None))
        with np.errstate(invalid="ignore", divide="ignore"):
            corr = sigma / np.outer(sd, sd)
        zv = sd == 0
        corr[:, zv] = 0.0
        corr[zv, :] = 0.0
        corr[np.diag_indices_from(corr)] = 1.0
        np.testing.assert_array_equal(corr[2:], np.eye(4)[2:])
        np.testing.assert_array_equal(corr[:, 2:], np.eye(4)[:, 2:])


if __name__ == "__main__":
    unittest.main()
