"""CPU tests for prefix normalization + inpaint Euler schedule."""
import types
import unittest

import numpy as np

from roboeval_entry.training.serving import (  # noqa: E402 — pure-numpy module
    _euler_loop,
    normalize_actions,
)


def _stats(mean=None, std=None, q01=None, q99=None):
    return types.SimpleNamespace(mean=mean, std=std, q01=q01, q99=q99)


class NormalizeActionsTests(unittest.TestCase):
    def test_meanstd(self):
        stats = _stats(mean=np.zeros(14), std=np.ones(14))
        out = normalize_actions(np.ones((3, 14), np.float32) * 0.5, stats,
                                use_quantiles=False)
        self.assertEqual(out.shape, (3, 32))
        np.testing.assert_allclose(out[:, :14], 0.5, atol=1e-6)
        np.testing.assert_array_equal(out[:, 14:], 0)

    def test_quantile(self):
        stats = _stats(q01=np.zeros(14), q99=np.ones(14))
        out = normalize_actions(np.full((2, 14), 0.5, np.float32), stats,
                                use_quantiles=True)
        np.testing.assert_allclose(out[:, :14], 0.0, atol=1e-5)

    def test_euler_inpaint_schedule(self):
        # denoise returns zero velocity: inpainted rows get frozen at the last
        # overwrite (t=0.4); free rows keep the initial noise.
        noise = np.random.default_rng(0).standard_normal((1, 16, 32))
        a_prefix = np.full((1, 4, 32), 0.7)
        x0 = noise.copy()
        out = _euler_loop(lambda x, t: np.zeros_like(x), x0, noise,
                          a_prefix, threshold=0.3, num_steps=10)
        # last overwrite at time=0.4 -> x = 0.4*noise + 0.6*a_prefix
        np.testing.assert_allclose(
            out[0, :4], 0.4 * noise[0, :4] + 0.6 * 0.7, atol=1e-6)
        np.testing.assert_array_equal(out[0, 4:], noise[0, 4:])


if __name__ == "__main__":
    unittest.main()
