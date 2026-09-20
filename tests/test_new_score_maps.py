"""Property tests for the gradient_magnitude and local_entropy score maps.

Run: python3 -m unittest discover -s tests -v
"""

import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np

from read_changes import (
    SCORE_MAP_BUILDERS,
    SCORE_MAP_EXTRA_DEFAULTS,
    build_sampling_probabilities,
    build_score_map,
    resolve_score_map_params,
)

SHAPE = (32, 32)


def _noise(seed=0):
    return np.random.default_rng(seed).integers(0, 256, size=SHAPE, dtype=np.uint8)


class TestGradientMagnitude(unittest.TestCase):
    def test_registered(self):
        self.assertIn("gradient_magnitude", SCORE_MAP_BUILDERS)

    def test_constant_image_is_zero(self):
        e = build_score_map("gradient_magnitude", np.full(SHAPE, 90, dtype=np.uint8))
        self.assertEqual(e.shape, SHAPE)
        np.testing.assert_allclose(e, 0.0)

    def test_step_edge_localised(self):
        img = np.zeros(SHAPE, dtype=np.uint8)
        img[:, 16:] = 200
        e = build_score_map("gradient_magnitude", img)
        self.assertGreater(e[:, 15:17].min(), 0.0)
        np.testing.assert_allclose(e[:, :13], 0.0)
        np.testing.assert_allclose(e[:, 20:], 0.0)

    def test_probabilities_sum_to_one(self):
        p = build_sampling_probabilities(SHAPE, "gradient_magnitude", None, sampling_image=_noise())
        assert p is not None
        self.assertAlmostEqual(float(p.sum()), 1.0, places=10)
        self.assertTrue(np.all(p >= 0))


class TestLocalEntropy(unittest.TestCase):
    def test_registered_with_defaults(self):
        self.assertIn("local_entropy", SCORE_MAP_BUILDERS)
        self.assertEqual(SCORE_MAP_EXTRA_DEFAULTS["local_entropy"], {"window": 9, "bins": 32})
        params = resolve_score_map_params("local_entropy")
        self.assertEqual((params["window"], params["bins"]), (9, 32))

    def test_constant_image_is_zero(self):
        e = build_score_map("local_entropy", np.full(SHAPE, 90, dtype=np.uint8))
        np.testing.assert_allclose(e, 0.0, atol=1e-12)

    def test_bounded_by_log2_bins_and_positive_on_noise(self):
        e = build_score_map("local_entropy", _noise())
        self.assertLessEqual(float(e.max()), np.log2(32) + 1e-9)
        self.assertGreater(float(e.mean()), 1.0)

    def test_two_level_checkerboard_is_one_bit(self):
        img = np.indices(SHAPE).sum(axis=0) % 2 * 255
        e = build_score_map("local_entropy", img.astype(np.uint8))
        np.testing.assert_allclose(e[4:-4, 4:-4], 1.0, atol=0.05)  # 9x9 window: 41/40 split ≈ 1 bit

    def test_bins_validated(self):
        with self.assertRaises(ValueError):
            resolve_score_map_params("local_entropy", {"bins": 1})
        with self.assertRaises(ValueError):
            resolve_score_map_params("local_entropy", {"bins": 300})


if __name__ == "__main__":
    unittest.main()
