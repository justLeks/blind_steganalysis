"""Property tests for probing distributions, sampling, seeding, and the conseal oracle.

Run: python3 -m unittest discover -s tests -v
The oracle test exercises conseal (numba JIT) and may take ~10-20 s on first run.
"""

import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np

from read_changes import (
    SUPPORTED_PROBE_DISTRIBUTIONS,
    build_sampling_probabilities,
    reconstruct_change_mask,
    sample_pixel_indices,
)
from embedding import derive_image_seed, simulate_embedding
from selection_channel import change_probability_map


SHAPE = (24, 24)


def _synthetic_image(seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=SHAPE, dtype=np.uint8)


class TestDistributions(unittest.TestCase):
    def test_probabilities_sum_to_one(self):
        img = _synthetic_image()
        for dist in sorted(SUPPORTED_PROBE_DISTRIBUTIONS):
            probs = build_sampling_probabilities(SHAPE, dist, {"gamma": 1.0, "floor": 1e-6}, sampling_image=img)
            if dist == "uniform":
                self.assertIsNone(probs)  # uniform is represented as None (rng.choice with p=None)
                continue
            assert probs is not None  # narrowed: only uniform returns None
            self.assertEqual(probs.shape, (SHAPE[0] * SHAPE[1],))
            self.assertAlmostEqual(float(probs.sum()), 1.0, places=10)
            self.assertTrue(np.all(probs >= 0))

    def test_texture_energy_prefers_edges(self):
        # Left half flat, right half noisy -> right half should carry most of the probability mass.
        img = np.zeros(SHAPE, dtype=np.uint8)
        img[:, SHAPE[1] // 2:] = _synthetic_image(1)[:, SHAPE[1] // 2:]
        probs = build_sampling_probabilities(SHAPE, "texture_energy", {"gamma": 1.0, "floor": 1e-6}, sampling_image=img)
        assert probs is not None
        grid = probs.reshape(SHAPE)
        self.assertGreater(grid[:, SHAPE[1] // 2:].sum(), grid[:, : SHAPE[1] // 2].sum())


class TestSampling(unittest.TestCase):
    def test_sampling_without_replacement(self):
        for dist, img in [("uniform", None), ("texture_energy", _synthetic_image())]:
            idx = sample_pixel_indices(SHAPE, sample_count=200, seed=7, distribution=dist, sampling_image=img,
                                       distribution_params={"gamma": 1.0, "floor": 1e-6})
            self.assertEqual(idx.size, 200)
            self.assertEqual(np.unique(idx).size, 200)  # no repeats
            self.assertTrue(np.all((idx >= 0) & (idx < SHAPE[0] * SHAPE[1])))

    def test_seed_determinism(self):
        a = sample_pixel_indices(SHAPE, 100, seed=42, distribution="uniform")
        b = sample_pixel_indices(SHAPE, 100, seed=42, distribution="uniform")
        c = sample_pixel_indices(SHAPE, 100, seed=43, distribution="uniform")
        np.testing.assert_array_equal(a, b)
        self.assertFalse(np.array_equal(a, c))

    def test_image_seed_deterministic(self):
        p = pathlib.Path("sub/00001.tif")
        self.assertEqual(derive_image_seed(12345, p), derive_image_seed(12345, p))
        self.assertNotEqual(derive_image_seed(12345, p), derive_image_seed(12345, pathlib.Path("sub/00002.tif")))


class TestChangeMask(unittest.TestCase):
    def test_roundtrip(self):
        idx = np.array([0, 5, 23, 100])
        mask = reconstruct_change_mask(SHAPE, idx)
        np.testing.assert_array_equal(np.flatnonzero(mask.reshape(-1)), np.sort(idx))


class TestOracleIdentity(unittest.TestCase):
    def test_expected_changes_matches_realized(self):
        # E[#changes] = sum(p); realized count must fall within a few sigma of it.
        cover = _synthetic_image(3).astype(np.uint8)
        alpha = 0.4
        p = change_probability_map(cover, "HUGO", alpha)
        self.assertTrue(np.all((p >= 0) & (p <= 1)))
        expected = float(p.sum())
        stego = simulate_embedding(cover, "HUGO", alpha, seed=derive_image_seed(12345, pathlib.Path("t.tif")))
        realized = int(np.count_nonzero(stego != cover))
        sigma = float(np.sqrt((p * (1.0 - p)).sum()))
        self.assertLess(abs(realized - expected), 5 * sigma + 1.0,
                        f"realized={realized} expected={expected:.1f} sigma={sigma:.1f}")


if __name__ == "__main__":
    unittest.main()
