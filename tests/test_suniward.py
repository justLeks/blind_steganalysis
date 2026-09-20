"""S-UNIWARD embedding and oracle: seed determinism, ±1 changes, E[#changes] = Σp.

Run: python3 -m unittest discover -s tests -v
"""

import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np

from embedding import SUPPORTED_METHODS, simulate_embedding
from selection_channel import change_probability_map


def _textured(seed=0, size=64):
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 256, size=(size, size), dtype=np.uint8)
    base[: size // 2, :] = 128  # flat half: wet-ish costs; noisy half: cheap costs
    return base


class TestSUNIWARD(unittest.TestCase):
    def test_registered(self):
        self.assertIn("SUNIWARD", SUPPORTED_METHODS)

    def test_seed_determinism_and_ternary_changes(self):
        img = _textured()
        s1 = simulate_embedding(img, "SUNIWARD", 0.2, seed=5)
        s2 = simulate_embedding(img, "SUNIWARD", 0.2, seed=5)
        s3 = simulate_embedding(img, "SUNIWARD", 0.2, seed=6)
        np.testing.assert_array_equal(s1, s2)
        self.assertFalse(np.array_equal(s1, s3))
        changed = int((s1 != img).sum())
        self.assertGreater(changed, 0)
        self.assertLessEqual(int(np.abs(s1.astype(np.int16) - img.astype(np.int16)).max()), 1)

    def test_oracle_expected_changes(self):
        img = _textured(seed=1, size=128)
        p = change_probability_map(img, "SUNIWARD", 0.2)
        self.assertEqual(p.shape, img.shape)
        self.assertTrue(np.all((p >= 0.0) & (p <= 1.0)))
        counts = [int((simulate_embedding(img, "SUNIWARD", 0.2, seed=s) != img).sum()) for s in range(6)]
        expected = float(p.sum())
        sd = float(np.sqrt((p * (1.0 - p)).sum()))
        self.assertLess(abs(float(np.mean(counts)) - expected), 4.0 * sd / np.sqrt(6) + 1.0)


if __name__ == "__main__":
    unittest.main()
